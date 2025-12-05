from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Any, Union
import httpx
import os

app = FastAPI(
    title="Universal API",
    description="A universal API built with FastAPI on Vercel - Anthropic Compatible",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== Base Endpoints ====================

@app.get("/")
def read_root():
    return {"message": "Welcome to Universal API", "Python": "on Vercel"}


@app.get("/health")
def health_check():
    return {"status": "healthy"}


# ==================== Anthropic Compatible API ====================

ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"

# Request Models
class TextContent(BaseModel):
    type: str = "text"
    text: str


class ImageSource(BaseModel):
    type: str = "base64"
    media_type: str
    data: str


class ImageContent(BaseModel):
    type: str = "image"
    source: ImageSource


class ToolUseContent(BaseModel):
    type: str = "tool_use"
    id: str
    name: str
    input: dict


class ToolResultContent(BaseModel):
    type: str = "tool_result"
    tool_use_id: str
    content: Union[str, List[Any]]


class ThinkingContent(BaseModel):
    type: str = "thinking"
    thinking: str


ContentBlock = Union[TextContent, ImageContent, ToolUseContent, ToolResultContent, ThinkingContent, dict]


class Message(BaseModel):
    role: str
    content: Union[str, List[ContentBlock]]


class Tool(BaseModel):
    name: str
    description: Optional[str] = None
    input_schema: dict


class ToolChoice(BaseModel):
    type: str = "auto"
    name: Optional[str] = None


class ThinkingConfig(BaseModel):
    type: str = "enabled"
    budget_tokens: Optional[int] = None


class MessagesRequest(BaseModel):
    model: str
    messages: List[Message]
    max_tokens: int = 4096
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    stop_sequences: Optional[List[str]] = None
    stream: Optional[bool] = False
    system: Optional[Union[str, List[dict]]] = None
    tools: Optional[List[Tool]] = None
    tool_choice: Optional[ToolChoice] = None
    thinking: Optional[ThinkingConfig] = None
    metadata: Optional[dict] = None


# Response Models
class Usage(BaseModel):
    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: Optional[int] = None
    cache_read_input_tokens: Optional[int] = None


class MessagesResponse(BaseModel):
    id: str
    type: str = "message"
    role: str = "assistant"
    content: List[Any]
    model: str
    stop_reason: Optional[str] = None
    stop_sequence: Optional[str] = None
    usage: Usage


class ConversationManager:
    """
    Helper class for managing multi-turn conversations with tool use.
    
    IMPORTANT: In multi-turn function call conversations, the complete model response
    (assistant message) must be appended to the conversation history to maintain
    the continuity of the reasoning chain.
    
    Usage:
        manager = ConversationManager()
        
        # Initial request
        response = await manager.send_message(client, request)
        
        # For tool use, append FULL response and continue
        if response.stop_reason == "tool_use":
            # The full response.content (including thinking/text/tool_use blocks)
            # is automatically appended to maintain reasoning chain continuity
            tool_results = execute_tools(response.content)
            response = await manager.continue_with_tool_results(client, tool_results)
    """
    
    def __init__(self):
        self.messages: List[Message] = []
        self.system: Optional[Union[str, List[dict]]] = None
        self.tools: Optional[List[Tool]] = None
    
    def append_assistant_response(self, response_content: List[Any]):
        """
        Append the full response.content list to message history.
        This includes ALL content blocks: thinking, text, and tool_use.
        
        CRITICAL: Do not filter or modify the content blocks - append them exactly
        as received to maintain the reasoning chain for multi-turn tool use.
        """
        self.messages.append(Message(
            role="assistant",
            content=response_content  # Full content list with all blocks
        ))
    
    def append_tool_results(self, tool_results: List[ToolResultContent]):
        """
        Append tool results as a user message to continue the conversation.
        """
        self.messages.append(Message(
            role="user",
            content=[result.model_dump() for result in tool_results]
        ))


@app.post("/anthropic/v1/messages", response_model=MessagesResponse)
async def create_message(
    request: MessagesRequest,
    x_api_key: Optional[str] = Header(None, alias="x-api-key"),
    authorization: Optional[str] = Header(None),
):
    """
    Anthropic-compatible messages endpoint.
    
    Supports:
    - Text generation
    - Vision (images)
    - Tool use / Function calling
    - Extended thinking
    - Streaming (passthrough)
    
    Multi-turn Tool Use:
    When handling tool_use responses, append the COMPLETE response.content 
    (including all thinking/text/tool_use blocks) to your message history
    before adding tool results and making the next request.
    
    Example flow:
    1. Send initial request with tools
    2. Receive response with stop_reason="tool_use"
    3. Append FULL response.content to messages as assistant message
    4. Execute tools and create tool_result content blocks
    5. Append tool_results as user message
    6. Send next request with updated messages
    """
    # Get API key from header or environment
    api_key = x_api_key or (authorization.replace("Bearer ", "") if authorization else None)
    if not api_key:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
    
    if not api_key:
        raise HTTPException(
            status_code=401,
            detail="API key required. Provide via x-api-key header or ANTHROPIC_API_KEY env var"
        )
    
    # Build request payload
    payload = {
        "model": request.model,
        "messages": [msg.model_dump() for msg in request.messages],
        "max_tokens": request.max_tokens,
    }
    
    # Add optional parameters
    if request.temperature is not None:
        payload["temperature"] = request.temperature
    if request.top_p is not None:
        payload["top_p"] = request.top_p
    if request.top_k is not None:
        payload["top_k"] = request.top_k
    if request.stop_sequences:
        payload["stop_sequences"] = request.stop_sequences
    if request.stream:
        payload["stream"] = request.stream
    if request.system:
        payload["system"] = request.system
    if request.tools:
        payload["tools"] = [tool.model_dump() for tool in request.tools]
    if request.tool_choice:
        payload["tool_choice"] = request.tool_choice.model_dump()
    if request.thinking:
        payload["thinking"] = request.thinking.model_dump()
    if request.metadata:
        payload["metadata"] = request.metadata
    
    headers = {
        "Content-Type": "application/json",
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
    }
    
    # Add beta header for extended thinking if enabled
    if request.thinking:
        headers["anthropic-beta"] = "interleaved-thinking-2025-05-14"
    
    async with httpx.AsyncClient(timeout=300.0) as client:
        try:
            response = await client.post(
                ANTHROPIC_API_URL,
                json=payload,
                headers=headers,
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            raise HTTPException(
                status_code=e.response.status_code,
                detail=e.response.json() if e.response.content else str(e)
            )
        except httpx.RequestError as e:
            raise HTTPException(status_code=502, detail=f"Upstream request failed: {str(e)}")


@app.get("/anthropic/v1/models")
async def list_models():
    """List available Anthropic models."""
    return {
        "models": [
            {"id": "claude-sonnet-4-20250514", "name": "Claude Sonnet 4"},
            {"id": "claude-3-5-sonnet-20241022", "name": "Claude 3.5 Sonnet"},
            {"id": "claude-3-5-haiku-20241022", "name": "Claude 3.5 Haiku"},
            {"id": "claude-3-opus-20240229", "name": "Claude 3 Opus"},
            {"id": "claude-3-sonnet-20240229", "name": "Claude 3 Sonnet"},
            {"id": "claude-3-haiku-20240307", "name": "Claude 3 Haiku"},
        ]
    }


# ==================== Multi-Turn Tool Use Example ====================

MULTI_TURN_EXAMPLE = """
# Multi-Turn Function Call Example

In multi-turn function call conversations, you MUST append the complete model response
to maintain the continuity of the reasoning chain.

```python
import httpx

async def multi_turn_tool_use():
    messages = [
        {"role": "user", "content": "What's the weather in Paris and Tokyo?"}
    ]
    
    tools = [{
        "name": "get_weather",
        "description": "Get weather for a city",
        "input_schema": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"]
        }
    }]
    
    async with httpx.AsyncClient() as client:
        # First request
        response = await client.post(
            "https://your-api/anthropic/v1/messages",
            headers={"x-api-key": "your-key"},
            json={
                "model": "claude-sonnet-4-20250514",
                "max_tokens": 4096,
                "messages": messages,
                "tools": tools,
                "tool_choice": {"type": "auto"}
            }
        )
        result = response.json()
        
        # CRITICAL: Append FULL response.content to messages
        # This includes ALL content blocks: thinking, text, AND tool_use
        messages.append({
            "role": "assistant",
            "content": result["content"]  # Full content list!
        })
        
        # Process tool calls and add results
        tool_results = []
        for block in result["content"]:
            if block["type"] == "tool_use":
                # Execute tool and get result
                tool_result = execute_tool(block["name"], block["input"])
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block["id"],
                    "content": tool_result
                })
        
        # Add tool results as user message
        messages.append({
            "role": "user", 
            "content": tool_results
        })
        
        # Continue conversation
        response = await client.post(
            "https://your-api/anthropic/v1/messages",
            headers={"x-api-key": "your-key"},
            json={
                "model": "claude-sonnet-4-20250514",
                "max_tokens": 4096,
                "messages": messages,
                "tools": tools
            }
        )
        return response.json()
```
"""


@app.get("/anthropic/v1/examples/multi-turn")
async def get_multi_turn_example():
    """Get example code for multi-turn tool use conversations."""
    return {"example": MULTI_TURN_EXAMPLE}
