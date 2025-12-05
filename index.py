from fastapi import FastAPI, HTTPException, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Any, Union
import httpx
import os
import json
import uuid

app = FastAPI(
    title="Universal API",
    description="Universal API Gateway - Anthropic Compatible with Multi-Provider Backend (Groq, xAI)",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== Provider Configuration ====================

PROVIDERS = {
    "groq": {
        "url": "https://api.groq.com/openai/v1/chat/completions",
        "env_key": "GROQ_API_KEY",
    },
    "xai": {
        "url": "https://api.x.ai/v1/chat/completions",
        "env_key": "XAI_API_KEY",
    },
}

# Model mapping: model name -> (provider, actual_model)
MODEL_MAPPING = {
    # Claude models -> Groq (qwen-qwq-32b for reasoning)
    "claude-sonnet-4-20250514": ("groq", "qwen-qwq-32b"),
    "claude-3-5-sonnet-20241022": ("groq", "qwen-qwq-32b"),
    "claude-3-5-haiku-20241022": ("groq", "llama-3.3-70b-versatile"),
    "claude-3-opus-20240229": ("xai", "grok-2-latest"),
    "claude-3-sonnet-20240229": ("groq", "qwen-qwq-32b"),
    "claude-3-haiku-20240307": ("groq", "llama-3.3-70b-versatile"),
    
    # Direct Groq models
    "qwen-qwq-32b": ("groq", "qwen-qwq-32b"),
    "llama-3.3-70b-versatile": ("groq", "llama-3.3-70b-versatile"),
    "llama-3.1-8b-instant": ("groq", "llama-3.1-8b-instant"),
    "mixtral-8x7b-32768": ("groq", "mixtral-8x7b-32768"),
    "gemma2-9b-it": ("groq", "gemma2-9b-it"),
    
    # xAI Grok models
    "grok-2-latest": ("xai", "grok-2-latest"),
    "grok-2": ("xai", "grok-2-latest"),
    "grok-beta": ("xai", "grok-beta"),
}

# ==================== Base Endpoints ====================

@app.get("/")
def read_root():
    return {
        "message": "Universal API Gateway",
        "description": "Anthropic-compatible API with multi-provider backend",
        "providers": ["groq", "xai"],
        "docs": "/docs",
        "endpoints": {
            "messages": "/anthropic/v1/messages",
            "models": "/anthropic/v1/models"
        }
    }


@app.get("/health")
def health_check():
    return {"status": "healthy"}


# ==================== Request/Response Models ====================

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
    temperature: Optional[float] = 0.6
    top_p: Optional[float] = 0.95
    top_k: Optional[int] = None
    stop_sequences: Optional[List[str]] = None
    stream: Optional[bool] = False
    system: Optional[Union[str, List[dict]]] = None
    tools: Optional[List[Tool]] = None
    tool_choice: Optional[ToolChoice] = None
    thinking: Optional[ThinkingConfig] = None
    metadata: Optional[dict] = None


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


# ==================== Conversion Functions ====================

def convert_anthropic_to_openai_messages(
    messages: List[Message],
    system: Optional[Union[str, List[dict]]] = None
) -> List[dict]:
    """Convert Anthropic message format to OpenAI format."""
    openai_messages = []
    
    # Add system message if present
    if system:
        if isinstance(system, str):
            openai_messages.append({"role": "system", "content": system})
        elif isinstance(system, list):
            system_text = "\n".join([
                block.get("text", "") for block in system if block.get("type") == "text"
            ])
            if system_text:
                openai_messages.append({"role": "system", "content": system_text})
    
    for msg in messages:
        role = msg.role
        content = msg.content
        
        if isinstance(content, str):
            openai_messages.append({"role": role, "content": content})
        elif isinstance(content, list):
            # Handle complex content blocks
            text_parts = []
            tool_calls = []
            tool_results = []
            
            for block in content:
                if isinstance(block, dict):
                    block_type = block.get("type", "")
                    
                    if block_type == "text":
                        text_parts.append(block.get("text", ""))
                    elif block_type == "thinking":
                        # Include thinking in the message
                        text_parts.append(f"<thinking>{block.get('thinking', '')}</thinking>")
                    elif block_type == "tool_use":
                        tool_calls.append({
                            "id": block.get("id", ""),
                            "type": "function",
                            "function": {
                                "name": block.get("name", ""),
                                "arguments": json.dumps(block.get("input", {}))
                            }
                        })
                    elif block_type == "tool_result":
                        tool_result_content = block.get("content", "")
                        if isinstance(tool_result_content, list):
                            tool_result_content = json.dumps(tool_result_content)
                        tool_results.append({
                            "role": "tool",
                            "tool_call_id": block.get("tool_use_id", ""),
                            "content": str(tool_result_content)
                        })
                elif hasattr(block, "type"):
                    if block.type == "text":
                        text_parts.append(block.text)
            
            # Build the message
            if role == "assistant" and tool_calls:
                msg_dict = {"role": "assistant", "content": "\n".join(text_parts) if text_parts else None}
                msg_dict["tool_calls"] = tool_calls
                openai_messages.append(msg_dict)
            elif role == "user" and tool_results:
                for tr in tool_results:
                    openai_messages.append(tr)
            elif text_parts:
                openai_messages.append({"role": role, "content": "\n".join(text_parts)})
    
    return openai_messages


def convert_anthropic_tools_to_openai(tools: Optional[List[Tool]]) -> Optional[List[dict]]:
    """Convert Anthropic tools format to OpenAI format."""
    if not tools:
        return None
    
    openai_tools = []
    for tool in tools:
        openai_tools.append({
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description or "",
                "parameters": tool.input_schema
            }
        })
    return openai_tools


def convert_openai_response_to_anthropic(
    openai_response: dict,
    original_model: str
) -> dict:
    """Convert OpenAI response to Anthropic format."""
    choice = openai_response.get("choices", [{}])[0]
    message = choice.get("message", {})
    
    content_blocks = []
    
    # Add text content
    if message.get("content"):
        content_blocks.append({
            "type": "text",
            "text": message["content"]
        })
    
    # Add tool calls
    if message.get("tool_calls"):
        for tc in message["tool_calls"]:
            func = tc.get("function", {})
            try:
                tool_input = json.loads(func.get("arguments", "{}"))
            except json.JSONDecodeError:
                tool_input = {"raw": func.get("arguments", "")}
            
            content_blocks.append({
                "type": "tool_use",
                "id": tc.get("id", f"toolu_{uuid.uuid4().hex[:24]}"),
                "name": func.get("name", ""),
                "input": tool_input
            })
    
    # Determine stop reason
    finish_reason = choice.get("finish_reason", "end_turn")
    stop_reason_map = {
        "stop": "end_turn",
        "length": "max_tokens",
        "tool_calls": "tool_use",
        "content_filter": "end_turn"
    }
    stop_reason = stop_reason_map.get(finish_reason, "end_turn")
    
    if message.get("tool_calls"):
        stop_reason = "tool_use"
    
    usage = openai_response.get("usage", {})
    
    return {
        "id": f"msg_{openai_response.get('id', uuid.uuid4().hex)}",
        "type": "message",
        "role": "assistant",
        "content": content_blocks,
        "model": original_model,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0)
        }
    }


# ==================== Streaming ====================

async def stream_to_anthropic(response: httpx.Response, original_model: str):
    """Convert streaming response to Anthropic SSE format."""
    msg_id = f"msg_{uuid.uuid4().hex}"
    
    # Send message_start event
    yield f"event: message_start\ndata: {json.dumps({'type': 'message_start', 'message': {'id': msg_id, 'type': 'message', 'role': 'assistant', 'content': [], 'model': original_model, 'stop_reason': None, 'stop_sequence': None, 'usage': {'input_tokens': 0, 'output_tokens': 0}}})}\n\n"
    
    # Send content_block_start
    yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': 0, 'content_block': {'type': 'text', 'text': ''}})}\n\n"
    
    collected_content = ""
    async for line in response.aiter_lines():
        if line.startswith("data: "):
            data = line[6:]
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
                delta = chunk.get("choices", [{}])[0].get("delta", {})
                content = delta.get("content", "")
                if content:
                    collected_content += content
                    yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': content}})}\n\n"
            except json.JSONDecodeError:
                continue
    
    # Send content_block_stop
    yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n"
    
    # Send message_delta with stop reason
    yield f"event: message_delta\ndata: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': 'end_turn', 'stop_sequence': None}, 'usage': {'output_tokens': len(collected_content.split())}})}\n\n"
    
    # Send message_stop
    yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"


# ==================== Main Endpoint ====================

@app.post("/anthropic/v1/messages")
async def create_message(
    request: MessagesRequest,
    x_api_key: Optional[str] = Header(None, alias="x-api-key"),
    authorization: Optional[str] = Header(None),
):
    """
    Anthropic-compatible messages endpoint with multi-provider backend.
    
    Providers: Groq (qwen-qwq-32b, llama) and xAI (grok-2)
    
    Supports:
    - Text generation
    - Tool use / Function calling (auto mode)
    - Streaming
    
    Multi-turn Tool Use:
    Append the COMPLETE response.content (all thinking/text/tool_use blocks)
    to your message history to maintain reasoning chain continuity.
    """
    # Map model to provider and actual model name
    original_model = request.model
    
    if request.model in MODEL_MAPPING:
        provider_name, backend_model = MODEL_MAPPING[request.model]
    else:
        # Default to groq with qwen-qwq-32b
        provider_name = "groq"
        backend_model = "qwen-qwq-32b"
    
    provider = PROVIDERS[provider_name]
    
    # Get API key from environment
    api_key = os.environ.get(provider["env_key"])
    if not api_key:
        raise HTTPException(
            status_code=401,
            detail=f"API key required. Set {provider['env_key']} environment variable."
        )
    
    # Convert messages to OpenAI format
    openai_messages = convert_anthropic_to_openai_messages(
        request.messages,
        request.system
    )
    
    # Build request payload
    payload = {
        "model": backend_model,
        "messages": openai_messages,
        "temperature": request.temperature or 0.6,
        "top_p": request.top_p or 0.95,
        "stream": request.stream or False,
    }
    
    # Handle max tokens (different param names per provider)
    if provider_name == "groq":
        payload["max_completion_tokens"] = request.max_tokens
    else:
        payload["max_tokens"] = request.max_tokens
    
    # Add stop sequences
    if request.stop_sequences:
        payload["stop"] = request.stop_sequences
    
    # Add tools if present
    if request.tools:
        payload["tools"] = convert_anthropic_tools_to_openai(request.tools)
        if request.tool_choice:
            if request.tool_choice.type == "auto":
                payload["tool_choice"] = "auto"
            elif request.tool_choice.type == "any":
                payload["tool_choice"] = "required"
            elif request.tool_choice.type == "tool" and request.tool_choice.name:
                payload["tool_choice"] = {
                    "type": "function",
                    "function": {"name": request.tool_choice.name}
                }
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    
    async with httpx.AsyncClient(timeout=300.0) as client:
        try:
            if request.stream:
                # Streaming response
                async with client.stream(
                    "POST",
                    provider["url"],
                    json=payload,
                    headers=headers,
                ) as response:
                    response.raise_for_status()
                    return StreamingResponse(
                        stream_to_anthropic(response, original_model),
                        media_type="text/event-stream",
                        headers={
                            "Cache-Control": "no-cache",
                            "Connection": "keep-alive",
                        }
                    )
            else:
                # Non-streaming response
                response = await client.post(
                    provider["url"],
                    json=payload,
                    headers=headers,
                )
                response.raise_for_status()
                openai_response = response.json()
                
                # Convert to Anthropic format
                return convert_openai_response_to_anthropic(openai_response, original_model)
                
        except httpx.HTTPStatusError as e:
            error_detail = str(e)
            try:
                error_detail = e.response.json()
            except:
                pass
            raise HTTPException(
                status_code=e.response.status_code,
                detail=error_detail
            )
        except httpx.RequestError as e:
            raise HTTPException(status_code=502, detail=f"Upstream request failed: {str(e)}")


@app.get("/anthropic/v1/models")
async def list_models():
    """List available models with their descriptions and backend mapping."""
    return {
        "models": [
            {
                "id": "claude-sonnet-4-20250514",
                "name": "Claude Sonnet 4",
                "description": "Agentic capabilities, Advanced reasoning",
                "provider": "groq",
                "backend": "qwen-qwq-32b"
            },
            {
                "id": "claude-3-5-sonnet-20241022",
                "name": "Claude 3.5 Sonnet",
                "description": "High performance, balanced capabilities",
                "provider": "groq",
                "backend": "qwen-qwq-32b"
            },
            {
                "id": "claude-3-opus-20240229",
                "name": "Claude 3 Opus",
                "description": "Most capable, powered by Grok",
                "provider": "xai",
                "backend": "grok-2-latest"
            },
            {
                "id": "grok-2-latest",
                "name": "Grok 2",
                "description": "xAI's latest model, advanced reasoning",
                "provider": "xai",
                "backend": "grok-2-latest"
            },
            {
                "id": "qwen-qwq-32b",
                "name": "Qwen QWQ 32B",
                "description": "Advanced reasoning, high concurrency",
                "provider": "groq",
                "backend": "qwen-qwq-32b"
            },
            {
                "id": "llama-3.3-70b-versatile",
                "name": "Llama 3.3 70B",
                "description": "Versatile, commercial use",
                "provider": "groq",
                "backend": "llama-3.3-70b-versatile"
            },
        ]
    }


# ==================== Multi-Turn Example ====================

@app.get("/anthropic/v1/examples/multi-turn")
async def get_multi_turn_example():
    """Get example code for multi-turn tool use conversations."""
    return {
        "description": "Multi-turn function call example with full response appending",
        "important": "Append the FULL response.content list to maintain reasoning chain continuity",
        "example": '''
import anthropic

# Using Anthropic SDK with Universal API Gateway
client = anthropic.Anthropic(
    api_key="any-key",  # Not used, backend uses env vars
    base_url="https://universal-api-gateway.vercel.app/anthropic"
)

messages = [{"role": "user", "content": "What is the weather in Paris?"}]

tools = [{
    "name": "get_weather",
    "description": "Get weather for a city",
    "input_schema": {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"]
    }
}]

# First request
response = client.messages.create(
    model="claude-sonnet-4-20250514",  # -> groq/qwen-qwq-32b
    max_tokens=4096,
    messages=messages,
    tools=tools,
    tool_choice={"type": "auto"}
)

# CRITICAL: Append FULL response.content (ALL blocks: thinking/text/tool_use)
messages.append({
    "role": "assistant",
    "content": response.content  # Complete content list!
})

# Execute tools and add results
for block in response.content:
    if block.type == "tool_use":
        result = execute_tool(block.name, block.input)
        messages.append({
            "role": "user",
            "content": [{
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": result
            }]
        })

# Continue conversation with full context
response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=4096,
    messages=messages,
    tools=tools
)
'''
    }


# ==================== OpenAI Compatible Endpoint ====================

@app.post("/v1/chat/completions")
async def openai_chat_completions(
    request: Request,
    authorization: Optional[str] = Header(None),
):
    """
    OpenAI-compatible chat completions endpoint.
    Directly proxies to the appropriate backend based on model.
    """
    body = await request.json()
    model = body.get("model", "qwen-qwq-32b")
    
    # Determine provider
    if model in MODEL_MAPPING:
        provider_name, backend_model = MODEL_MAPPING[model]
    elif "grok" in model.lower():
        provider_name = "xai"
        backend_model = model
    else:
        provider_name = "groq"
        backend_model = model
    
    provider = PROVIDERS[provider_name]
    api_key = os.environ.get(provider["env_key"])
    
    if not api_key:
        raise HTTPException(
            status_code=401,
            detail=f"API key required. Set {provider['env_key']} environment variable."
        )
    
    body["model"] = backend_model
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    
    async with httpx.AsyncClient(timeout=300.0) as client:
        try:
            if body.get("stream"):
                async with client.stream(
                    "POST",
                    provider["url"],
                    json=body,
                    headers=headers,
                ) as response:
                    response.raise_for_status()
                    
                    async def stream_response():
                        async for line in response.aiter_lines():
                            yield line + "\n"
                    
                    return StreamingResponse(
                        stream_response(),
                        media_type="text/event-stream",
                    )
            else:
                response = await client.post(
                    provider["url"],
                    json=body,
                    headers=headers,
                )
                response.raise_for_status()
                return response.json()
        except httpx.HTTPStatusError as e:
            raise HTTPException(
                status_code=e.response.status_code,
                detail=e.response.json() if e.response.content else str(e)
            )
