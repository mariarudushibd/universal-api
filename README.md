# Universal API Gateway

Anthropic-compatible API with multi-provider backend (Groq, xAI). Deploy on Vercel.

**Live:** https://universal-api-gateway.vercel.app

## Supported Providers

| Provider | Models | Description |
|----------|--------|-------------|
| **Groq** | qwen-qwq-32b, llama-3.3-70b-versatile | Fast inference, high concurrency |
| **xAI** | grok-2-latest | Advanced reasoning |

## Model Mapping

| Anthropic Model | Backend | Provider |
|-----------------|---------|----------|
| claude-sonnet-4-20250514 | qwen-qwq-32b | Groq |
| claude-3-5-sonnet-20241022 | qwen-qwq-32b | Groq |
| claude-3-opus-20240229 | grok-2-latest | xAI |
| claude-3-5-haiku-20241022 | llama-3.3-70b-versatile | Groq |

## Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/anthropic/v1/messages` | POST | Anthropic-compatible messages |
| `/anthropic/v1/models` | GET | List available models |
| `/v1/chat/completions` | POST | OpenAI-compatible endpoint |

## Usage

### With Anthropic SDK

```python
import anthropic

client = anthropic.Anthropic(
    api_key="not-used",
    base_url="https://universal-api-gateway.vercel.app/anthropic"
)

response = client.messages.create(
    model="claude-sonnet-4-20250514",  # -> qwen-qwq-32b
    max_tokens=4096,
    messages=[{"role": "user", "content": "Hello!"}]
)
```

### With OpenAI SDK

```python
from openai import OpenAI

client = OpenAI(
    api_key="not-used",
    base_url="https://universal-api-gateway.vercel.app"
)

response = client.chat.completions.create(
    model="grok-2-latest",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

### Multi-Turn Tool Use

**IMPORTANT:** Append the FULL `response.content` list to maintain reasoning chain continuity.

```python
# First request with tools
response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=4096,
    messages=messages,
    tools=tools,
    tool_choice={"type": "auto"}
)

# CRITICAL: Append ALL content blocks (thinking/text/tool_use)
messages.append({
    "role": "assistant",
    "content": response.content  # Full content list!
})

# Add tool results
for block in response.content:
    if block.type == "tool_use":
        result = execute_tool(block.name, block.input)
        messages.append({
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": block.id, "content": result}]
        })

# Continue conversation
response = client.messages.create(model="claude-sonnet-4-20250514", max_tokens=4096, messages=messages)
```

## Environment Variables

```bash
GROQ_API_KEY=gsk_xxx      # Required for Groq models
XAI_API_KEY=xai-xxx       # Required for xAI/Grok models
```

## Deploy to Vercel

```bash
vercel deploy
```

Set environment variables in Vercel dashboard under Settings > Environment Variables.

## Local Development

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export GROQ_API_KEY=your-key
export XAI_API_KEY=your-key
vercel dev
```
