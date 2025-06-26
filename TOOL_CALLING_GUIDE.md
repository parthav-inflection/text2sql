# Tool Calling Guide for Text2SQL

This guide explains the elegant tool calling system that provides a clean alternative to traditional SQL parsing from LLM responses.

## Overview

Our tool calling implementation follows clean architecture principles with clear separation of concerns, making it both powerful and maintainable.

### Key Benefits

1. **🎯 Clean Architecture**: Separate parsers, prompt generators, and model interfaces
2. **🔄 Automatic Fallback**: Tool calling falls back to traditional parsing seamlessly  
3. **🧩 Highly Extensible**: Easy to add new parsing strategies or prompt formats
4. **⚡ Zero Breaking Changes**: Existing code continues to work unchanged
5. **🔍 Better Reliability**: Structured output reduces parsing errors significantly

## Configuration

### Enable Tool Calling in Model Config

Add `enable_tool_calling: true` to your model configuration:

```yaml
name: "OmniSQL-7B-ToolCall"
model_path: "seeklhy/OmniSQL-7B"
model_type: "vllm"
enable_tool_calling: true
enable_sqlfluff_formatting: false  # Optional: disable since tool calling provides structured output
```

### Backward Compatibility

The system maintains full backward compatibility:
- Models with `enable_tool_calling: false` (or not set) use traditional parsing
- Existing configurations continue to work unchanged
- Tool calling falls back to traditional parsing if tool extraction fails

## Usage Examples

### 1. Simple Model Usage

```python
from src.models.factory import ModelFactory

# Everything is unified - just set enable_tool_calling: true in config
model = ModelFactory.create_model("configs/models/omnisql_toolcall.yaml")
sql, prompt = model.generate_sql("What is the total revenue?", schema)
```

### 2. Custom Parser Extension

```python
from src.models.response_parsers import ResponseParser, ChainedParser

class CustomParser(ResponseParser):
    def parse(self, response: str) -> str:
        # Your custom parsing logic here
        return extracted_sql

# Use with automatic fallback
parser = ChainedParser([
    CustomParser(),
    ToolCallParser(), 
    TraditionalParser()
])
```

### 3. Testing Different Approaches

```bash
# Run the elegant test suite
python scripts/test_tool_calling.py

# Run comparison evaluation
python scripts/run_eval.py --experiment configs/experiments/tool_calling_comparison.yaml
```

## Testing

### Quick Test

```bash
python scripts/test_tool_calling.py
```

### CPU Test (if using CPU inference)

```bash
python scripts/test_cpu_setup.py --agent-config configs/agents/standard_agent.yaml --model-config configs/models/omnisql_toolcall.yaml
```

## Tool Calling Format

The system uses a standardized tool calling format:

### Tool Definition

```json
{
  "type": "function",
  "function": {
    "name": "generate_sql",
    "description": "Generate a SQL query to answer the user's question based on the provided database schema.",
    "parameters": {
      "type": "object",
      "properties": {
        "sql_query": {
          "type": "string",
          "description": "The SQL query that answers the question. Should be valid SQL syntax."
        },
        "explanation": {
          "type": "string",
          "description": "Brief explanation of how the query answers the question."
        }
      },
      "required": ["sql_query"]
    }
  }
}
```

### Expected Response Format

```json
{
  "tool_calls": [{
    "type": "function",
    "function": {
      "name": "generate_sql",
      "arguments": {
        "sql_query": "SELECT SUM(revenue) FROM sales WHERE date >= '2024-01-01';",
        "explanation": "Sums total revenue from sales table for the current year"
      }
    }
  }]
}
```

## Implementation Details

### Parsing Hierarchy

1. **Tool Call Extraction**: Try to parse structured tool calls first
2. **Traditional Fallback**: If tool call parsing fails, fall back to regex-based extraction
3. **Error Handling**: Graceful degradation ensures system reliability

### Model Support

- **Chat Models**: Use chat templates with tool descriptions
- **Base Models**: Use specialized prompting with tool format instructions
- **All Models**: Maintain fallback to traditional parsing

## Migration Strategy

### Phase 1: Add Tool Calling Support (Current)
- ✅ Add tool calling interface to BaseModel
- ✅ Implement tool calling in VLLMModel
- ✅ Update evaluation framework
- ✅ Create test configurations

### Phase 2: Gradual Migration (Recommended)
1. Test tool calling with a subset of models
2. Compare performance with traditional parsing
3. Enable tool calling for well-performing models
4. Monitor and adjust

### Phase 3: Full Migration (Future)
1. Enable tool calling by default for new models
2. Migrate existing successful configurations
3. Deprecate traditional parsing (optional)

## Troubleshooting

### Common Issues

1. **Tool Call Not Detected**: Check model support and prompt formatting
2. **JSON Parsing Errors**: Verify model output format
3. **Performance Degradation**: Some models may perform better with traditional parsing

### Debugging

Enable debug logging to see tool calling behavior:

```python
import logging
logging.getLogger("src.models.vllm_model").setLevel(logging.DEBUG)
```

## Best Practices

1. **Test Before Migration**: Always test tool calling with your specific models
2. **Monitor Performance**: Compare accuracy metrics between approaches
3. **Use Fallbacks**: Keep traditional parsing as backup
4. **Validate Outputs**: Tool calling doesn't guarantee correct SQL, just better structure
5. **Model Selection**: Some models may be better suited for tool calling than others 