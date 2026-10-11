package scan

import (
	"encoding/json"
	"fmt"
	"net/http"
	"strings"

	cfgReader "github.com/americanexpress/earlybird/v4/pkg/config"
)

const (
	anthropicVersion   = "2023-06-01"
	anthropicMaxTokens = 16000
)

const anthropicFindingsSchema = `{
  "type": "object",
  "properties": {
    "findings": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "line": {"type": "integer"},
          "credential_type": {"type": "string"},
          "confidence": {"type": "string"},
          "candidate": {"type": "string"},
          "reason": {"type": "string"}
        },
        "required": ["line", "credential_type", "confidence", "reason"],
        "additionalProperties": false
      }
    }
  },
  "required": ["findings"],
  "additionalProperties": false
}`

type AnthropicProvider struct{}

func isAnthropicModel(model string) bool {
	return strings.Contains(strings.ToLower(model), "claude")
}

func (anthropic AnthropicProvider) setHeader(req *http.Request, value string) {
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("x-api-key", value)
	req.Header.Set("anthropic-version", anthropicVersion)
}

func (anthropic AnthropicProvider) buildRequestBody(cfg *cfgReader.EarlybirdConfig, scanJob LLMJob, chunk llmFileChunk) ([]byte, error) {
	requestBody := AnthropicRequestBody{
		Model:     cfg.LLMModel,
		MaxTokens: anthropicMaxTokens,
		System:    llmSystemPrompt(cfg),
		Messages: []anthropicMessage{
			{Role: "user", Content: buildLLMUserPrompt(scanJob, chunk)},
		},
		OutputConfig: anthropicOutputConfig{
			Format: anthropicOutputFormat{Type: "json_schema", Schema: json.RawMessage(anthropicFindingsSchema)},
		},
	}
	return json.Marshal(requestBody)
}

func (anthropic AnthropicProvider) parseResponse(responseBody []byte) ([]LLMFinding, error) {
	var message AnthropicResponseBody
	if err := json.Unmarshal(responseBody, &message); err != nil {
		return nil, err
	}
	if message.Error != nil {
		return nil, fmt.Errorf("anthropic response error: %s", message.Error.Message)
	}

	switch message.StopReason {
	case "refusal":
		category := ""
		if message.StopDetails != nil {
			category = message.StopDetails.Category
		}
		return nil, fmt.Errorf("anthropic declined the request (category %q)", category)
	case "max_tokens":
		return nil, fmt.Errorf("anthropic response was cut off at max_tokens")
	}

	// Thinking blocks come before the answer, so collect only the text blocks.
	var content strings.Builder
	for _, block := range message.Content {
		if block.Type == "text" {
			content.WriteString(block.Text)
		}
	}

	var structured llmStructuredResponse
	if err := json.Unmarshal([]byte(content.String()), &structured); err != nil {
		return nil, err
	}
	return structured.Findings, nil
}
