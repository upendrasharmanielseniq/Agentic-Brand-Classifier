# Prompt Classifier Agent

An AI-powered agent for classifying prompts using DSPy and ollama.

## Setup

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Copy `.env` and fill in your API keys

4. Configure ollama locally
   on mac run
   brew install ollama

   in another terminal(start ollama server locally and keep it running)
   ollama serve

   Now pull the model
   ollama pull phi3

## Environment Variables

- `DSPY_MODEL`: The DSPy model to use (default: ollama)
- `DSPY_MODEL_NAME`: The specific model name
- `SERPAPI_API_KEY`: API key for SerpAPI integration
