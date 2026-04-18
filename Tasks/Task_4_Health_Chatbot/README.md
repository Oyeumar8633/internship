# Task 4 — General Health Query Chatbot

## Objective

Build a **conversational agent** for general health-related questions using **prompt engineering**: a clear system persona, structured behavior, and guardrails so responses stay helpful without replacing professional medical care.

## Tools

- **Python** (3.10+ recommended)
- **LLM integration** (choose one):
  - **OpenAI API** — [`openai`](https://pypi.python.org/pypi/openai) package for chat completions
  - **Hugging Face Inference** — [`huggingface_hub`](https://pypi.python.org/pypi/huggingface_hub) for serverless text generation or chat models

See `health_chatbot.ipynb` for implementation details and how to switch providers.

## Key Features

| Feature | Description |
|--------|-------------|
| **Prompt-based persona** | System prompt instructs the model to act as a helpful, friendly, professional medical assistant while avoiding definitive diagnoses. |
| **Safety filtering** | A `safety_filter` scans user messages for high-risk terms (e.g. emergency, self-harm, severe symptoms). When triggered, a **mandatory disclaimer** is shown: the assistant is not a doctor and emergencies require local emergency services. |

## Environment variables (API keys)

**Never commit API keys.** Use a `.env` file (gitignored) or your shell profile.

### OpenAI

```bash
export OPENAI_API_KEY="sk-..."
```

Optional:

```bash
export OPENAI_MODEL="gpt-4o-mini"   # or your preferred chat model
```

### Hugging Face (`provider="huggingface"`)

The notebook **does not** use the Hugging Face Inference API / router (those often return **“model not supported”** or **403**). Instead it loads a **small instruct model locally** with **`transformers`** (good for **Google Colab**: enable a **GPU** runtime for speed).

```bash
# Optional: only needed for gated/private models or higher Hub rate limits
export HF_TOKEN="hf_..."
```

Model (default is small and public):

```bash
export HF_MODEL="HuggingFaceTB/SmolLM2-360M-Instruct"
```

First run **downloads** the model weights (can take a few minutes). The prompt format matches **SmolLM2**; other models may need a different chat template.

### Loading from a `.env` file

```bash
pip install python-dotenv
```

Create `.env` in the project root (do not commit):

```
OPENAI_API_KEY=sk-...
```

The notebook documents `python-dotenv` usage.

### Google Colab

**Do not paste tokens into notebook cells.** Use **Colab Secrets**:

1. Sidebar → **Secrets** (key icon) → add **`HF_TOKEN`** with your Hugging Face token.
2. Enable **Notebook access** for that secret.
3. Run the **“Google Colab — Hugging Face token”** code cell in `health_chatbot.ipynb` (it installs packages and loads `HF_TOKEN` from Secrets).

Then run the chat loop with `run_health_chat_loop(provider="huggingface")`.

### Troubleshooting: Hugging Face

- **Install deps:** `pip install transformers accelerate torch safetensors` (Colab: the notebook’s Colab cell installs these).
- **Out of memory:** use a smaller `HF_MODEL` or enable **GPU** in Colab (**Runtime → Change runtime type**).
- **For a stable cloud API without local weights:** use **`provider="openai"`** and set `OPENAI_API_KEY` instead.

## How to run

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install openai huggingface_hub python-dotenv
jupyter notebook
```

Open: `Tasks/Task_4_Health_Chatbot/health_chatbot.ipynb`

Run all cells, set your API key, then use the chat loop in the final section.

## Disclaimer

This project is for **education and demonstration** only. It does not provide medical advice, diagnosis, or treatment. Always consult qualified healthcare professionals for health decisions.
