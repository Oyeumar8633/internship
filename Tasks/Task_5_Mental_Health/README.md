# Task 5 — Mental Health Support Chatbot (Fine-Tuned)

## Objective

Fine-tune a small language model (**DistilGPT2**) on **EmpatheticDialogues** so it learns to produce **warmer, more listener-style** text for stress and emotional wellness topics. This is an **educational** project demonstrating **supervised fine-tuning** with Hugging Face `Trainer` — not a clinical or therapeutic product.

## What’s done (Task 5 checklist)

| Item | Status |
|------|--------|
| Notebook `mental_health_finetuning.ipynb` — load **EmpatheticDialogues** (`revision="refs/convert/parquet"`), format text, tokenize, **`Trainer` + `TrainingArguments`**, **`DataCollatorForLanguageModeling(..., mlm=False)`** | Done |
| Train **cross-entropy** next-token loss; save checkpoint to **`mental_health_distilgpt2/`** | Done |
| `app.py` — **CLI** (`python app.py`) and **Streamlit** (`streamlit run app.py`), prompt template matches training (`### Situation:` / `### Supportive reply:`) | Done |
| Optional **`MENTAL_HEALTH_MODEL_DIR`** for a non-default model path | Done |
| Fine-tune in **Colab** (GPU), zip/download **`mental_health_distilgpt2/`**, place **next to `app.py`** on a local machine | Done |
| **Local run verified** — model loads (`config.json`, `model.safetensors`, tokenizer files), Streamlit serves on **http://localhost:8501** | Done |
| Root **`requirements.txt`** + note: **`torch`** installed per platform; project **`.venv`** used for local runs | Done |
| Large weights **gitignored** — `Tasks/Task_5_Mental_Health/mental_health_distilgpt2/` in repo **`.gitignore`** | Done |

**Note:** Recent `transformers` versions may reject `overwrite_output_dir` in `TrainingArguments`; delete the output folder manually before re-running training if you need a clean save.

## Dataset

- **EmpatheticDialogues** (`facebook/empathetic_dialogues`) — multi-turn dialogues where listeners respond with empathy to a speaker’s situation.
- This repo’s notebook loads the Hub’s **parquet** revision (`revision="refs/convert/parquet"`) because older `datasets` script loaders are deprecated in recent `datasets` releases.

## Base model

- **`distilgpt2`** — compact GPT-2-style causal LM, suitable for teaching and faster iteration than full GPT-2.

## Fine-tuning process (summary)

1. **Formatting:** Each example is turned into a single string with a fixed template:
   - `### Situation:` + speaker context (`prompt` field)
   - `### Supportive reply:` + empathetic listener (`utterance` field)
2. **Tokenization:** `text` is truncated/padded to `max_length=256` with DistilGPT2’s tokenizer.
3. **Training:** `Trainer` with `DataCollatorForLanguageModeling(..., mlm=False)` trains **causal** next-token prediction (not masked LM).
4. **Loss function:** **Cross-entropy** (negative log-likelihood) over next-token predictions at each position — the standard objective for causal LMs. Minimizing it increases the probability of **empathetic** token sequences seen in the dataset.
5. **Saving:** Weights + tokenizer are written to `./mental_health_distilgpt2/` (used by `app.py`).

## Hardware requirements

- **GPU strongly recommended** (e.g. NVIDIA with CUDA, or Google Colab T4): larger batch sizes, mixed precision (`fp16=True` when CUDA is available), much shorter training time.
- **CPU** is possible but slower; reduce `MAX_TRAIN_SAMPLES`, `per_device_train_batch_size`, and epochs if you hit memory or time limits.

## Reference training run (Colab example)

Exact numbers depend on your runtime settings. One successful run looked roughly like:

- **~1 epoch**, order of **minutes** wall time on a small GPU sample configuration.
- **Train loss** in the **~2.6** range (cross-entropy; lower is better, scale depends on sequence length and masking).

Loading a checkpoint may log a message about **`lm_head.weight`** for DistilGPT2; this is often **benign**. If generation looks odd, tune **`max_new_tokens`**, **temperature**, **top_p**, and **repetition** settings in `app.py`.

## Key results — how the fine-tuned model differs in tone from base DistilGPT2

| Aspect | Base `distilgpt2` | After fine-tuning on EmpatheticDialogues |
|--------|-------------------|------------------------------------------|
| **Training signal** | General web text (broad, generic style) | **Listener-style** replies paired with emotional situations |
| **Typical tone** | Often generic or “continues prose” | **More aligned** with supportive, validating phrasing *in the dataset’s style* (imitation learning) |
| **Limitations** | No guarantee of safety or clinical quality | Still **not** a therapist; can hallucinate, **repeat phrases**, or mishandle crises — use only as a **demo** |

**Important:** Fine-tuning does **not** add a special “empathy metric” — improvement comes from **maximum-likelihood** training on human-written empathetic text. Always add **human review** and **safety policies** for any real deployment.

## Files

| File | Role |
|------|------|
| `mental_health_finetuning.ipynb` | Load data, fine-tune, save `mental_health_distilgpt2/` |
| `app.py` | **CLI** (`python app.py`) or **Streamlit** (`streamlit run app.py`) |
| `mental_health_distilgpt2/` | **Saved model** (not committed — large). Create by training or copy from Colab. |
| `README.md` | This document |

## Setup

### Dependencies

From the **repository root** (recommended: use a virtual environment, e.g. `.venv`):

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install torch           # PyTorch — use pytorch.org if you need a specific CUDA build
```

Core packages include **`datasets`**, **`transformers`**, **`accelerate`**, **`evaluate`**, **`safetensors`**, **`streamlit`**, etc. **`torch`** is called out separately in the root `requirements.txt` because installs vary by machine.

### Model folder

After training (local or Colab), ensure **`mental_health_distilgpt2/`** sits **in this directory** next to `app.py`, containing at least:

- `config.json`, `model.safetensors` (or `pytorch_model.bin`), tokenizer files (`tokenizer.json` / vocab pieces as saved).

If the model lives elsewhere:

```bash
export MENTAL_HEALTH_MODEL_DIR="/path/to/mental_health_distilgpt2"
```

### Run the app

```bash
cd Tasks/Task_5_Mental_Health
python app.py              # terminal CLI
```

Or:

```bash
cd Tasks/Task_5_Mental_Health
streamlit run app.py       # browser UI — default http://localhost:8501
```

## Google Colab + optional Streamlit + ngrok

### Will this let me use the Streamlit UI “directly from Colab”?

**Yes, with one clarification:** Colab does not embed Streamlit inside the notebook cell output. Instead, you run Streamlit on the Colab VM and use **ngrok** to get a **public HTTPS link**. Opening that link in your browser shows the **full Streamlit UI** (same buttons, text areas, and layout as on your laptop)—it just loads in a **new tab**, not inside the Colab notebook page.

You need a **free [ngrok](https://ngrok.com/) account** to obtain an **authtoken** (required for stable tunnels on the free tier).

### Copy-paste cells (run after training and saving `mental_health_distilgpt2/`)

**Cell 1 — install dependencies**

```python
!pip install -q streamlit pyngrok
```

**Cell 2 — go to the folder that contains `app.py`**

Adjust the path if your project is not mounted at `/content/Internship`:

```python
import os
os.chdir("/content/Internship/Tasks/Task_5_Mental_Health")  # change if needed
```

**Cell 3 — (one-time) set your ngrok authtoken**

Sign in at [ngrok dashboard → Your Authtoken](https://dashboard.ngrok.com/get-started/your-authtoken), then paste the token below:

```python
from pyngrok import ngrok, conf
conf.get_default().auth_token = "YOUR_NGROK_AUTHTOKEN_HERE"
```

**Cell 4 — start Streamlit in the background, then expose port 8501**

```python
import subprocess
import sys
import time
from pyngrok import ngrok

# Start Streamlit (headless) on port 8501
subprocess.Popen(
    [sys.executable, "-m", "streamlit", "run", "app.py", "--server.port", "8501", "--server.headless", "true"],
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
)
time.sleep(5)  # give Streamlit time to bind

public_url = ngrok.connect(8501)
print("Open this URL in your browser for the Streamlit UI:")
print(public_url)
```

When you open the printed URL, you may see ngrok’s **warning page** once; click **Visit Site** to reach Streamlit.

**Stopping:** disconnect the session or run `ngrok.kill()` when finished. Tunnels expire when the Colab runtime shuts down.

### If ngrok does not work

- Some networks or Colab sessions block tunnels; try again later or use **`python app.py`** in Colab’s **terminal** (text-only, no GUI).
- You can also download the `mental_health_distilgpt2` folder and run `streamlit run app.py` on your **local** machine.

## Limitations (read before demo or submission)

- **Not safe for crises:** The model may produce **nonsensical**, **repetitive**, or **harmful-sounding** text. It is **not** a substitute for professional help or crisis services.
- **Educational scope:** Treat outputs as **illustrations of fine-tuning**, not advice.
- **Improvements** (optional): tighter generation (repetition penalty, shorter `max_new_tokens`), UI disclaimers, and **never** relying on the LM alone for self-harm or emergency content — use fixed crisis resources in real products.

## Disclaimer

This software is for **learning and research** only. It does **not** provide medical advice, diagnosis, or therapy. If you or someone else is in crisis, contact local emergency services or a qualified mental health professional.
