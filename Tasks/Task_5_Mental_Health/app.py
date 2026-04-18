#!/usr/bin/env python3
"""
Mental Health Support Chatbot — interact with the fine-tuned DistilGPT2 model.

- Terminal:  python app.py
- Streamlit: streamlit run app.py
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODEL_DIR = Path(__file__).resolve().parent / "mental_health_distilgpt2"

DISCLAIMER = (
    "This is an AI demo for learning purposes only. It is not therapy or medical advice. "
    "If you are in crisis, contact a professional or local emergency services."
)


def load_model_and_tokenizer(model_dir: Path):
    if not model_dir.is_dir():
        raise FileNotFoundError(
            f"Fine-tuned model folder not found: {model_dir}\n"
            "Run mental_health_finetuning.ipynb first (or set MENTAL_HEALTH_MODEL_DIR)."
        )
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model, tokenizer, device


def build_prompt(user_text: str) -> str:
    """Must match the template used during fine-tuning in the notebook."""
    return f"### Situation:\n{user_text.strip()}\n### Supportive reply:\n"


def generate_reply(
    model,
    tokenizer,
    device: torch.device,
    user_text: str,
    *,
    max_new_tokens: int = 120,
    temperature: float = 0.75,
    top_p: float = 0.92,
) -> str:
    prompt = build_prompt(user_text)
    enc = tokenizer(prompt, return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=pad_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    new_tokens = out[0, enc["input_ids"].shape[1] :]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    if "\n" in text:
        text = text.split("\n")[0].strip()
    return text


def run_cli(model_dir: Path) -> None:
    model, tokenizer, device = load_model_and_tokenizer(model_dir)
    print(DISCLAIMER)
    print("Type your message (or 'quit' to exit).\n")
    while True:
        try:
            user = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break
        if not user:
            continue
        if user.lower() in {"quit", "exit", "q"}:
            break
        reply = generate_reply(model, tokenizer, device, user)
        print(f"Assistant: {reply}\n")


def run_streamlit(model_dir: Path) -> None:
    import streamlit as st

    st.set_page_config(page_title="Mental Health Support (Demo)", layout="centered")
    st.title("Mental Health Support Chatbot (fine-tuned demo)")
    st.caption(DISCLAIMER)

    md = str(model_dir)
    if "model" not in st.session_state:
        with st.spinner("Loading model…"):
            m, tok, dev = load_model_and_tokenizer(Path(md))
            st.session_state["model"] = m
            st.session_state["tokenizer"] = tok
            st.session_state["device"] = dev

    user_text = st.text_area(
        "Share what you're going through (stress, worry, sadness, etc.)",
        height=120,
        placeholder="e.g. I've been overwhelmed with work and can't sleep.",
    )
    if st.button("Get a supportive response"):
        if not user_text.strip():
            st.warning("Please enter a message.")
        else:
            with st.spinner("Thinking…"):
                reply = generate_reply(
                    st.session_state["model"],
                    st.session_state["tokenizer"],
                    st.session_state["device"],
                    user_text,
                )
            st.success(reply)


def _running_inside_streamlit() -> bool:
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        return get_script_run_ctx() is not None
    except ImportError:
        return False


if __name__ == "__main__":
    model_dir = Path(os.environ.get("MENTAL_HEALTH_MODEL_DIR", DEFAULT_MODEL_DIR))
    if _running_inside_streamlit():
        run_streamlit(model_dir)
    else:
        run_cli(model_dir)
