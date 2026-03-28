# Unsloth Qwen3.5 27B Local Chat GUI

Simple ChatGPT-style local chatbot UI using **Unsloth's 4-bit Qwen3.5 27B** model.

The app provides:

- Chat interface similar to ChatGPT/Gemini style (`st.chat_message` + `st.chat_input`)
- File upload support (text and PDF)
- Live, real-time token streaming while the model generates
- Local inference through `unsloth` + `bitsandbytes` 4-bit loading

## Model

- Default model: `unsloth/Qwen3.5-27B`
- Quantization: 4-bit enabled by default
- Memory note: this is still a large model and generally needs a CUDA GPU

Reference: https://unsloth.ai/docs/models/qwen3.5

## Quickstart

### 1) Python environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

### 2) Start the app

```bash
streamlit run app.py
```

Then open the URL shown in the terminal (usually `http://localhost:8501`).

## How to use

1. Set model options in the sidebar (or keep defaults).
2. Upload one or more files for context (text/PDF).
3. Ask questions in the chat input.
4. Watch output stream in real time with generated token count.

## Notes

- If no CUDA GPU is available, the app warns you at startup.
- PDF parsing uses `pypdf`.
- Long file contents are truncated to keep prompt sizes manageable.

## Main files

- `app.py` - Streamlit UI + Unsloth model loading + streaming inference
- `requirements.txt` - Python dependencies
