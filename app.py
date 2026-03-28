import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import streamlit as st
import torch
from transformers import TextIteratorStreamer
from unsloth import FastLanguageModel

try:
    from pypdf import PdfReader
except Exception:  # pragma: no cover - optional dependency at runtime
    PdfReader = None


DEFAULT_MODEL_NAME = "unsloth/Qwen3.5-27B"
DEFAULT_MAX_SEQ_LENGTH = 8192
SUPPORTED_TEXT_EXTENSIONS = {
    ".txt",
    ".md",
    ".rst",
    ".py",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".csv",
    ".tsv",
    ".xml",
    ".html",
    ".css",
    ".js",
    ".ts",
    ".sql",
    ".log",
}


@dataclass
class ExtractedFile:
    name: str
    content: str


def _dtype_from_name(dtype_name: str) -> Optional[torch.dtype]:
    mapping = {
        "auto": None,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    return mapping[dtype_name]


@st.cache_resource(show_spinner=False)
def load_unsloth_model(
    model_name: str,
    max_seq_length: int,
    dtype_name: str,
    load_in_4bit: bool,
):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=_dtype_from_name(dtype_name),
        load_in_4bit=load_in_4bit,
    )
    FastLanguageModel.for_inference(model)
    return model, tokenizer


def _decode_text_bytes(raw: bytes) -> str:
    for encoding in ("utf-8", "utf-16", "latin-1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def extract_file_contents(uploaded_files: Iterable) -> tuple[list[ExtractedFile], list[str]]:
    extracted: list[ExtractedFile] = []
    warnings: list[str] = []

    for uploaded_file in uploaded_files:
        suffix = Path(uploaded_file.name).suffix.lower()
        raw = uploaded_file.getvalue()

        if uploaded_file.type.startswith("text/") or suffix in SUPPORTED_TEXT_EXTENSIONS:
            extracted.append(ExtractedFile(name=uploaded_file.name, content=_decode_text_bytes(raw)))
            continue

        if suffix == ".pdf":
            if PdfReader is None:
                warnings.append(
                    f"Skipped `{uploaded_file.name}`: install `pypdf` to read PDF files."
                )
                continue
            try:
                reader = PdfReader(uploaded_file)
                pages = [page.extract_text() or "" for page in reader.pages]
                extracted.append(ExtractedFile(name=uploaded_file.name, content="\n".join(pages)))
            except Exception as err:  # pragma: no cover - depends on user files
                warnings.append(f"Skipped `{uploaded_file.name}`: failed to read PDF ({err}).")
            continue

        warnings.append(
            f"Skipped `{uploaded_file.name}`: unsupported type `{uploaded_file.type or suffix}`."
        )
    return extracted, warnings


def build_file_context(
    extracted_files: list[ExtractedFile],
    max_chars_per_file: int = 8000,
    max_total_chars: int = 24000,
) -> str:
    if not extracted_files:
        return ""

    chunks: list[str] = []
    remaining = max_total_chars
    for file_info in extracted_files:
        if remaining <= 0:
            break
        cleaned = file_info.content.strip()
        if not cleaned:
            continue
        if len(cleaned) > max_chars_per_file:
            cleaned = cleaned[:max_chars_per_file] + "\n...[truncated]..."
        if len(cleaned) > remaining:
            cleaned = cleaned[:remaining] + "\n...[truncated by total budget]..."

        remaining -= len(cleaned)
        chunks.append(f"[File: {file_info.name}]\n{cleaned}")

    if not chunks:
        return ""
    return "\n\n".join(chunks)


def build_generation_prompt(
    system_prompt: str,
    chat_history: list[dict[str, str]],
    latest_user_prompt: str,
    file_context: str,
) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    if system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt.strip()})

    messages.extend(chat_history)
    if file_context:
        latest_user_prompt = (
            f"{latest_user_prompt}\n\n"
            "Use the attached file contents as context when relevant:\n"
            f"{file_context}"
        )
    messages.append({"role": "user", "content": latest_user_prompt})
    return messages


def stream_generate(
    model,
    tokenizer,
    messages: list[dict[str, str]],
    temperature: float,
    top_p: float,
    repetition_penalty: float,
    max_new_tokens: int,
):
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
    ).to(model.device)

    streamer = TextIteratorStreamer(
        tokenizer,
        skip_special_tokens=True,
        skip_prompt=True,
    )

    do_sample = temperature > 0
    generate_kwargs = dict(
        input_ids=input_ids,
        streamer=streamer,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        pad_token_id=tokenizer.eos_token_id,
    )
    if do_sample:
        generate_kwargs["temperature"] = temperature

    generation_error: dict[str, Exception] = {}

    def _generate():
        try:
            model.generate(**generate_kwargs)
        except Exception as err:
            generation_error["error"] = err

    worker = threading.Thread(target=_generate, daemon=True)
    worker.start()
    return streamer, worker, generation_error


def token_count(tokenizer, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def init_state():
    st.session_state.setdefault("messages", [])


def render_css():
    st.markdown(
        """
        <style>
            .stChatMessage {
                border-radius: 14px;
                padding: 8px 12px;
                margin-bottom: 12px;
            }
            .main .block-container {
                max-width: 980px;
                padding-top: 1.5rem;
            }
            .token-counter {
                color: #7f8799;
                font-size: 0.9rem;
                margin-top: 0.4rem;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main():
    st.set_page_config(page_title="Unsloth Qwen3.5 Local Chat", page_icon="🤖", layout="wide")
    render_css()
    init_state()

    st.title("🤖 Unsloth Qwen3.5 27B Local Chat")
    st.caption(
        "ChatGPT-style local interface with file context upload and live token streaming."
    )

    if not torch.cuda.is_available():
        st.warning(
            "CUDA GPU was not detected. Qwen3.5 27B 4-bit generally needs GPU memory "
            "for usable performance."
        )

    with st.sidebar:
        st.header("Model")
        model_name = st.text_input("Model name", value=DEFAULT_MODEL_NAME)
        max_seq_length = st.number_input(
            "Max sequence length",
            min_value=1024,
            max_value=32768,
            step=1024,
            value=DEFAULT_MAX_SEQ_LENGTH,
        )
        load_in_4bit = st.checkbox("Load in 4-bit", value=True)
        dtype_name = st.selectbox("Compute dtype", options=["auto", "float16", "bfloat16"], index=0)

        st.header("Generation")
        max_new_tokens = st.slider("Max new tokens", min_value=32, max_value=4096, value=1024, step=32)
        temperature = st.slider("Temperature", min_value=0.0, max_value=2.0, value=0.7, step=0.05)
        top_p = st.slider("Top-p", min_value=0.1, max_value=1.0, value=0.95, step=0.01)
        repetition_penalty = st.slider(
            "Repetition penalty", min_value=1.0, max_value=2.0, value=1.05, step=0.01
        )

        st.header("System Prompt")
        system_prompt = st.text_area(
            "Instruction",
            value="You are a helpful local AI assistant.",
            height=120,
        )

        st.header("Files")
        uploaded_files = st.file_uploader(
            "Attach files (text + PDF)",
            accept_multiple_files=True,
            type=None,
        )
        if st.button("Clear chat history", use_container_width=True):
            st.session_state["messages"] = []
            st.rerun()

    with st.spinner("Loading model... this can take a while on first run."):
        model, tokenizer = load_unsloth_model(
            model_name=model_name,
            max_seq_length=int(max_seq_length),
            dtype_name=dtype_name,
            load_in_4bit=load_in_4bit,
        )
    st.success(f"Loaded `{model_name}`")

    extracted_files, extraction_warnings = extract_file_contents(uploaded_files or [])
    for warning in extraction_warnings:
        st.warning(warning)
    if extracted_files:
        st.info("Attached context files: " + ", ".join(f.name for f in extracted_files))

    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_prompt = st.chat_input("Send a message...")
    if not user_prompt:
        return

    with st.chat_message("user"):
        st.markdown(user_prompt)

    st.session_state["messages"].append({"role": "user", "content": user_prompt})

    generation_messages = build_generation_prompt(
        system_prompt=system_prompt,
        chat_history=st.session_state["messages"][:-1],
        latest_user_prompt=user_prompt,
        file_context=build_file_context(extracted_files),
    )

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        token_placeholder = st.empty()
        streamed_chunks: list[str] = []

        streamer, worker, generation_error = stream_generate(
            model=model,
            tokenizer=tokenizer,
            messages=generation_messages,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            max_new_tokens=max_new_tokens,
        )

        for chunk in streamer:
            streamed_chunks.append(chunk)
            partial = "".join(streamed_chunks)
            response_placeholder.markdown(partial)
            token_placeholder.markdown(
                f"<div class='token-counter'>Generated tokens: {token_count(tokenizer, partial)}</div>",
                unsafe_allow_html=True,
            )
        worker.join()

        if generation_error:
            err = generation_error["error"]
            st.error(f"Generation failed: {err}")
            return

        full_response = "".join(streamed_chunks).strip()
        response_placeholder.markdown(full_response)
        token_placeholder.markdown(
            f"<div class='token-counter'>Generated tokens: {token_count(tokenizer, full_response)}</div>",
            unsafe_allow_html=True,
        )

    st.session_state["messages"].append({"role": "assistant", "content": full_response})


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
