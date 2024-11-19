import os
import argparse
from pathlib import Path
from threading import Thread
from typing import Union
import requests
from io import BytesIO
from PIL import Image

import gradio as gr
import torch
from peft import AutoPeftModelForCausalLM, PeftModelForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer, BitsAndBytesConfig


# Parse arguments
def parse_args():
    parser = argparse.ArgumentParser(description="GLM-Edge-Chat Gradio Demo with adjustable parameters")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model")
    parser.add_argument("--server_name", type=str, default="127.0.0.1", help="Server name")
    parser.add_argument("--server_port", type=int, default=7860, help="Server port")
    parser.add_argument("--lora_path", type=str, default=None, help="Path to LoRA model if available")
    parser.add_argument("--precision", type=str, choices=["float16", "bfloat16", "int4"],
                        default="bfloat16",
                        help="Precision for model"
                        )
    return parser.parse_args()


args = parse_args()


def _resolve_path(path: Union[str, Path]) -> Path:
    return Path(path).expanduser().resolve()


def load_model_and_tokenizer(model_dir: Union[str, Path], precision: str, trust_remote_code: bool = True):
    model_dir = _resolve_path(model_dir)
    if precision == "int4":
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            trust_remote_code=trust_remote_code,
            quantization_config=BitsAndBytesConfig(load_in_4bit=True),
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        ).eval()
    elif (model_dir / "adapter_config.json").exists():
        model = AutoPeftModelForCausalLM.from_pretrained(
            model_dir, trust_remote_code=trust_remote_code, device_map="auto"
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(model_dir, trust_remote_code=trust_remote_code, device_map="auto")

    tokenizer_dir = model.peft_config["default"].base_model_name_or_path if hasattr(model, 'peft_config') else model_dir
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, trust_remote_code=trust_remote_code, use_fast=False)
    return model, tokenizer


model, tokenizer = load_model_and_tokenizer(args.model_path, args.precision, trust_remote_code=True)


def get_image(image_path=None, image_url=None):
    if image_path:
        return Image.open(image_path).convert("RGB")
    elif image_url:
        response = requests.get(image_url)
        return Image.open(BytesIO(response.content)).convert("RGB")
    return None


def predict(history, prompt, max_length, top_p, temperature, image=None):
    messages = []
    if prompt:
        messages.append({"role": "system", "content": prompt})
    for idx, (user_msg, model_msg) in enumerate(history):
        if prompt and idx == 0:
            continue
        if idx == len(history) - 1 and not model_msg:
            messages.append({"role": "user", "content": user_msg})
            break
        if user_msg:
            messages.append({"role": "user", "content": user_msg})
        if model_msg:
            messages.append({"role": "assistant", "content": model_msg})

    # Check if the model supports image input
    if "image" in tokenizer.apply_chat_template.__code__.co_varnames:
        messages.append({"role": "user", "content": "", "image": image})

    model_inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True, return_tensors="pt", return_dict=True
    )
    streamer = TextIteratorStreamer(tokenizer, timeout=60, skip_prompt=True, skip_special_tokens=True)
    generate_kwargs = {
        "input_ids": model_inputs["input_ids"],
        "attention_mask": model_inputs["attention_mask"],
        "streamer": streamer,
        "max_new_tokens": max_length,
        "do_sample": True,
        "top_p": top_p,
        "temperature": temperature,
        "repetition_penalty": 1.2,
        "eos_token_id": tokenizer.encode("<|user|>"),
    }
    t = Thread(target=model.generate, kwargs=generate_kwargs)
    t.start()
    for new_token in streamer:
        if new_token:
            history[-1][1] += new_token
        yield history


def main():
    with gr.Blocks() as demo:
        gr.HTML("""<h1 align="center">GLM-Edge-Chat Gradio Chat Demo</h1>""")

        # Top row: Chatbot and Image upload
        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot()
            with gr.Column(scale=1):
                image_input = gr.Image(label="Upload an Image", type="filepath")

        # Bottom row: System prompt, user input, and controls
        with gr.Row():
            with gr.Column(scale=2):
                prompt_input = gr.Textbox(show_label=True, placeholder="System Prompt", label="System Prompt")
                user_input = gr.Textbox(show_label=True, placeholder="Input...", label="User Input")
                submitBtn = gr.Button("Submit")
                pBtn = gr.Button("Set System prompt")
                emptyBtn = gr.Button("Clear History")
            with gr.Column(scale=1):
                max_length = gr.Slider(0, 8192, value=4096, step=1.0, label="Maximum length", interactive=True)
                top_p = gr.Slider(0, 1, value=0.8, step=0.01, label="Top P", interactive=True)
                temperature = gr.Slider(0.01, 1, value=0.6, step=0.01, label="Temperature", interactive=True)

        # Define functions for button actions
        def user(query, history):
            return "", history + [[query, ""]]

        def set_prompt(prompt_text):
            return [[prompt_text, "Prompt set successfully"]]

        # Button actions and callbacks
        pBtn.click(set_prompt, inputs=[prompt_input], outputs=chatbot)
        submitBtn.click(user, [user_input, chatbot], [user_input, chatbot], queue=False).then(
            predict, [chatbot, prompt_input, max_length, top_p, temperature, image_input], chatbot
        )
        emptyBtn.click(lambda: (None, None), None, [chatbot, prompt_input], queue=False)

    demo.queue()
    demo.launch(server_name=args.server_name, server_port=args.server_port)


if __name__ == "__main__":
    main()