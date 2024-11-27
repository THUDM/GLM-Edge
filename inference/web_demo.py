import time
import argparse
from pathlib import Path
from threading import Thread
from typing import Union
import requests
from io import BytesIO
from PIL import Image

import gradio as gr
import torch
from vllm import SamplingParams, AsyncEngineArgs, AsyncLLMEngine
from peft import AutoPeftModelForCausalLM
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoImageProcessor,
    TextIteratorStreamer,
    BitsAndBytesConfig,
)

# Parse arguments
def parse_args():
    parser = argparse.ArgumentParser(description="GLM-Edge-Chat Gradio Demo with adjustable parameters")
    parser.add_argument(
        "--backend",
        type=str,
        choices=["vllm", "transformers"],
        required=True,
        help="Choose inference backend: vllm or transformers",
    )
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model")
    parser.add_argument("--server_name", type=str, default="127.0.0.1", help="Server name")
    parser.add_argument("--server_port", type=int, default=7860, help="Server port")
    parser.add_argument("--lora_path", type=str, default=None, help="Path to LoRA model if available")
    parser.add_argument(
        "--precision",
        type=str,
        choices=["float16", "bfloat16", "int4"],
        default="bfloat16",
        help="Precision for model",
    )
    return parser.parse_args()

args = parse_args()

def _resolve_path(path: Union[str, Path]) -> Path:
    return Path(path).expanduser().resolve()

# Load Model and Tokenizer for VLLM
def load_vllm_model_and_tokenizer(model_dir: str, precision: str):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    engine_args = AsyncEngineArgs(
        model=model_dir,
        tokenizer=model_dir,
        tensor_parallel_size=1,
        dtype="bfloat16" if precision == "bfloat16" else "float16",
        gpu_memory_utilization=0.9,
        enforce_eager=True,
        worker_use_ray=True,
        disable_log_requests=True,
    )
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    return engine, tokenizer

# Load Model and Tokenizer for transformers
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

    tokenizer_dir = (
        model.peft_config["default"].base_model_name_or_path if hasattr(model, "peft_config") else model_dir
    )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, trust_remote_code=trust_remote_code, use_fast=False)
    return model, tokenizer

if args.backend == "vllm":
    engine, tokenizer = load_vllm_model_and_tokenizer(args.model_path, args.precision)
else:
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
    pixel_values = None
    if prompt:
        messages.append({"role": "system", "content": [{"type": "text", "text": prompt}]})
    for idx, (user_msg, model_msg) in enumerate(history):
        if prompt and idx == 0:
            continue
        if idx == len(history) - 1 and not model_msg:
            messages.append({"role": "user", "content": [{"type": "text", "text": user_msg}]})
            break
        if user_msg:
            messages.append({"role": "user", "content": [{"type": "text", "text": user_msg}]})
        if model_msg:
            messages.append({"role": "assistant", "content": [{"type": "text", "text": model_msg}]})

    if image:
        try:
            image_input = Image.open(image).convert("RGB")
            messages.append({"role": "user", "content": [{"type": "text", "text": ""},{"type": "image"}]})
        except:
            print("Invalid image path. Continuing with text conversation.")
        processor = AutoImageProcessor.from_pretrained(
                args.model_path, 
                trust_remote_code=True
                )
        pixel_values = torch.tensor(
            processor(image_input).pixel_values).to("cuda")
        
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
    }
    if image:
        generate_kwargs['pixel_values'] = pixel_values
    else:
        generate_kwargs['eos_token_id'] = tokenizer.encode("<|user|>")
        
    t = Thread(target=model.generate, kwargs=generate_kwargs)
    t.start()
    for new_token in streamer:
        if new_token:
            history[-1][1] += new_token
        yield history

async def vllm_gen(history, prompt, max_length, top_p, temperature):
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

    inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    sampling_params = SamplingParams(
        n=1,
        best_of=1,
        presence_penalty=1.0,
        frequency_penalty=0.0,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_length,
    )
    async for output in engine.generate(
        prompt=inputs, sampling_params=sampling_params, request_id=f"{time.time()}"
    ):
        yield output.outputs[0].text

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
        if args.backend == "vllm":
            submitBtn.click(user, [user_input, chatbot], [user_input, chatbot], queue=False).then(
                vllm_gen, [chatbot, prompt_input, max_length, top_p, temperature], chatbot
            )
        else:
            submitBtn.click(user, [user_input, chatbot], [user_input, chatbot], queue=False).then(
                predict, [chatbot, prompt_input, max_length, top_p, temperature, image_input], chatbot
            )
        emptyBtn.click(lambda: (None, None), None, [chatbot, prompt_input], queue=False)

    demo.queue()
    demo.launch(server_name=args.server_name, server_port=args.server_port)


if __name__ == "__main__":
    main()
