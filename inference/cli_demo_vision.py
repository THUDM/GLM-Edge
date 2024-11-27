import argparse
from threading import Thread

from PIL import Image
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoImageProcessor,
    TextIteratorStreamer,
    BitsAndBytesConfig,
)
import torch
from ov_convert.convert_v import OvGLMv


def generic_chat(tokenizer, processor, model, temperature, top_p, max_length, backend="transformers"):
    history = []
    image = None
    backend_label = "OpenVINO" if backend == "ov" else "Transformers"
    print(f"Welcome to the GLM-Edge-v CLI chat ({backend_label}). Type your messages below.")
    image_path = input("Image Path:")
    try:
        image = Image.open(image_path).convert("RGB")
        pixel_values = torch.tensor(processor(image).pixel_values).to(model.device)
    except:
        print("Invalid image path. Continuing with text conversation.")

    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        history.append([user_input, ""])

        messages = []
        for idx, (user_msg, model_msg) in enumerate(history):
            if idx == len(history) - 1 and not model_msg:
                messages.append({"role": "user", "content": [{"type": "text", "text": user_msg}, {"type": "image"}]})
                break
            if user_msg:
                messages.append({"role": "user", "content": [{"type": "text", "text": user_msg}]})
            if model_msg:
                messages.append({"role": "assistant", "content": [{"type": "text", "text": model_msg}]})

        model_inputs = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_tensors="pt", return_dict=True
        )

        model_inputs = model_inputs.to(model.device)

        streamer = TextIteratorStreamer(tokenizer=tokenizer, timeout=60, skip_prompt=True, skip_special_tokens=True)
        generate_kwargs = {
            **model_inputs,
            "streamer": streamer,
            "max_new_tokens": max_length,
            "do_sample": True,
            "top_p": top_p,
            "temperature": temperature,
            "repetition_penalty": 1.2,
            "pixel_values": pixel_values if image else None,
        }

        t = Thread(target=model.generate, kwargs=generate_kwargs)
        t.start()
        print("GLM-Edge-v:", end="", flush=True)
        for new_token in streamer:
            if new_token:
                print(new_token, end="", flush=True)
                history[-1][1] += new_token

        history[-1][1] = history[-1][1].strip()


def main():
    parser = argparse.ArgumentParser(description="Run GLM-Edge-v DEMO Chat with Transformers or OpenVINO backend")

    parser.add_argument(
        "--backend",
        type=str,
        choices=["transformers", "ov"],
        required=True,
        help="Choose inference backend: transformers or OpenVINO",
    )
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model")
    parser.add_argument(
        "--precision", type=str, default="bfloat16", choices=["float16", "bfloat16", "int4"], help="Model precision"
    )
    parser.add_argument("--temperature", type=float, default=0.6, help="Temperature for sampling")
    parser.add_argument("--top_p", type=float, default=0.8, help="Top-p (nucleus) sampling probability")
    parser.add_argument("--max_length", type=int, default=8192, help="Maximum token length for generation")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, encode_special_tokens=True)
    processor = AutoImageProcessor.from_pretrained(args.model_path, trust_remote_code=True)

    if args.backend == "ov":
        model = OvGLMv(args.model_path, device="CPU")  # Use OpenVINO
    else:
        if args.precision == "int4":
            model = AutoModelForCausalLM.from_pretrained(
                args.model_path,
                trust_remote_code=True,
                quantization_config=BitsAndBytesConfig(load_in_4bit=True),
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
            ).eval()
        else:
            model = AutoModelForCausalLM.from_pretrained(
                args.model_path,
                torch_dtype=torch.bfloat16 if args.precision == "bfloat16" else torch.float16,
                trust_remote_code=True,
                device_map="auto",
            ).eval()

    generic_chat(tokenizer, processor, model, args.temperature, args.top_p, args.max_length, backend=args.backend)


if __name__ == "__main__":
    main()
