import argparse
import time
import asyncio
from threading import Thread
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TextIteratorStreamer,
    BitsAndBytesConfig,
)
from vllm import SamplingParams, AsyncEngineArgs, AsyncLLMEngine
from vllm.lora.request import LoRARequest
from optimum.intel.openvino import OVModelForCausalLM
import torch


# Load Model and Tokenizer for VLLM
def load_vllm_model_and_tokenizer(model_dir: str, lora_path: str, precision: str):
    enable_lora = bool(lora_path)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    engine_args = AsyncEngineArgs(
        model=model_dir,
        tokenizer=model_dir,
        enable_lora=enable_lora,
        tensor_parallel_size=1,
        dtype="bfloat16" if precision == "bfloat16" else "float16",
        gpu_memory_utilization=0.9,
        enforce_eager=True,
        worker_use_ray=True,
        disable_log_requests=True,
    )
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    return engine, tokenizer, enable_lora


async def vllm_gen(engine, tokenizer, lora_path, enable_lora, messages, top_p, temperature, max_length):
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
    if enable_lora:
        async for output in engine.generate(
            inputs=inputs,
            sampling_params=sampling_params,
            request_id=f"{time.time()}",
            lora_request=LoRARequest("GLM-Edge-lora", 1, lora_path=lora_path),
        ):
            yield output.outputs[0].text
    else:
        async for output in engine.generate(
            prompt=inputs, sampling_params=sampling_params, request_id=f"{time.time()}"
        ):
            yield output.outputs[0].text


# CLI Chat Function for Transformers and OpenVINO
def generic_chat(tokenizer, model, temperature, top_p, max_length, backend="transformers"):
    history = []
    backend_label = "OpenVINO" if backend == "ov" else "Transformers"
    print(f"Welcome to the GLM-Edge CLI chat ({backend_label}). Type your messages below.")
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        history.append([user_input, ""])

        messages = [{"role": "user", "content": user_input}]
        model_inputs = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
        )
        model_inputs = {k: v.to("cpu") for k, v in model_inputs.items()}  # Ensure CPU for OpenVINO

        streamer = TextIteratorStreamer(tokenizer=tokenizer, timeout=60, skip_prompt=True, skip_special_tokens=True)
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

        print("GLM-Edge:", end="", flush=True)
        for new_token in streamer:
            print(new_token, end="", flush=True)
            history[-1][1] += new_token

        history[-1][1] = history[-1][1].strip()


# Main Async Chat Function for VLLM
async def vllm_chat(engine, tokenizer, lora_path, enable_lora, temperature, top_p, max_length):
    history = []
    print("Welcome to the GLM-Edge DEMO chat (VLLM). Type your messages below.")
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        history.append([user_input, ""])

        messages = [{"role": "user", "content": user_input}]
        print("\nGLM-Edge: ", end="")
        current_length = 0
        output = ""
        async for output in vllm_gen(
            engine, tokenizer, lora_path, enable_lora, messages, top_p, temperature, max_length
        ):
            print(output[current_length:], end="", flush=True)
            current_length = len(output)
        history[-1][1] = output


def main():
    parser = argparse.ArgumentParser(description="Run GLM-Edge DEMO Chat with VLLM, Transformers, or OpenVINO backend")
    parser.add_argument(
        "--backend",
        type=str,
        choices=["vllm", "transformers", "ov"],
        required=True,
        help="Choose inference backend: vllm, transformers, or OpenVINO",
    )
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model")
    parser.add_argument("--lora_path", type=str, default=None, help="Path to LoRA (leave empty to skip)")
    parser.add_argument(
        "--precision", type=str, default="bfloat16", choices=["float16", "bfloat16", "int4"], help="Model precision"
    )
    parser.add_argument("--temperature", type=float, default=0.6, help="Temperature for sampling")
    parser.add_argument("--top_p", type=float, default=0.8, help="Top-p (nucleus) sampling probability")
    parser.add_argument("--max_length", type=int, default=8192, help="Maximum token length for generation")
    args = parser.parse_args()

    if args.backend == "vllm":
        engine, tokenizer, enable_lora = load_vllm_model_and_tokenizer(args.model_path, args.lora_path, args.precision)
        asyncio.run(
            vllm_chat(engine, tokenizer, args.lora_path, enable_lora, args.temperature, args.top_p, args.max_length)
        )
    elif args.backend == "ov":
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        model = OVModelForCausalLM.from_pretrained(args.model_path, device="CPU") # CPU,GPU and XPU are supported
        generic_chat(tokenizer, model, args.temperature, args.top_p, args.max_length, backend="ov")
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
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
        generic_chat(tokenizer, model, args.temperature, args.top_p, args.max_length, backend="transformers")


if __name__ == "__main__":
    main()
