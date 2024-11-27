from transformers import AutoTokenizer
from optimum.intel import OVWeightQuantizationConfig
from optimum.intel.openvino import OVModelForCausalLM
from optimum.exporters.tasks import TasksManager
import os
import argparse

TasksManager._SUPPORTED_MODEL_TYPE["glm"] = TasksManager._SUPPORTED_MODEL_TYPE[
    "llama"
]  # Using with Llama Type in converting

if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--model_path", default="THUDM/glm-edge-1.5b-chat", type=str, help="orignal model path")
    parser.add_argument(
        "--precision", default="int4", type=str, choices=["fp16", "int8", "int4"], help="fp16, int8 or int4"
    )
    parser.add_argument("--output_path", default="glm-edge-1.5b-chat-ov", type=str, help="path to save the IR model")
    args = parser.parse_args()
    os.makedirs(args.output_path, exist_ok=True)

    compression_configs = {
        "sym": True,
        "group_size": 128,
        "ratio": 0.8,
    }

    TasksManager._SUPPORTED_MODEL_TYPE["glm"] = TasksManager._SUPPORTED_MODEL_TYPE["llama"]

    print("====Exporting IR=====")
    if args.precision == "int4":
        ov_model = OVModelForCausalLM.from_pretrained(
            args.model_path,
            export=True,
            compile=False,
            quantization_config=OVWeightQuantizationConfig(bits=4, **compression_configs),
            trust_remote_code=True,
        )
    elif args.precision == "int8":
        ov_model = OVModelForCausalLM.from_pretrained(
            args.model_path, export=True, compile=False, load_in_8bit=True, trust_remote_code=True
        )
    else:
        ov_model = OVModelForCausalLM.from_pretrained(
            args.model_path, export=True, compile=False, load_in_8bit=False, trust_remote_code=True
        )

    print("====Saving IR=====")
    ov_model.save_pretrained(args.output_path)

    print("====Exporting tokenizer=====")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    tokenizer.save_pretrained(args.output_path)

    print("====Exporting IR tokenizer=====")
    from optimum.exporters.openvino.convert import export_tokenizer

    export_tokenizer(tokenizer, args.output_path)
    print("====Finished=====")
