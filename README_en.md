# GLM-Edge

## Model Introduction

The GLM-Edge series is our attempt to meet real-world deployment scenarios for edge devices. It consists of two sizes of large language dialogue models and multimodal understanding models (`GLM-Edge-1.5B-Chat`, `GLM-Edge-4B-Chat`, `GLM-Edge-V-2B`, `GLM-Edge-V-5B`). Among them, the `1.5B / 2B` models are mainly targeted at platforms like mobile phones and car machines, while the `4B / 5B` models are aimed at platforms like PCs.

Based on the technological advancements of the GLM-4 series, we have made targeted adjustments to the model structure and size, balancing model performance, real-world inference efficiency, and deployment convenience. Through deep collaboration with partner enterprises and relentless efforts in inference optimization, the GLM-Edge series models can run at extremely high speeds on some edge platforms.

For example, on the Qualcomm Snapdragon 8 Elite platform, leveraging its powerful NPU computing power and using a mixed quantization scheme, the 1.5B dialogue model and the 2B multimodal model can achieve decoding speeds of over 60 tokens per second. With speculative sampling techniques, these models can reach peak decoding speeds of over 100 tokens per second. These inference solutions will be released later by us or our partners.

## Real-World Performance Data

Data collection is up to November 28, 2024. We are actively working with partners to optimize these performances.

### Qualcomm

| Model              | Task                     | Quantization | Framework | 1st Token Latency (ms) | Token Rate (tokens/s) | Peak Memory Footprint (GB) |
|--------------------|--------------------------|--------------|-----------|------------------------|-----------------------|----------------------------|
| GLM-Edge-4B-Chat   | (input/output=512/128)   | INT4         | QNN       | 260                    | 65                    | 2.9                        |
| GLM-Edge-1.5B-Chat | (input/output=512/128)   | INT4         | QNN       | 660                    | 24                    | 1.2                        |

- Tested on the Qualcomm 8 Elite (Gen4) platform with models fully running on the NPU.
- For V models, an additional 890ms processing time per image and about 660MB extra memory is required.
- With speculative decoding, the Token Rate can achieve up to 50% improvement.

### Intel

| Model              | Task                               | Quantization | Framework | 1st Token Latency (ms) | Token Rate (tokens/s) | Peak Memory Footprint (GB) |
|--------------------|------------------------------------|--------------|-----------|------------------------|-----------------------|----------------------------|
| GLM-Edge-4B-Chat   | (input/output=1024/128)           | INT4         | OPENVINO  | 541.2                  | 27                    | 3.9                        |
| GLM-Edge-1.5B-Chat | (input/output=1024/128)           | INT4         | OPENVINO  | 228.2                  | 63                    | 2.3                        |
| GLM-Edge-V-2B      | Single image understanding (672x672) | INT4       | OPENVINO  | 362.1                  | 70                    | 3.4                        |

- Tested on the Intel LNL 288V (ARC 140V 8X@2.05GHz) platform.
- For V models, an additional 1.7s processing time per image and about 2GB extra memory is required.

## Install Dependencies

Ensure your Python version is `3.10` or higher. Install dependencies as follows to ensure all code in this repository runs correctly:

```shell
pip install -r requirements.txt
```

## Model Inference

### Transformers / OpenVINO / vLLM Demo

We provide three backend inference options: vLLM, OpenVINO, and transformers. You can run the models using the following commands. This is a command-line interaction code.

```shell
python cli_demo.py --backend transformers --model_path THUDM/glm-edge-1.5b-chat --precision bfloat16
python cli_demo.py --backend vllm --model_path THUDM/glm-edge-1.5b-chat --precision bfloat16
python cli_demo.py --backend ov --model_path THUDM/glm-edge-1.5b-chat-ov --precision int4
```

> Note:
>
> OpenVINO version models need conversion. Please visit [here](inference/ov_convert) to run the conversion code.
>
> ```python convert_chat.py --model_path  THUDM/glm-edge-1.5b-chat --precision int4 ``` to convert dialogue models.
>
> ```python convert.py --model_path  THUDM/glm-edge-v-2b --precision int4``` to convert visual understanding models.
>
> You can also view the original conversion code [here](https://github.com/openvino-dev-samples/glm-edge.openvino).
>
> vLLM version models require installation of source code from [here](https://github.com/sixsixcoder/vllm/tree/glm-4) to run properly.

To use glm-edge-v series models, you can run the following command-line interaction code:

```shell
python cli_demo_vision.py --backend transformers --model_path THUDM/glm-edge-v-2b --precision bfloat16
python cli_demo.py --backend ov --model_path THUDM/glm-edge-1.5b-chat-ov --precision int4
```

You can also use Gradio to launch a WebUI.

```shell
python cli_demo.py --backend transformers --model_path THUDM/glm-edge-1.5b-chat --precision bfloat16
python cli_demo.py --backend vllm --model_path THUDM/glm-edge-1.5b-chat --precision int4 # For Int4 Inference
```

### XInference

If you use XInference for inference, you can run the model using the following commands. This is a command-line interaction code.

```python
xinference launch --model-engine Transformers --model-name glm-edge-v --size-in-billions 2 --model-format pytorch --quantization none
```

Using OpenAI API for inference:

```python
import openai

client = openai.Client(
    api_key="cannot be empty",
    base_url="http://<XINFERENCE_HOST>:<XINFERENCE_PORT>/v1"
)
output = client.chat.completions.create(
    model="glm-edge-v",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    'type': 'text', 
                    'text': 'describe this image',
                },
                {
                    'type': 'image_url', 
                    'image_url': {
                        "url": "img.png",
                    }
                },
            ],
        }
    ],
    max_tokens=512,
    temperature=0.7
)

print(output)
```

## Fine-Tuning Models

We provide code for fine-tuning models. Please refer to the [Fine-Tuning Tutorial](finetune/README.md).

## License

The code in this GitHub repository uses the [Apache2.0 LICENSE](LICENSE).

Usage of model weights must follow the [Model License](MODEL_LICENSE).
