# GLM-Edge

Read this in [English](README_en.md)

## 模型介绍

GLM-Edge系列是我们在面向端侧真实落地使用的场景下的一次尝试，由两种尺寸的大语言对话模型和多模态理解模型组成（
`GLM-Edge-1.5B-Chat`，`GLM-Edge-4B-Chat`，`GLM-Edge-V-2B`，`GLM-Edge-V-5B`）。其中，`1.5B / 2B`模型主要面向手机、车机等平台，
`4B / 5B` 模型主要面向PC等平台。

基于GLM-4系列的技术积累，我们针对端侧实际部署情况，对模型结构和尺寸做了针对性的调整，以求在模型表现、实机推理效果和落地便利度之间达到平衡。同时，通过与伙伴企业的深入合作和在推理优化上的不懈努力，在一些端侧平台上，GLM-Edge系列模型能以极快的速度运行。

例如，在高通骁龙8 Elite平台上，借助其强大的NPU算力，GLM-Edge通过混合量化方案，1.5B对话模型、2B多模态模型能实现每秒60
tokens以上的解码速度。在应用投机采样技术之后，两个模型能以峰值每秒100 tokens以上的解码速度运行。这些推理方案会由我们或合作伙伴后续放出。

## 实机运行数据

数据采集日截止到2024年11月28日。我们还在积极地与合作伙伴们一道优化这些性能。

### 高通

| 模型                 | 任务                     | 量化方案 | 框架  | 1st token latency (ms) | Token Rate (tokens/s) | Peak Memory Footprint (GB) |
|--------------------|------------------------|------|-----|------------------------|-----------------------|----------------------------|
| GLM-Edge-4B-Chat   | (input/output=512/128) | INT4 | QNN | 260                    | 65                    | 2.9                        |
| GLM-Edge-1.5B-Chat | (input/output=512/128) | INT4 | QNN | 660                    | 24                    | 1.2                        |

* 在高通8 Elite（Gen4）平台上测试，模型全部运行在NPU上
* 如运行V模型，另外需要单图890ms的处理时间和约660M的额外内存
* 使用投机解码方案时，Token Rate还有最高50%的提升

### Intel

| 模型                 | 任务                                   | 量化方案 | 框架       | 1st token latency (ms) | Token Rate (tokens/s) | Peak Memory Footprint (GB) |
|--------------------|--------------------------------------|------|----------|------------------------|-----------------------|----------------------------|
| GLM-Edge-4B-Chat   | (input/output=1024/128)              | INT4 | OPENVINO | 541.2                  | 27                    | 3.9                        |
| GLM-Edge-1.5B-Chat | (input/output=1024/128)              | INT4 | OPENVINO | 228.2                  | 63                    | 2.3                        |
| GLM-Edge-V-2B      | Single image understanding (672x672) | INT4 | OPENVINO | 362.1                  | 70                    | 3.4                        |

* 在Intel LNL 288V (ARC 140V 8X@2.05GHz) 平台上测试。
* 如运行V模型，另外需要单图1.7s的处理时间和约2G的额外内存。

## 安装依赖

请确保你的Python版本为`3.10`或更高版本。并按照如下方式安装依赖，安装以下依赖能确保正确运行本仓库的所有代码。

```shell
pip install -r requirements.txt
```

## 模型推理

### Transformers / OpenVINO / vLLM Demo

我们提供了 vLLM, OpenVINO 和 transformers 三种后端推理方式,你可以通过运行以下命令来运行模型。这是一个命令行交互代码。

```shell
python cli_demo.py --backend transformers --model_path THUDM/glm-edge-1.5b-chat --precision bfloat16
python cli_demo.py --backend vllm --model_path THUDM/glm-edge-1.5b-chat --precision bfloat16
python cli_demo.py --backend ov --model_path THUDM/glm-edge-1.5b-chat-ov  --precision int4
```

> 注意：
>
> OpenVINO 版本模型需要进行转换，请前往 [这里](inference/ov_convert) 运行转换代码。
>
> ```python convert_chat.py --model_path  THUDM/glm-edge-1.5b-chat --precision int4 ``` 转换对话模型。
>
> ```python convert.py --model_path  THUDM/glm-edge-v-2b --precision int4``` 转换视觉理解模型。
>
> 你也可以在 [这里](https://github.com/openvino-dev-samples/glm-edge.openvino) 查看原始的转换代码。
>
> vLLM 版本模型需要从 [这里](https://github.com/sixsixcoder/vllm/tree/glm-4) 源代码 安装 vLLM 以正常运行。

如果你想使用 glm-edge-v 系列模型，你可以运行以下命令行交互代码

```shell
python cli_demo_vision.py  --backend transformers --model_path THUDM/glm-edge-v-2b --precision bfloat16
python cli_demo.py --backend ov --model_path THUDM/glm-edge-1.5b-chat-ov  --precision int4
```

你也可以使用 Gradio 启动 WebUI。

```shell
python cli_demo.py --backend transformers --model_path THUDM/glm-edge-1.5b-chat --precision bfloat16
python cli_demo.py --backend vllm --model_path THUDM/glm-edge-1.5b-chat --precision int4 # For Int4 Inference
```

### XInference

如果你使用 XInference 进行推理，你可以通过运行以下命令来运行模型。这是一个命令行交互代码。

```python
xinference launch --model-engine Transformers --model-name glm-edge-v --size-in-billions 2 --model-format pytorch --quantization none
```

使用 OpenAI API进行推理:

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

## 微调模型

我们提供了微调模型的代码，请参考 [微调教程](finetune/README.md)。

## 协议

本 github 仓库代码的使用 [Apache2.0 LICENSE](LICENSE)。

模型权重的使用请遵循 [Model License](MODEL_LICENSE)。
