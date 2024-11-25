# GLM-Edge

## 项目更新

- 🔥🔥 **News**: ```2024/11/30```: 我们发布 `GLM-Edge` 模型。共计`glm-edge-1.5b-chat`, `glm-edge-4b-chat`, `glm-edge-v-2b`,
  `glm-edge-v-5b` 四个模型。并发布了基础的推理代码。

## 模型介绍

GLM-Edge 系列模型是针对端侧领域设计的模型。 我们发布了`glm-edge-1.5b-chat`, `glm-edge-4b-chat`, `glm-edge-v-2b`,
`glm-edge-v-5b` 四个模型。

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
> OpenVINO 版本模型需要运行 convert_model_ov.py 来转换得到。
>
> vLLM 版本模型需要从这里 源代码 安装 vLLM 以正常运行。

你也可以使用 Gradio 启动 WebUI。

```shell
python cli_demo.py --backend transformers --model_path THUDM/glm-edge-1.5b-chat --precision bfloat16
python cli_demo.py --backend vllm --model_path THUDM/glm-edge-1.5b-chat --precision int4 # For Int4 Inference
```

> 注意：
>
> 这里没有提供 OV 的 实现。

### XInference

如果你使用 XInference 进行推理，你可以通过运行以下命令来运行模型。这是一个命令行交互代码。

```shell
```

## 微调模型

我们提供了微调模型的代码，请参考 [微调教程](finetune/README.md)。

## 协议

本 github 仓库代码的使用 [Apache2.0 LICENSE](LICENSE)。

模型权重的使用请遵循 [Model License](MODEL_LICENSE)。
