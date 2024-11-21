# GLM-Edge dialogue model fine-tuning

Read this in [Chinese](README.md)

In this demo, you will experience how to fine-tune the GLM-Edge-4B-Chat open source dialogue model. Please strictly follow the steps in the document to avoid unnecessary errors.

## Environment Configuration

Before you start fine-tuning, make sure you have cloned the latest version of the model repository. You also need to install the dependencies in this directory:

```bash
pip install -r requirements.txt
```

## Multi-turn dialogue format

The multi-turn dialogue fine-tuning example uses the GLM-Edge dialogue format convention, adding different `loss_mask` to different roles to calculate `loss` for multiple rounds of replies in one calculation.

For data files, the sample uses the following format

If you only want to fine-tune the model's dialogue capabilities, not the tool capabilities, you should organize the data in the following format.

```json
[
  {
    "messages": [
      {
        "role": "system",
        "content": "<system prompt text>",
      },
      {
        "role": "user",
        "content": "<user prompt text>"
      },
      {
        "role": "assistant",
        "content": "<assistant response text>"
      },
      // Multi_turns
      {
        "role": "user",
        "content": "<user prompt text>"
      },
      {
        "role": "assistant",
        "content": "<assistant response text>"
      },
    ]
  }
]
```

Here is an example of a single-turn conversation:

```json
{
  "messages": [
    {
      "role": "user",
      "content": "Type#Pants*Material#Denim*Style#Sexy"
    },
    {
      "role": "assistant",
      "content": "This pair of jeans from 3x1 is made of light white denim fabric. Its soft feel and delicate texture make it comfortable to wear while revealing a pure and sweet personality. In addition, the smooth cut of the pants fully highlights the sexy leg curves, making it a must-have item for casual outings."
    }
  ]
}
```

## Configuration Files

The fine-tuning configuration files are located in the `config` directory and include the following files:

1. `ds_zereo_2 / ds_zereo_3.json`: deepspeed configuration file

2. `lora.yaml / sft.yaml`: Configuration files of different models, including model parameters, optimizer parameters, training parameters, etc. Some important parameters are explained as follows:

+ train_file: File path of training dataset.
+ val_file: File path of validation dataset.
+ test_file: File path of test dataset.
+ num_proc: Number of processes to use when loading data.
+ max_input_length: Maximum length of input sequence.
+ max_output_length: Maximum length of output sequence.
+ training_args section
+ output_dir: Directory for saving model and other outputs.
+ max_steps: Maximum number of training steps.
+ per_device_train_batch_size: Training batch size per device (such as GPU).
+ dataloader_num_workers: Number of worker threads to use when loading data.
+ remove_unused_columns: Whether to remove unused columns in data.
+ save_strategy: Model saving strategy (for example, how many steps to save).
+ save_steps: How many steps to save the model.
+ log_level: Log level (such as info).
+ logging_strategy: logging strategy.
+ logging_steps: how many steps to log at.
+ per_device_eval_batch_size: per-device evaluation batch size.
+ evaluation_strategy: evaluation strategy (e.g. how many steps to evaluate at).
+ eval_steps: how many steps to evaluate at.
+ predict_with_generate: whether to use generation mode for prediction.
+ generation_config section
+ max_new_tokens: maximum number of new tokens to generate.
+ peft_config section
+ peft_type: type of parameter tuning to use (supports LORA and PREFIX_TUNING).
+ task_type: task type, here is causal language model (don't change).
+ Lora parameters:
+ r: rank of LoRA.
+ lora_alpha: scaling factor of LoRA.
+ lora_dropout: dropout probability to use in LoRA layer.

## Start fine-tuning

Execute **single machine multi-card/multi-machine multi-card** run through the following code, which uses `deepspeed` as
the acceleration solution, and you need to install `deepspeed`.

```shell
OMP_NUM_THREADS=1 torchrun --standalone --nnodes=1 --nproc_per_node=8  finetune.py  data/AdvertiseGen/  THUDM/glm-edge-4b-chat  configs/lora.yaml # For Chat Fine-tune
```

Execute **single machine single card** run through the following code.

```shell
python finetune.py  data/AdvertiseGen/  THUDM/glm-edge-4b-chat  configs/lora.yaml # For Chat Fine-tune
```

## Fine-tune from a saved point

If you train as described above, each fine-tuning will start from the beginning. If you want to fine-tune from a
half-trained model, you can add a fourth parameter, which can be passed in two ways:

1. `yes`, automatically start training from the last saved Checkpoint

2. `XX`, breakpoint number, for example `600`, start training from Checkpoint 600

For example, this is an example code to continue fine-tuning from the last saved point

```shell
python finetune.py data/AdvertiseGen/ THUDM/glm-edge-4b-chat configs/lora.yaml yes
```
