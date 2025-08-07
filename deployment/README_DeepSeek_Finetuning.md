# 🔧 Fine-Tuning DeepSeek-R1-Distill-Qwen-7B with LLaMA Factory

This repository provides an end-to-end deployment and fine-tuning setup for the [`DeepSeek-R1-Distill-Qwen-7B`](https://huggingface.co/deepseek-ai/deepseek-llm-7b-chat) model using [LLaMA Factory](https://github.com/hiyouga/LLaMA-Factory).

---

## 📦 Environment Overview

### ✅ Hardware Configuration

- **CPU**: Intel(R) Xeon(R) Gold 6430  
- **GPU**: 4 × NVIDIA L20 (each with 46GB memory)  
- **Memory**: 512 GB RAM  
- **Storage**: ~3TB local disk  
- **Driver**: NVIDIA Driver 550.90.07  
- **CUDA Version**: 12.4  

### ✅ Software Configuration

- **Operating System**: Ubuntu 20.04.6 LTS  
- **Python Version**: 3.8.10  
- **CUDA Toolkit**: 12.4  
- **LLaMA Factory Version**: 0.9.3.dev0  
- **DeepSpeed**: Optional, used for distributed/memory-efficient training

---

## 🔧 Installation Guide

### Step 1. Clone LLaMA Factory

```bash
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
```

### Step 2. Set up Conda Environment

```bash
conda create -n llama_factory python=3.8 -y
conda activate llama_factory
```

### Step 3. Install Dependencies

```bash
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -e .[metrics]
```

### Step 4. Verify Installation

```bash
llamafactory-cli version
```

Expected Output:

```
Welcome to LLaMA Factory, version 0.9.3.dev0
Project page: https://github.com/hiyouga/LLaMA-Factory
```


## 🧪 Fine-tuning Example

### Sample Configuration File (YAML)

```yaml
model_name_or_path: deepseek-ai/deepseek-llm-7b-chat
template: alpaca
finetuning_type: lora
do_train: true
dataset: your_alpaca_data
output_dir: output/deepseek-lora
fp16: true
per_device_train_batch_size: 2
gradient_accumulation_steps: 8
num_train_epochs: 3
save_steps: 100
logging_steps: 10
```

### Launch Training

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
torchrun --nproc_per_node=4 \
    src/train_bash.py \
    --config_file config/finetune_config.yaml
```

---

## 📊 Input Format

Each sample should follow the [Alpaca format](https://github.com/tatsu-lab/stanford_alpaca):

```json
{
  "instruction": "Please analyze the following behavior sequence. Respond with both an anomaly score and a classification result (‘Normal‘ or ‘Abnormal‘)",
  "input": "After working hours, access website wikipedia.org, then send email from insider to outsider.",
  "output": "Anomaly Score = 1.00, Prediction = “Abnormal”"
}
```

---

## 🔍 Notes

- Make sure the CUDA version **matches** your installed driver (≥12.4 for L20).
- For distributed training, DeepSpeed is recommended (ensure correct CUDA version).
- You may adjust batch size, LoRA rank, and dataset paths according to your memory.

---

## 📮 Acknowledgments

- [DeepSeek AI](https://github.com/deepseek-ai)
- [LLaMA Factory](https://github.com/hiyouga/LLaMA-Factory)
- Alpaca format by [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca)

---

## ⬇️ Downloading the Base Model (DeepSeek-R1-Distill-Qwen-7B)

Before fine-tuning, you must download the base model weights from [Hugging Face](https://huggingface.co/deepseek-ai/deepseek-llm-7b-chat).

### Method 1: Using `transformers` (Recommended)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("deepseek-ai/deepseek-llm-7b-chat")
tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-llm-7b-chat")
```

### Method 2: Pre-download via CLI

```bash
huggingface-cli login  # if needed
huggingface-cli download deepseek-ai/deepseek-llm-7b-chat --local-dir ./models/deepseek-7b
```

Make sure to specify the correct `model_name_or_path` in your config:
```yaml
model_name_or_path: ./models/deepseek-7b
```


> 💡 Tip: If downloading fails due to region or permission issues, try using a mirror or manually download from Hugging Face and unzip into `./models/`.


