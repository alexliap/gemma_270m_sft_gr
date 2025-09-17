# Gemma 3 270M Fintuned with Greek Data

**Fine-tuned Greek Text Generation Model**
Built upon **Gemma-3 270M**, this model is fine-tuned for improved Greek text generation using a combination of QA / conversational / domain-specific datasets.

[![Model on HF](https://huggingface.co/datasets/huggingface/badges/resolve/main/model-on-hf-sm.svg)](https://huggingface.co/alexliap/gemma-3-270m-it-stf-gr)

---

## Model Overview

- **Base Model:** Google `gemma-3-270m`
- **Fine-tuning method:** LoRA (merged weights) via the [Unsloth](https://github.com/unsloth) approach & HuggingFace TRL library
- **Final Format:** Base + LoRA weights merged into a single model
- **Parameter Count:** 270M parameters
- **Precision / Tensor Type:** BF16

## Training Data

This model was fine-tuned on the following datasets:

| Dataset | Purpose / Domain |
|---|---|
| `facebook/belebele` | General conversational / text generation data |
| `alexandrainst/multi-wiki-qa` | QA style content, multilingual Wikipedia QA |
| `ilsp/truthful_qa_greek` | Truthful QA in Greek |
| `ilsp/medical_mcqa_greek` | Multiple-choice QA in medical domain, Greek |
| `ilsp/greek_civics_qa` | Greek civics domain QA |
| `ilsp/hellaswag_greek` | Greek version of Hellaswag (commonsense reasoning) |

## Usage Example

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
import torch

# Load model & tokenizer
model_name = "alexliap/gemma-3-270m-it-stf-gr"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16, device_map="cuda")

messages = [
    {"role": "system", "content": "Είσαι ένας χρήσιμος βοηθός που απαντά ερωτήσεις."},
    {"role": "user", "content": "Τι είναι κοινωνικότητα;"},
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,  # Must add for generation
).removeprefix("<bos>")

model.generate(
    **tokenizer(text, return_tensors="pt").to("cuda"),
    max_new_tokens=256,
    do_sample=True,
    top_p=0.9,
    temperature=0.8,
    streamer=TextStreamer(tokenizer, skip_prompt=True),
)
```

---

[![Share to Community](https://huggingface.co/datasets/huggingface/badges/resolve/main/powered-by-huggingface-light.svg)](https://huggingface.co)

[<img src="https://raw.githubusercontent.com/unslothai/unsloth/main/images/unsloth%20made%20with%20love.png" width="200"/>](https://github.com/unslothai/unsloth)
