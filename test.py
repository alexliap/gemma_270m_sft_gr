import polars as pl
from unsloth import FastModel

from datasets import Dataset, load_dataset

dataset = pl.read_parquet("datasets/truthful_qa_greek.parquet")
ds = Dataset.from_dict(dataset.to_dict())

max_seq_length = 2048

model, tokenizer = FastModel.from_pretrained(
    model_name="unsloth/gemma-3-270m-it",
    max_seq_length=max_seq_length,  # Choose any for long context!
    load_in_4bit=False,  # 4 bit quantization to reduce memory
    load_in_8bit=False,  # [NEW!] A bit more accurate, uses 2x memory
    full_finetuning=False,  # [NEW!] We have full finetuning now!
    # token = "hf_...", # use one if using gated models
)


def convert_to_chatml(example: dict):
    return {
        "conversations": [
            # {"role": "system", "content": example["questions"]},
            {"role": "user", "content": example["questions"]},
            {"role": "assistant", "content": example["correct_answers"]},
        ]
    }


ds = ds.map(convert_to_chatml)


def formatting_prompts_func(examples):
    convos = examples["conversations"]
    texts = [
        tokenizer.apply_chat_template(
            convo, tokenize=False, add_generation_prompt=False
        ).removeprefix("<bos>")
        for convo in convos
    ]
    return {"text": texts}


ds = ds.map(formatting_prompts_func, batched=True)

messages = [
    {"role": "system", "content": ds[0]["conversations"][0]["content"]},
    {"role": "user", "content": ds[0]["conversations"][0]["content"]},
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,  # Must add for generation
).removeprefix("<bos>")

# print(model.generate(
#     **tokenizer(text, return_tensors="pt").to("cuda"),
#     max_new_tokens=125,
#     temperature=1,
#     top_p=0.95,
#     top_k=64,
#     use_cache=False,
# ))

from transformers import TextStreamer
_ = model.generate(
    **tokenizer(text, return_tensors = "pt").to("cuda"),
    max_new_tokens = 125,
    temperature = 1, top_p = 0.95, top_k = 64,
    streamer = TextStreamer(tokenizer, skip_prompt = True),
    use_cache=False,
)
