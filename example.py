from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
import torch

# Load model & tokenizer
model_name = "alexliap/gemma-3-270m-it-stf-gr"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name, dtype=torch.bfloat16, device_map="cuda"
)

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
