import os
from unsloth import FastModel
from transformers import TextStreamer

os.environ["TOKENIZERS_PARALLELISM"] = "True"

model, tokenizer = FastModel.from_pretrained(
    model_name = "alexliap/gemma3_270m_sft_gr", # YOUR MODEL YOU USED FOR TRAINING
    max_seq_length = 2048,
    load_in_4bit=False,
    load_in_8bit=False,
    full_finetuning=False,
    use_cache=False,
)

messages = [
    {'role': 'system','content': "Είσαι ένας χρήσιμος βοηθός που απαντά ερωτήσεις."},
    {"role" : 'user', 'content': "Τι είναι κοινωνικότητα;"}
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize = False,
    add_generation_prompt = True, # Must add for generation
).removeprefix('<bos>')

model.generate(
    **tokenizer(text, return_tensors = "pt").to("cuda"),
    max_new_tokens = 1024,
    temperature = 0.2, top_k = 64,
    use_cache=False,
    streamer = TextStreamer(tokenizer, skip_prompt = True),
)