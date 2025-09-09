import os
import logging

import polars as pl
import torch
from unsloth import FastModel
from unsloth.chat_templates import train_on_responses_only
from trl import SFTConfig, SFTTrainer

from datasets import Dataset

os.environ['UNSLOTH_RETURN_LOGITS'] = '1'

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def convert_to_chatml(example: dict):
    return {
        "conversations": [
            {"role": "user", "content": example["questions"]},
            {"role": "assistant", "content": example["correct_answers"]},
        ]
    }


def formatting_prompts_func(examples):
    convos = examples["conversations"]
    texts = [
        tokenizer.apply_chat_template(
            convo, tokenize=False, add_generation_prompt=False
        ).removeprefix("<bos>")
        for convo in convos
    ]
    return {"text": texts}


if __name__ == "__main__":
    dataset = pl.read_parquet("datasets/truthful_qa_greek.parquet")
    ds = Dataset.from_dict(dataset.to_dict())

    model, tokenizer = FastModel.from_pretrained(
        model_name="unsloth/gemma-3-270m-it",
        max_seq_length=2048,  # Choose any for long context!
        load_in_4bit=False,  # 4 bit quantization to reduce memory
        load_in_8bit=False,  # [NEW!] A bit more accurate, uses 2x memory
        full_finetuning=False,  # [NEW!] We have full finetuning now!
        # token = "hf_...", # use one if using gated models
        use_cache=False,
    )

    model = FastModel.get_peft_model(
        model,
        r = 128, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj",],
        lora_alpha = 256,
        lora_dropout = 0, # Supports any, but = 0 is optimized
        bias = "none",    # Supports any, but = "none" is optimized
        use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
        random_state = 0,
        use_rslora = False,  # We support rank stabilized LoRA
        loftq_config = None, # And LoftQ
    )

    ds = ds.map(convert_to_chatml)
    ds = ds.map(formatting_prompts_func, batched=True)
    ds = ds.train_test_split(test_size=0.25, shuffle=True)

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=ds['train'],
        eval_dataset=ds['test'],  # Can set up evaluation!
        args=SFTConfig(
            dataset_text_field="text",
            per_device_train_batch_size=32,
            gradient_accumulation_steps=1,  # Use GA to mimic batch size!
            warmup_steps=5,
            num_train_epochs = 1, # Set this for 1 full training run.
            # max_steps=100,
            do_eval=True,
            eval_on_start=True,
            eval_strategy="steps",
            eval_steps=0.2,
            learning_rate=2e-5,  # Reduce to 2e-5 for long training runs
            logging_steps=1,
            optim="adamw_torch",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=0,
            output_dir="outputs",
            report_to="none",  # Use this for WandB etc
        ),
    )

    trainer = train_on_responses_only(
        trainer,
        instruction_part="<start_of_turn>user\n",
        response_part="<start_of_turn>model\n",
    )

    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)

    logger.info(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    logger.info(f"{start_gpu_memory} GB of memory reserved.")

    trainer_stats = trainer.train()

    used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
    used_percentage = round(used_memory / max_memory * 100, 3)
    lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)

    logger.info(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
    logger.info(
        f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training."
    )
    logger.info(f"Peak reserved memory = {used_memory} GB.")
    logger.info(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
    logger.info(f"Peak reserved memory % of max memory = {used_percentage} %.")
    logger.info(
        f"Peak reserved memory for training % of max memory = {lora_percentage} %."
    )
