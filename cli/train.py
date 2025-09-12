import logging
import os

import torch
from unsloth import FastModel
from unsloth.chat_templates import train_on_responses_only
from datasets import concatenate_datasets
from transformers import PreTrainedTokenizerFast
from trl import SFTConfig, SFTTrainer

from gemma_finetune.data_import import (
    el_wiki_qa,
    greek_civics_qa,
    medical_mcqa_gr,
    truthful_qa_gr,
    belebele_gr,
)


os.environ["UNSLOTH_RETURN_LOGITS"] = "1"


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def make_datasets(tokenizer: PreTrainedTokenizerFast):
    ds_1 = truthful_qa_gr(tokenizer=tokenizer)
    ds_1 = ds_1.train_test_split(test_size=0.25, shuffle=True)

    ds_2 = medical_mcqa_gr(tokenizer=tokenizer, split="train")
    ds_2 = ds_2.train_test_split(test_size=0.25, shuffle=True)

    ds_3 = greek_civics_qa(tokenizer=tokenizer)
    ds_3 = ds_3.train_test_split(test_size=0.25, shuffle=True)

    ds_4 = el_wiki_qa(tokenizer=tokenizer)
    ds_4 = ds_4.train_test_split(test_size=0.25, shuffle=True)

    ds_5 = belebele_gr(tokenizer=tokenizer)
    ds_5 = ds_5.train_test_split(test_size=0.25, shuffle=True)

    ds_train = concatenate_datasets(
        [ds_1["train"], ds_2["train"], ds_3["train"], ds_4["train"], ds_5["train"]]
    )

    ds_val = concatenate_datasets(
        [ds_1["test"], ds_2["test"], ds_3["test"], ds_4["test"], ds_5["test"]]
    )

    logger.info(f"Total training entries in dataset: {len(ds_train)}")

    logger.info(f"Total validation entries in dataset: {len(ds_val)}")

    return ds_train, ds_val


if __name__ == "__main__":
    model, tokenizer = FastModel.from_pretrained(
        model_name="unsloth/gemma-3-270m-it",
        load_in_4bit=False,  # 4 bit quantization to reduce memory
        load_in_8bit=False,  # [NEW!] A bit more accurate, uses 2x memory
        full_finetuning=False,  # [NEW!] We have full finetuning now!
        use_cache=False,
    )

    model = FastModel.get_peft_model(
        model,
        r=128,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=256,
        lora_dropout=0,  # Supports any, but = 0 is optimized
        bias="none",  # Supports any, but = "none" is optimized
        use_gradient_checkpointing="unsloth",  # True or "unsloth" for very long context
        random_state=0,
        use_rslora=False,  # We support rank stabilized LoRA
        loftq_config=None,  # And LoftQ
    )

    train, val = make_datasets(tokenizer=tokenizer)

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train,
        eval_dataset=val,  # Can set up evaluation!
        args=SFTConfig(
            dataset_text_field="text",
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            gradient_accumulation_steps=2,  # Use GA to mimic batch size!
            warmup_steps=5,
            num_train_epochs=1,  # Set this for 1 full training run.
            do_eval=True,
            eval_on_start=True,
            eval_strategy="steps",
            eval_steps=0.2,
            learning_rate=2e-5,  # Reduce to 2e-5 for long training runs
            logging_steps=0.05,
            optim="adamw_torch",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=0,
            output_dir="outputs_only",
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
