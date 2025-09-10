import polars as pl
from datasets import Dataset, load_dataset
from transformers import PreTrainedTokenizerFast


def convert_to_chatml(entry: dict, dataset: str):
    if dataset == "truthful_qa_gr":
        prompt = entry["questions"]
        expected_answer = entry["correct_answers"]

    elif dataset == "medical_mcqa_gr":
        prompt = entry["inputs"] + "\n\n" + "\n".join(entry["multiple_choice_targets"])
        expected_answer = entry["targets"][0]

    return {
        "conversations": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": expected_answer},
        ]
    }


def formatting_prompts_func(entry: dict, tokenizer: PreTrainedTokenizerFast):
    convos = entry["conversations"]
    texts = [
        tokenizer.apply_chat_template(
            convo, tokenize=False, add_generation_prompt=False
        ).removeprefix("<bos>")
        for convo in convos
    ]
    return {"text": texts}


def truthful_qa_gr(tokenizer: PreTrainedTokenizerFast) -> Dataset:
    dataset = load_dataset(path="ilsp/truthful_qa_greek", name="generation")

    questions = dataset.data["train"]["question"].to_numpy().tolist()

    correct_answers = dataset.data["train"]["correct_answers"].to_numpy()
    correct_answers = [answer.tolist() for answer in correct_answers]

    dataset = pl.DataFrame(
        {"questions": questions, "correct_answers": correct_answers}
    ).explode("correct_answers")

    dataset = Dataset.from_dict(dataset.to_dict())

    dataset = dataset.map(convert_to_chatml, fn_kwargs={"dataset": "truthful_qa_gr"})
    
    dataset = dataset.map(formatting_prompts_func, fn_kwargs={"tokenizer": tokenizer}, batched=True)

    dataset = dataset.remove_columns(['questions', 'correct_answers', 'conversations'])

    return dataset


def medical_mcqa_gr(tokenizer: PreTrainedTokenizerFast, split: str) -> Dataset:
    dataset = load_dataset(path="ilsp/medical_mcqa_greek", split=split)
    dataset = dataset.remove_columns(["idx", "multiple_choice_scores", "subject"])

    dataset = dataset.map(convert_to_chatml, fn_kwargs={"dataset": "medical_mcqa_gr"})
    dataset = dataset.map(formatting_prompts_func, fn_kwargs={"tokenizer": tokenizer}, batched=True)

    dataset = dataset.remove_columns(['inputs', 'targets', 'multiple_choice_targets', 'conversations'])

    return dataset
