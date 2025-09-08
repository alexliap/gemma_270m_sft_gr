import polars as pl

from datasets import load_dataset

dataset = load_dataset(path="ilsp/truthful_qa_greek", name="generation")

questions = dataset.data["train"]["question"].to_numpy().tolist()

correct_answers = dataset.data["train"]["correct_answers"].to_numpy()
correct_answers = [answer.tolist() for answer in correct_answers]

truthful_qa_gr = pl.DataFrame(
    {"questions": questions, "correct_answers": correct_answers}
)

truthful_qa_gr.write_parquet("datasets/truthful_qa_greek.parquet")
