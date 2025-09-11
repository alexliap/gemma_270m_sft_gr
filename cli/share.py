from huggingface_hub import login
import argparse
from unsloth import FastModel


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload model to the HF hub.")
    parser.add_argument("--hf_token", required=True)
    parser.add_argument("--local_model_dir", required=True)
    parser.add_argument("--hf_repo", required=True)
    
    args = parser.parse_args()

    login(token=args.hf_token)

    model, tokenizer = FastModel.from_pretrained(
        model_name = args.local_model_dir,
        max_seq_length = 2048,
        load_in_4bit=False,  # 4 bit quantization to reduce memory
        load_in_8bit=False,  # [NEW!] A bit more accurate, uses 2x memory
        full_finetuning=False, 
    )

    model.push_to_hub_merged(args.hf_repo, tokenizer=tokenizer, save_method="merged_16bit")
    