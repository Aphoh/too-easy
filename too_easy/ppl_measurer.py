import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import DataLoader
from datasets import load_dataset
import torch
import math

def get_dataloader(dataset_name: str, split: str, tokenizer: AutoTokenizer, context_length: int, total_samples: int, text_field="text"):
    dset = load_dataset(dataset_name, split=split)
    dset = dset.select(range(total_samples)).map(
        lambda x: tokenizer(
            [x[text_field]], truncation=True, max_length=context_length+1, return_tensors="pt",
        ),
        remove_columns=dset.column_names,
        batched=False,
    )
    dset.set_format("torch")
    return DataLoader(dset, batch_size=None)


def main():

    parser = argparse.ArgumentParser(
        description="Measure ppl with a given model"
    )
    parser.add_argument(
        "--dset",
        type=str,
        default="stas/c4-en-10k",
        help="Hugging Face dataset to process.",
    )
    parser.add_argument(
        "--dset-split", default="train", type=str, help="The split of the dataset to use."
    )
    parser.add_argument("--model", type=str, required=True, help="The name of the model to use.")
    parser.add_argument(
        "--revision", type=str, default="main", help="The revision of the model to use."
    )
    parser.add_argument("--output-file", type=str, help="The csv file to write the output to")
    parser.add_argument("--total-samples", default=32, type=int, help="total samples to process")
    parser.add_argument(
        "--dtype",
        default="float16",
        type=str,
        help="dtype to use. defaults to float16.",
    )
    parser.add_argument("--device", default=None, type=str, help="device to use")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, revision=args.revision, torch_dtype=getattr(torch, args.dtype))
    device = None
    if args.device:
        device = args.device
    else:
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
    model = model.to(device)
    loader = get_dataloader(args.dset, args.dset_split, tokenizer, 2048, args.total_samples)

    agg_loss = 0
    num_samples = 0
    for elem in loader:
        with torch.no_grad():
            input_ids = elem["input_ids"].to(device)
            labels = input_ids
            attn_mask = elem["attention_mask"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attn_mask, labels=labels)
            loss = outputs.loss.cpu().item()
            print("Loss is ", loss)
            agg_loss += loss
            num_samples += input_ids.shape[0]
            if num_samples >= args.total_samples:
                break
    agg_loss /= num_samples
    print("PPL is ", math.exp(agg_loss))
    with open(args.output_file, "w") as f:
        f.write(f"{args.model},{args.revision},{args.dset},{math.exp(agg_loss)}\n")

if __name__ == "__main__":
    main()

    
