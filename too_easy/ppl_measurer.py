import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import DataLoader
from datasets import load_dataset
import torch
from functools import partial
from tqdm import tqdm


def tokenize_fn(tokenizer, text_field, context_length, x):
    res = tokenizer(
        x[text_field],
        truncation=True,
        max_length=context_length,
        return_tensors="pt",
        padding="max_length",
    )
    res["labels"] = res["input_ids"].clone()
    res["labels"][res["input_ids"] == tokenizer.pad_token_id] = -100
    return res


def pad_sequence(context_length, text_field, x):
    batch_size = len(x[text_field])
    input_ids = torch.zeros(batch_size, context_length, dtype=torch.long)
    labels = -100 * torch.ones(batch_size, context_length, dtype=torch.long)
    for i, elem in enumerate(x[text_field]):
        input_ids[i, : len(elem)] = elem
        labels[i, : len(elem)] = elem
    attention_mask = torch.ones_like(input_ids)
    return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}


def get_dataloader(
    dataset_name: str,
    split: str,
    already_tokenized: bool,
    tokenizer: AutoTokenizer,
    batch_size: int,
    context_length: int,
    total_samples: int,
    text_field="text",
):
    tokenizer.add_tokens("<custom padding>")
    tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids("<custom padding>")
    dset = load_dataset(dataset_name, split=split)
    if not already_tokenized:
        dset = dset.select(range(total_samples)).map(
            partial(tokenize_fn, tokenizer, text_field, context_length),
            remove_columns=dset.column_names,
            batched=True,
        )
    else:
        dset = dset.select(range(total_samples)).map(
            partial(pad_sequence, context_length, text_field),
            batched=True,
        )
    dset.set_format("torch")
    return DataLoader(dset, batch_size=batch_size)


def main():

    parser = argparse.ArgumentParser(description="Measure ppl with a given model")
    parser.add_argument(
        "--dset",
        type=str,
        default="stas/c4-en-10k",
        help="Hugging Face dataset to process.",
    )
    parser.add_argument(
        "--dset-already-tokenized",
        action="store_true",
        help="Whether the dataset is already tokenized.",
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
    parser.add_argument("--batch-size", default=8, type=int, help="total samples to process")
    parser.add_argument(
        "--dtype",
        default="float16",
        type=str,
        help="dtype to use. defaults to float16.",
    )
    parser.add_argument("--device", default=None, type=str, help="device to use")
    args = parser.parse_args()

    context_length = 2048
    text_field = "text"
    if args.dset == "EleutherAI/pythia-memorized-evals":
        args.dset_already_tokenized = True
        context_length = 64
        text_field = "tokens"

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, revision=args.revision, torch_dtype=getattr(torch, args.dtype)
    )
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
    loader = get_dataloader(
        args.dset,
        args.dset_split,
        args.dset_already_tokenized,
        tokenizer,
        args.batch_size,
        context_length,
        args.total_samples,
        text_field=text_field,
    )

    losses = []
    num_samples = 0
    for elem in tqdm(loader):
        with torch.no_grad():
            num_samples += elem["input_ids"].shape[0]
            outputs = model(**{k: v.to(device) for k, v in elem.items()})
            loss = outputs.loss.cpu().item()
            losses.append(loss)
            if num_samples >= args.total_samples:
                break

    losses = torch.tensor(losses)
    ppls = losses.exp()
    mean_loss, std_loss = losses.mean().item(), losses.std().item()
    mean_ppl, std_ppl = ppls.mean().item(), ppls.std().item()
    print(f"Mean loss: {mean_loss}, std loss: {std_loss}")
    print(f"Mean ppl: {mean_ppl}, std ppl: {std_ppl}")

    with open(args.output_file, "w") as f:
        f.write(
            f"{args.model},{args.revision},{args.dset},{mean_loss},{std_loss},{mean_ppl},{std_ppl}\n"
        )


if __name__ == "__main__":
    main()
