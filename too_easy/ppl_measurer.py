import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import DataLoader
from datasets import load_dataset, VerificationMode
import os
import torch
from functools import partial
from tqdm import tqdm
import time
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler


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
    input_ids = torch.tensor(x[text_field], dtype=torch.int)
    attention_mask = torch.ones_like(input_ids)
    return {"input_ids": input_ids, "labels": input_ids, "attention_mask": attention_mask}


def get_dataloader(
    dataset_name: str,
    split: str,
    already_tokenized: bool,
    tokenizer: AutoTokenizer,
    batch_size: int,
    context_length: int,
    total_samples: int,
    world_size: int,
    text_field="text",
):
    tokenizer.add_tokens("<custom padding>")
    tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids("<custom padding>")
    t = time.time() 
    dset = load_dataset(dataset_name, split=split, verification_mode=VerificationMode.NO_CHECKS)
    print(f"Took {time.time() - t} seconds to load data")
    if not already_tokenized:
        dset = dset.select(range(total_samples)).map(
            partial(tokenize_fn, tokenizer, text_field, context_length),
            remove_columns=dset.column_names,
            batched=True,
        )
    else:
        pass
        dset = dset.select(range(total_samples)).map(
            partial(pad_sequence, context_length, text_field),
            remove_columns=dset.column_names,
            batched=False,
        )
    dset.set_format("torch")
    if world_size > 1:
        sampler = DistributedSampler(dset)
        return DataLoader(dset, batch_size=batch_size, sampler=sampler)
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
    parser.add_argument("--batch-size", default=None, type=int, help="total samples to process")
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

    rank = os.getenv("RANK", 0)
    world_size = os.getenv("WORLD_SIZE", 1)
    if world_size > 1 and torch.cuda.is_available():
        dist.init_process_group("nccl", rank=rank, world_size=world_size)

    device = None
    if args.device:
        device = args.device
    else:
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda:" + rank
        elif torch.backends.mps.is_available():
            device = "mps"
    model = model.to(device)
    #model = torch.compile(model)
    
    if args.batch_size is None:
        if "cuda" in device:
            memory = torch.cuda.get_device_properties(0).total_memory
            model_size = sum(p.numel() * p.element_size() for p in model.parameters())
            args.batch_size = max(int(memory / (model_size * 1.5)), 1)
            print("Setting batch size to ", args.batch_size)
        else:
            args.batch_size = 1

    loader = get_dataloader(
        args.dset,
        args.dset_split,
        args.dset_already_tokenized,
        tokenizer,
        args.batch_size,
        context_length,
        args.total_samples,
        world_size=world_size,
        text_field=text_field,
    )

    total_loss = torch.tensor(0.0, device=device)
    num_samples = torch.tensor(0.0, device=device)
    for elem in tqdm(loader, disable=rank != 0):
        with torch.inference_mode():
            num_samples += elem["input_ids"].shape[0]
            outputs = model(**{k: v.to(device) for k, v in elem.items()})
            total_loss += outputs.loss.float()
            if num_samples >= args.total_samples:
                break

    if world_size > 1:
        dist.all_reduce(total_loss)
        dist.all_reduce(num_samples)
    if rank == 1:
        res = (total_loss / num_samples).cpu().item()
        print("Mean loss: ", res.item())

        with open(args.output_file, "a") as f:
            f.write(
                f"{args.model},{args.revision},{args.dset},{res}\n"
            )


if __name__ == "__main__":
    main()