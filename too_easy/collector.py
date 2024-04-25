import argparse
from too_easy.instrumenter import Instrumenter
from too_easy.tensor_writer import TensorStoreWriter
from datasets import load_dataset, load_from_disk, Dataset
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from tqdm import tqdm
import asyncio
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from pathlib import Path
import torchist
import os
import torch
import torch.distributed as dist
import numpy as np
from typing import Optional


def get_tokenizer(model: str, revision: str, use_fast: bool = True):
    if "opt" in model.lower() and use_fast:
        print("Warning: Using fast tokenizer with OPT model, this may not work.")
    return AutoTokenizer.from_pretrained(model, revision=revision, use_fast=use_fast)


def get_dataloader(
    dataset_name: str,
    cache_path: Optional[str],
    split: str,
    tokenizer: AutoTokenizer,
    batch_size: int,
    context_length: int,
    total_samples: int,
    text_field="text",
    append_eod=False,
):
    if cache_path and Path(cache_path).exists():
        dset = load_from_disk(cache_path)
    else:
        dset = load_dataset(dataset_name, split=split, streaming=True)
        dset_iter = iter(dset)
        rows = []
        i = 0
        for i in tqdm(range(total_samples), desc="Creating dataset rows"):
            rows.append([])
            no_data = False
            while len(rows[i]) < context_length:
                try:
                    text = next(dset_iter)[text_field]
                    input_ids = tokenizer(
                        text, truncation=True, max_length=context_length, add_special_tokens=False
                    )["input_ids"]
                    input_ids.insert(0, tokenizer.bos_token_id)
                    if append_eod:
                        input_ids.append(tokenizer.eos_token_id)
                except StopIteration:
                    print(f"WARN: Ran out of samples in the dataset! Got {len(rows)} rows")
                    no_data = True
                    break
                rows[i].extend(input_ids)
            if no_data:
                break

            rows[i] = rows[i][:context_length]
        dset = Dataset.from_dict({"input_ids": rows})
        if cache_path:
            print("Saving dataset cache to disk at ", cache_path)
            dset.save_to_disk(cache_path)

    dset.set_format("torch")
    return DataLoader(dset, batch_size=batch_size)


async def get_tensor_writer(model, bins: torch.Tensor, output: Path):
    num_layers = get_num_layers(model)
    ts = TensorStoreWriter(
        path=output, layers=num_layers, output_shape=(bins.shape[0] - 1,), output_dtype="uint32"
    )
    await ts.init_tensorstore()
    return ts


def get_base_model(model_name: str, revision: str, dtype: str):
    attn_impl = "eager"
    try:
        import flash_attn

        attn_impl = "flash_attention_2"
    except ImportError:
        print("Flash attention not found. Using default attention")

    dtype = getattr(torch, dtype)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        revision=revision,
        attn_implementation=attn_impl,
    ).to(dtype)

    return model

def histogram_transform(bins: torch.Tensor):
    def closure(tensor: torch.Tensor):
        try:
            if tensor.device.type == "cuda":
                torch.cuda.synchronize(tensor.device)
            res = torchist.histogram(tensor, edges=bins)
        except Exception as e:
            print(e)
            return torch.zeros(bins.shape[0] - 1)
        return res

    return closure


def get_instrumenter(
    model,
    pool: ThreadPoolExecutor,
    writer: TensorStoreWriter,
    fc1_pattern: str,
    num_layers: int,
    bins: torch.Tensor,
) -> Instrumenter:
    transform = histogram_transform(bins)
    return Instrumenter(
        model, transform, asyncio.get_event_loop(), pool, writer, fc1_pattern, num_layers
    )


def get_num_layers(model):
    cfg = model.config
    if hasattr(cfg, "num_hidden_layers"):
        return cfg.num_hidden_layers
    if hasattr(cfg, "num_layers"):
        return cfg.num_layers
    raise ValueError("Cannot find the number of layers in the model")


def get_max_context_length(model):
    cfg = model.config
    if hasattr(cfg, "max_position_embeddings"):
        return cfg.max_position_embeddings
    raise ValueError("Cannot find the maximum context length in the model")


async def main():
    parser = argparse.ArgumentParser(
        description="Process a Hugging Face dataset with a given model."
    )
    parser.add_argument(
        "--dset",
        type=str,
        default="mli-will/long-c4-hq-subset",
        help="Hugging Face dataset to process.",
    )
    parser.add_argument(
        "--dset-split", default="data", type=str, help="The split of the dataset to use."
    )
    parser.add_argument(
        "--dset-cache-path", type=str, default=None, help="The cache path for the dataset."
    )

    parser.add_argument("--model", type=str, required=True, help="The name of the model to use.")
    parser.add_argument(
        "--revision", type=str, default="main", help="The revision of the model to use."
    )
    parser.add_argument("--fc1-pattern", type=str, help="The pattern for finding the fc1 layers.")
    parser.add_argument("--output-file", type=str, help="The file to write the output to.")
    parser.add_argument("--batch-size", default=8, type=int, help="batch size")
    parser.add_argument("--total-samples", default=32, type=int, help="total samples to process")
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        type=str,
        help="dtype to use. defaults to bfloat16.",
    )
    parser.add_argument(
        "--tokenizer-no-fast",
        action="store_false",
        dest="tokenizer_use_fast",
        help="Use the fast tokenizer.",
    )

    rank = 0
    world_size = 1
    device = torch.device("cpu")
    if (rank := os.environ.get("RANK", None)) is not None:
        rank = int(rank)
        world_size = int(os.environ.get("WORLD_SIZE"))
        local_rank = int(os.environ.get("LOCAL_RANK"))
        torch.cuda.set_device(local_rank)
        device = torch.cuda.current_device()
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    elif torch.cuda.is_available():
        device = torch.cuda.current_device()



    args = parser.parse_args()
    args.total_samples //= world_size
    assert args.model is not None, "Please provide a model name."

    tokenizer = get_tokenizer(args.model, args.revision, use_fast=args.tokenizer_use_fast)
    model = get_base_model(args.model, args.revision, args.dtype)
    print("Model loaded with type ", next(iter(model.parameters())).dtype)

    out_path = Path(args.output_file)
    print("Outputting to ", out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    bins = torch.logspace(-8, np.log2(5), steps=250, dtype=torch.float32, base=2.0)
    bins = torch.cat([-torch.flip(bins, (0,)), torch.tensor([0.0]), bins])
    torch.save(bins, Path(args.output_file).parent / "bins.pt")
    writer = await get_tensor_writer(
        model,
        bins,
        output=Path(args.output_file),
    )

    context_length = get_max_context_length(model)
    dataloader = get_dataloader(
        args.dset,
        args.dset_cache_path,
        args.dset_split,
        tokenizer,
        batch_size=args.batch_size,
        context_length=context_length,
        total_samples=args.total_samples,
    )

    if args.fc1_pattern is None:
        print("No pattern provided, find the name of the layer in this model")
        print(model)
        return

    model = model.to(device)
    if dist.is_initialized():
        model = FSDP(model)

    num_layers = get_num_layers(model)

    with ThreadPoolExecutor() as pool:
        instrumenter = get_instrumenter(model, pool, writer, args.fc1_pattern, num_layers, bins)
        instrumenter.instrument()
        model.eval()
        with torch.inference_mode():
            t = tqdm(dataloader, desc="Processing ", disable=(rank != 0))
            for batch in t:
                n_samples = batch["input_ids"].shape[0]
                input_ids = batch["input_ids"].to(device)
                if "attention_mask" not in batch:
                    attn_mask = torch.ones_like(input_ids, device=device)
                else:
                    attn_mask = batch["attention_mask"].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attn_mask, labels=input_ids)
                t.set_description(f"lm loss {outputs.loss:.2f}")
                if outputs.loss > 6:
                    print("Loss too high, bug")
                    return
                instrumenter.step(n_samples)
            await instrumenter.flush()

    if rank == 0:
        dist.destroy_process_group()


if __name__ == "__main__":
    asyncio.run(main())
