import argparse
from .instrumenter import Instrumenter
from .tensor_writer import TensorStoreWriter
from datasets import load_dataset, load_from_disk, Dataset
import os
from torch.utils.data import DataLoader
import psutil
from accelerate import Accelerator
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from tqdm import tqdm
import asyncio
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from pathlib import Path
import torchist
import torch
import torch.distributed as dist
import numpy as np
from typing import Optional


def get_rank():
    return dist.get_rank() if dist.is_initialized() else 0


def print_rank_0(*args, **kwargs):
    if get_rank() == 0:
        mem = psutil.virtual_memory()[2]
        print(f"CPU: {mem}%", *args, **kwargs)


def get_tokenizer(model: str, revision: str, use_fast: bool = True):
    if "opt" in model.lower() and use_fast:
        print_rank_0("Warning: Using fast tokenizer with OPT model, this may not work.")
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
    dset = None
    if get_rank() == 0 and (not cache_path or (cache_path and not Path(cache_path).exists())):
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
                    print_rank_0(f"WARN: Ran out of samples in the dataset! Got {len(rows)} rows")
                    no_data = True
                    break
                rows[i].extend(input_ids)
            rows[i] = rows[i][:context_length]
            if no_data:
                break

        dset = Dataset.from_dict({"input_ids": rows})
        if cache_path:
            dset.save_to_disk(cache_path)

    if dist.is_initialized():
        dist.barrier()

    if dset is None:
        assert cache_path and Path(cache_path).exists()
        dset = load_from_disk(cache_path)

    assert len(dset) == total_samples, "Dataset length does not match the requested length."
    dset.set_format("torch")
    return DataLoader(dset, batch_size=batch_size)


async def get_tensor_writer(model, bins: torch.Tensor, output: Path):
    num_layers = get_num_layers(model)
    ffn_size = model.config.intermediate_size
    ts = TensorStoreWriter(
        path=output,
        layers=num_layers,
        output_shape=(ffn_size,),
        output_dtype="uint32",  # output_dtype="float32"
    )
    await ts.init_tensorstore()
    return ts


def get_base_model(model_name: str, revision: str, dtype: str):
    attn_impl = "eager"
    try:
        import flash_attn

        attn_impl = "flash_attention_2"
    except ImportError:
        print_rank_0("Flash attention not found. Using default attention")

    kwargs = {
        "pretrained_model_name_or_path": model_name,
        "revision": revision,
        "attn_implementation": attn_impl,
        "torch_dtype": getattr(torch, dtype),
    }
    dtype = getattr(torch, dtype)
    return AutoModelForCausalLM.from_pretrained(**kwargs, low_cpu_mem_usage=True).to(dtype)


def histogram_transform(bins: torch.Tensor, thresh=0.0):
    def closure(tensor: torch.Tensor):
        try:
            if tensor.device.type == "cuda":
                torch.cuda.synchronize(tensor.device)
            # res = tensor.abs().view(-1, tensor.shape[-1]).sum(dim=0)
            res = (tensor > thresh).view(-1, tensor.shape[-1]).sum(dim=0)
        except Exception as e:
            print_rank_0(e)
            return torch.zeros(tensor.shape[-1])
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
    if hasattr(model, "module"):
        model = model.module
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
    accelerator = Accelerator()
    parser = argparse.ArgumentParser(
        description="Process a Hugging Face dataset with a given model."
    )
    parser.add_argument(
        "--dset",
        type=str,
        default="mit-han-lab/pile-val-backup",
        help="Hugging Face dataset to process.",
    )
    parser.add_argument(
        "--dset-split", default="validation", type=str, help="The split of the dataset to use."
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
    parser.add_argument(
        "--npy-output", type=str, help="Write a file with the numpy output.", default=None
    )

    args, _ = parser.parse_known_args()

    tokenizer = get_tokenizer(args.model, args.revision, use_fast=args.tokenizer_use_fast)
    print_rank_0("Getting base model")
    model = get_base_model(args.model, args.revision, args.dtype)
    print_rank_0("Model loaded with type ", next(iter(model.parameters())).dtype)

    if get_rank() == 0:
        out_path = Path(args.output_file)
        print_rank_0("Outputting to ", out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

    bins = torch.logspace(-8, np.log2(25), steps=250, dtype=torch.float32, base=2.0)
    bins = torch.cat([bins, torch.tensor([float("inf")])])
    bins = torch.cat([-torch.flip(bins, (0,)), torch.tensor([0.0]), bins])
    if get_rank() == 0:
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
        print_rank_0("No pattern provided, find the name of the layer in this model")
        print_rank_0(model)
        return

    print_rank_0("Wrapping model")

    model, dataloader = accelerator.prepare(model, dataloader)

    unwrapped_model = model
    if hasattr(model, "module"):
        unwrapped_model = model.module

    num_layers = get_num_layers(model)
    loss_agg = 0.0
    loss_count = 0
    samples = 0

    with ThreadPoolExecutor() as pool:
        instrumenter = get_instrumenter(
            unwrapped_model, pool, writer, args.fc1_pattern, num_layers, bins
        )
        # instrumenter.instrument()
        model.eval()
        with torch.inference_mode():
            t = tqdm(dataloader, desc="Processing ", disable=(get_rank() != 0))
            for batch in t:
                samples += batch["input_ids"].numel()
                input_ids = batch["input_ids"]
                if "attention_mask" not in batch:
                    attn_mask = torch.ones_like(input_ids, device=input_ids.device)
                else:
                    attn_mask = batch["attention_mask"]
                outputs = model(input_ids=input_ids, attention_mask=attn_mask, labels=input_ids)
                loss = outputs.loss.detach().clone()
                t.set_description(f"lm loss {outputs.loss:.2f}")
                if outputs.loss > 8:
                    print_rank_0("Loss too high, bug")
                    return
                accelerator.reduce(loss, reduction="mean")
                loss_agg += loss.item()
                loss_count += 1
                instrumenter.step(batch["input_ids"].shape[0])
            await instrumenter.flush()
    if get_rank() == 0:
        with open(Path(args.output_file).parent / "losses.csv", "a") as f:
            f.write(f"{args.output_file},{loss_agg / loss_count:.5f}\n")

    if args.npy_output:
        Path(args.npy_output).parent.mkdir(parents=True, exist_ok=True)
        np.save(Path(args.npy_output), writer.ts_arr[:] / samples)


if __name__ == "__main__":
    asyncio.run(main())
