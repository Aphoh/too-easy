import argparse
from pathlib import Path
from typing import List, Optional

import psutil
import torch
import torch.distributed as dist
from accelerate import Accelerator
from datasets import Dataset, load_dataset, load_from_disk
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from too_easy.tensor_writer import TensorStoreWriter


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
        rows: List[List[int]] = []
        i = 0
        for i in tqdm(range(total_samples), desc="Creating dataset rows"):
            rows.append([])
            no_data = False
            while len(rows[i]) < context_length:
                try:
                    text = next(dset_iter)[text_field]
                    input_ids = tokenizer(
                        text,
                        truncation=True,
                        max_length=context_length,
                        add_special_tokens=False,
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


async def get_tensor_writer(model, output_shape, output: Path, output_dtype: str = "uint32"):
    num_layers = get_num_layers(model)
    ts = TensorStoreWriter(path=output, layers=num_layers, output_shape=output_shape, output_dtype=output_dtype)
    await ts.init_tensorstore()
    return ts


def get_base_model(model_name: str, revision: str, dtype: str):
    attn_impl = "eager"
    try:
        import flash_attn  # noqa: F401

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


def setup(description):
    accelerator = Accelerator()
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--dset",
        type=str,
        default="mit-han-lab/pile-val-backup",
        help="Hugging Face dataset to process.",
    )
    parser.add_argument(
        "--dset-split",
        default="validation",
        type=str,
        help="The split of the dataset to use.",
    )
    parser.add_argument(
        "--dset-cache-path",
        type=str,
        default=None,
        help="The cache path for the dataset.",
    )

    parser.add_argument("--model", type=str, required=True, help="The name of the model to use.")
    parser.add_argument(
        "--revision",
        type=str,
        default="main",
        help="The revision of the model to use.",
    )
    parser.add_argument(
        "--fc1-pattern",
        type=str,
        help="The pattern for finding the fc1 layers.",
    )
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
        "--npy-output",
        type=str,
        help="Write a file with the numpy output.",
        default=None,
    )

    return accelerator, parser
