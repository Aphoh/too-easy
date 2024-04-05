import argparse
from too_easy.instrumenter import Instrumenter, Writeable
from too_easy.tensor_writer import TensorStoreWriter
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from tqdm import tqdm
import asyncio
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from pathlib import Path
import torch


def get_dataloader(
    dataset_name: str,
    split: str,
    tokenizer_name: str,
    batch_size: int,
    context_length: int,
    total_samples: int,
    text_field="text",
):
    dset = load_dataset(dataset_name, split=split)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    dset = dset.select(range(total_samples)).map(
        lambda x: tokenizer(
            x[text_field], truncation=True, max_length=context_length, return_tensors="pt"
        ),
        remove_columns=dset.column_names,
        batched=True,
    )
    dset.set_format("torch")
    return DataLoader(dset, batch_size=batch_size)


async def get_tensor_writer(model, output: Path, total_samples: int, dtype: str):
    context_length = get_max_context_length(model)
    num_layers = get_num_layers(model)
    dff = get_dff(model)
    ts = TensorStoreWriter(
        path=output,
        layers=num_layers,
        total_samples=total_samples,
        seq_len=context_length,
        dff=dff,
        dtype=dtype,
    )
    await ts.init_tensorstore()
    return ts


def get_base_model(model_name: str, revision: str, dtype: str):
    return AutoModelForCausalLM.from_config(
        AutoConfig.from_pretrained(model_name, revision=revision), torch_dtype=getattr(torch, dtype)
    )


def get_instrumenter(
    model, pool: ThreadPoolExecutor, writer: TensorStoreWriter, fc1_pattern: str, num_layers: int
) -> Instrumenter:
    return Instrumenter(model, asyncio.get_event_loop(), pool, writer, fc1_pattern, num_layers)


def get_num_layers(model):
    cfg = model.config
    if hasattr(cfg, "num_hidden_layers"):
        return cfg.num_hidden_layers
    if hasattr(cfg, "num_layers"):
        return cfg.num_layers
    raise ValueError("Cannot find the number of layers in the model")


def get_dff(model):
    cfg = model.config
    if hasattr(cfg, "intermediate_size"):
        return cfg.intermediate_size
    raise ValueError("Cannot find the hidden size in the model")


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

    args = parser.parse_args()
    assert args.model is not None, "Please provide a model name."

    model = get_base_model(args.model, args.revision, args.dtype)
    print("Model loaded with type ", next(iter(model.parameters())).dtype)
    out_path = Path(args.output_file)
    print("Outputting to ", out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = await get_tensor_writer(
        model, Path(args.output_file), total_samples=args.total_samples, dtype=args.dtype
    )

    context_length = get_max_context_length(model)
    dataloader = get_dataloader(
        args.dset,
        args.dset_split,
        args.model,
        batch_size=args.batch_size,
        context_length=context_length,
        total_samples=args.total_samples,
    )

    if args.fc1_pattern is None:
        print("No pattern provided, find the name of the layer in this model")
        print(model)
        return

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
        model = model.cuda()
    elif torch.backends.mps.is_available():
        device = "mps"
        model = model.to("mps")

    num_layers = get_num_layers(model)

    with ProcessPoolExecutor() as pool:
        instrumenter = get_instrumenter(model, pool, writer, args.fc1_pattern, num_layers)
        instrumenter.instrument()
        model.eval()
        with torch.inference_mode():
            for batch in tqdm(dataloader):
                n_samples = batch["input_ids"].shape[0]
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                _ = model(input_ids=input_ids, attention_mask=attention_mask)
                instrumenter.step(n_samples)
                await instrumenter.flush()


if __name__ == "__main__":
    asyncio.run(main())
