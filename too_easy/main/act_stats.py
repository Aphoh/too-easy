from concurrent.futures import ThreadPoolExecutor
from too_easy.instrumenter import Instrumenter
from too_easy.tensor_writer import TensorStoreWriter
from tqdm import tqdm
import asyncio
from pathlib import Path
import torch
import numpy as np


from too_easy.common import (
    get_rank,
    print_rank_0,
    get_num_layers,
    get_max_context_length,
    get_tokenizer,
    get_base_model,
    get_dataloader,
    get_tensor_writer,
    setup,
)


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


async def main():
    accelerator, parser = setup("Measure activation statistics for a model.")
    parser.add_argument(
        "--fc1-pattern", type=str, help="The pattern to match the layer name for the FC1 layer."
    )
    parser.add_argument("--output-file", type=str, help="The file to write the output to.")

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
        instrumenter.instrument()
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
