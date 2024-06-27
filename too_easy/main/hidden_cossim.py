import asyncio
from pathlib import Path

import numpy as np
import torch
from torch.nn.functional import cosine_similarity
from tqdm import tqdm
from transformers.modeling_outputs import CausalLMOutputWithPast

from too_easy.common import (
    get_base_model,
    get_dataloader,
    get_max_context_length,
    get_num_layers,
    get_rank,
    get_tensor_writer,
    get_tokenizer,
    print_rank_0,
    setup,
)


async def main():
    accelerator, parser = setup("Measure cosine similarities of hidden states in a model.")

    args, _ = parser.parse_known_args()

    tokenizer = get_tokenizer(args.model, args.revision, use_fast=args.tokenizer_use_fast)
    model = get_base_model(args.model, args.revision, args.dtype)
    print_rank_0("Model loaded with type ", next(iter(model.parameters())).dtype)

    if get_rank() == 0:
        out_path = Path(args.output_file)
        print_rank_0("Outputting to ", out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

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

    num_layers = get_num_layers(model)
    writer = await get_tensor_writer(
        model,
        output_shape=(num_layers, args.total_samples, context_length),
        output=Path(args.output_file),
        output_dtype="float32",
    )

    model, dataloader = accelerator.prepare(model, dataloader)

    loss_agg = 0.0
    loss_count = 0
    samples = 0

    curr_elem = 0
    model.eval()
    with torch.inference_mode():
        t = tqdm(dataloader, desc="Processing ", disable=(get_rank() != 0))
        for i, batch in enumerate(t):
            samples += batch["input_ids"].numel()
            input_ids = batch["input_ids"]
            if "attention_mask" not in batch:
                attn_mask = torch.ones_like(input_ids, device=input_ids.device)
            else:
                attn_mask = batch["attention_mask"]
            outputs: CausalLMOutputWithPast = model(
                input_ids=input_ids,
                attention_mask=attn_mask,
                labels=input_ids,
                output_hidden_states=True,
            )
            hs = torch.stack(outputs.hidden_states)
            ssims = cosine_similarity(hs[1:], hs[:-1], dim=-1).cpu()
            writer.ts_arr[:, curr_elem : curr_elem + input_ids.shape[0], :] = writer.convert_tensor(ssims)
            curr_elem += input_ids.shape[0]

            loss = outputs.loss.detach().clone()
            t.set_description(f"lm loss {outputs.loss:.2f}")
            if outputs.loss > 8:
                print_rank_0("Loss too high, bug")
                return
            accelerator.reduce(loss, reduction="mean")
            loss_agg += loss.item()
            loss_count += 1

    if get_rank() == 0:
        with open(Path(args.output_file).parent / "losses.csv", "a") as f:
            f.write(f"{args.output_file},{loss_agg / loss_count:.5f}\n")

    if args.npy_output:
        Path(args.npy_output).parent.mkdir(parents=True, exist_ok=True)
        np.save(Path(args.npy_output), writer.ts_arr[:] / samples)


if __name__ == "__main__":
    asyncio.run(main())
