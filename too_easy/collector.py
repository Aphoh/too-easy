import argparse
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
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


def get_base_model(model_name: str):
    return AutoModelForCausalLM.from_pretrained(model_name)


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


def instrument(model, fc1_pattern: str, n_layers: int):
    def fwd_hook(module, input, output):
        in_d, out_d = input[0].shape[-1], output.shape[-1]
        assert (in_d * 4 == out_d) or (in_d * 8 == out_d), "FC1 layer should 4x or 8x the input."
        print("Forward hook called with:")
        print("Input shape:", [a.shape for a in input])
        print("Output shape:", output.shape)

    left, right = fc1_pattern.split(".{}.")
    lefts, rights = left.split("."), right.split(".")
    module = model
    for latt in lefts:
        module = getattr(module, latt)

    for i in range(n_layers):
        i_module = module[i]
        for ratt in rights:
            i_module = getattr(i_module, ratt)
        print("Registering hook for module", i_module)
        i_module.register_forward_hook(fwd_hook)


def main():
    parser = argparse.ArgumentParser(
        description="Process a Hugging Face dataset with a given model."
    )
    parser.add_argument(
        "--dset",
        type=str,
        default="mli-will/long-c4-hq-subset",
        help="The name of the Hugging Face dataset to process.",
    )
    parser.add_argument(
        "--dset-split", default="data", type=str, help="The split of the dataset to use."
    )
    parser.add_argument("--model", type=str, required=True, help="The name of the model to use.")
    parser.add_argument("--fc1-pattern", type=str, help="The pattern for finding the fc1 layers.")
    parser.add_argument("--output-file", type=str, help="The file to write the output to.")
    parser.add_argument("--batch-size", default=8, type=int, help="batch size")
    parser.add_argument("--total-samples", default=32, type=int, help="total samples to process")

    args = parser.parse_args()
    assert args.model is not None, "Please provide a model name."

    model = get_base_model(args.model)

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

    # model = torch.compile(model) # we don't run long enough to compile

    num_layers = get_num_layers(model)
    instrument(model, args.fc1_pattern, num_layers)
    model.eval()
    with torch.inference_mode():
        for batch in tqdm(dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            _ = model(input_ids=input_ids, attention_mask=attention_mask)


if __name__ == "__main__":
    main()
