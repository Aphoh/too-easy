from datasets import load_dataset
from transformers import AutoTokenizer
from argparse import ArgumentParser

def get_tokenizer(model: str, revision: str, use_fast: bool = True):
    return AutoTokenizer.from_pretrained(model, revision=revision, use_fast=use_fast)

def get_dataset(path, split):
    dset = load_dataset(path, split=split, streaming=True)
    return dset

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--revision", default="main", type=str)
    parser.add_argument("--no-fast", action="store_false", dest="use_fast")
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--dset", type=str, required=True)
    parser.add_argument("--split", type=str, required=True)
    parser.add_argument("--eps", type=float, default=1e-7)
    parser.add_argument("--eps-interval", type=int, default=100000)
    parser.add_argument("--add-special-tokens", action="store_true")
    args = parser.parse_args()

    tokenizer = get_tokenizer(args.model, args.revision, args.use_fast)
    dset = get_dataset(args.dset, args.split)

    n_bytes = 0
    n_tokens = 0
    prev_tpb = -1.0
    acc = 0
    for sample in dset:
        n_bytes += len(sample["text"].encode("utf-8"))
        tokens = tokenizer.encode(sample["text"], add_special_tokens=args.add_special_tokens)
        n_tokens += len(tokens)
        acc += len(tokens)
        if acc > args.eps_interval:
            next_tpb = n_tokens / n_bytes
            if abs(next_tpb - prev_tpb) > args.eps:
                print(f"TPB: {next_tpb}")
                prev_tpb = next_tpb
            else:
                with open(args.output_path, "a") as f:
                    output_vals = [args.model, args.revision, args.dset, args.split, args.eps, args.add_special_tokens, n_bytes, n_tokens, next_tpb]
                    f.write(",".join(map(str, output_vals)) + "\n")
                    break
            acc = 0

    print("Final TBP: ", n_tokens / n_bytes, "After processing", n_bytes, "bytes and", n_tokens, "tokens")

