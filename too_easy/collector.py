import argparse
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

def process_dataset(dataset_name, model_name):
    # Load the dataset
    pass

class StreamedDataset(torch.utils.data.IterableDataset):
    def __init__(self, tokenizer, dset, chunk_length=2048, remove_columns=['text', "meta", "__index_level_0__"]):
        super(StreamedDataset).__init__()
        self.tokenizer = tokenizer
        assert self.tokenizer.eos_token_id
        self.dset = dset.map(
            self.process_fn,
            remove_columns=remove_columns,
            batched=True
        )
        self.dset_iter = iter(self.dset)
        self.chunk_length = chunk_length
        self.data_buffer = []

    def process_fn(self, examples):
        res = self.tokenizer(examples["text"])
        ids = []
        lens = []
        for sentence in res["input_ids"]:
            ids.extend(sentence)
            ids.append(self.tokenizer.eos_token_id)
            lens.append(len(sentence) + 1)
        return {"ids": [ids], "lens": [lens]}
    
    def __iter__(self):
        return self
    
    def __next__(self):
        assert torch.utils.data.get_worker_info() is None   
        if len(self.data_buffer) < self.chunk_length:
            next_data = next(self.dset_iter)
            self.data_buffer
            self.data_buffer.extend(next_data["ids"])
            assert len(self.data_buffer) > self.chunk_length
        res = self.data_buffer[:self.chunk_length]
        self.data_buffer = self.data_buffer[self.chunk_length:]
        return res

def main():
    parser = argparse.ArgumentParser(description='Process a Hugging Face dataset with a given model.')
    parser.add_argument('--dset', type=str, default="DKYoon/SlimPajama-6B", help='The name of the Hugging Face dataset to process.')
    parser.add_argument('--model', type=str, help='The name of the model to use.')
    parser.add_argument("--output-file", type=str, help="The file to write the output to.")
    parser.add_argument('--dset-split', type=str, help='The split of the dataset to use.')
    parser.add_argument('--dset-field', default="text", type=str, help='The text field of dataset entries')
    parser.add_argument('--batch-size', default=128, type=int, help='batch size')

    args = parser.parse_args()

    process_dataset(arg