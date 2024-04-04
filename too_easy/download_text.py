import torch
from datasets import load_dataset
from transformers import AutoTokenizer

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
    

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    dset = load_dataset("wikitext", "wikitext-2-v1")
    dset = dset["train"]
