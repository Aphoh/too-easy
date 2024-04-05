import tensorstore as ts
import torch
from pathlib import Path
import numpy as np
from ml_dtypes import bfloat16


class TensorStoreWriter:
    def __init__(
        self, path: Path, layers: int, total_samples: int, seq_len: int, dff: int, dtype: str
    ):
        """
        Initializes a TensorStoreWriter object to lazily write tensors to disk.

        Parameters:
        - path: The file path for the tensor store.
        - layers: The number of layers in the tensor.
        - b: The total number of samples
        - t: The sequence length.
        - dff: The feature dimension.
        - dtype: The data type of the tensor
        """
        self.path = path
        self.layers = layers
        self.total_samples = total_samples
        self.curr_b = 0
        self.seq_len = seq_len
        self.dff = dff
        self.dtype = dtype
        if dtype == "bfloat16":
            self.numpy_type = bfloat16
        else:
            self.numpy_type = getattr(np, dtype)

    async def init_tensorstore(self):
        """
        Initializes and returns a TensorStore array with the specified dimensions and settings.
        """
        self.ts_arr = await ts.open(
            {
                "driver": "zarr",
                "kvstore": {"driver": "file", "path": str(self.path)},
                "metadata": {
                    "chunks": [1, self.total_samples, self.seq_len, self.dff],
                },
            },
            open=True,
            create=True,
            shape=(self.layers, self.total_samples, self.seq_len, self.dff),
            dtype=getattr(ts, self.dtype),
        )

    def write_layer_tensor(self, layer: int, sample_idx: int, tensor: torch.Tensor):
        """
        Writes a tensor to the TensorStore array at the specified block index.

        Parameters:
        - tensor: The tensor to write, should be size [nb, t, dff]
        - block_index: The block index (within nb) where the tensor should be written.
        """
        nb = tensor.shape[0]
        assert sample_idx < self.total_samples
        if self.dtype == "bfloat16":
            tensor = tensor.float()
        tensor = tensor.numpy().astype(self.numpy_type)
        self.ts_arr[layer, sample_idx : sample_idx + nb, :, :] = tensor
