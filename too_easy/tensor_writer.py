import tensorstore as ts
import torch
from pathlib import Path
import numpy as np


class TensorStoreWriter:
    def __init__(self, path: Path, layers: int, bins: np.array):
        """
        Initializes a TensorStoreWriter object to lazily write tensors to disk.

        Parameters:
        - path: The file path for the tensor store.
        - layers: The number of layers in the tensor.
        - b: The total number of samples
        - t: The sequence length.
        - dff: The feature dimension.
        """
        self.path = path
        self.layers = layers
        self.bins = bins

    async def init_tensorstore(self):
        """
        Initializes and returns a TensorStore array with the specified dimensions and settings.
        """
        self.ts_arr = await ts.open(
            {
                "driver": "zarr",
                "kvstore": {"driver": "file", "path": str(self.path)},
                "metadata": {
                    "chunks": [self.layers, len(self.bins) - 1],
                },
            },
            open=True,
            create=True,
            shape=(self.layers, len(self.bins) - 1),
            dtype=ts.int32,
        )

    def write_layer_tensor(self, layer: int, tensor: torch.Tensor):
        """
        Writes a tensor to the TensorStore array at the specified block index.

        Parameters:
        - tensor: The tensor to write, should be size [n_bins]
        """
        tensor = tensor.numpy().astype(np.int32)
        self.ts_arr[layer] += tensor
