import tensorstore as ts
from ml_dtypes import bfloat16


class TensorStoreWriter:
    def __init__(self, path, layers, b, t, dff):
        """
        Initializes a TensorStoreWriter object to lazily write tensors to disk.

        Parameters:
        - path: The file path for the tensor store.
        - layers: The number of layers in the tensor.
        - b: The total number of batches
        - t: The sequence length.
        - dff: The feature dimension.
        - dtype: The data type of the tensor (defaults to torch.bfloat16).
        """
        self.path = path
        self.layers = layers
        self.b = b
        self.curr_b = 0
        self.t = t
        self.dff = dff
        self.ts_arr = self._init_tensorstore()

    async def _init_tensorstore(self):
        """
        Initializes and returns a TensorStore array with the specified dimensions and settings.
        """
        return await ts.open(
            {
                "driver": "zarr",
                "kvstore": {"driver": "file", "path": self.path},
                "metadata": {
                    "chunks": [1, self.b, self.t, self.dff],
                    "dtype": "bfloat16",
                },
            },
            create=True,
            shape=(self.layers, self.b, self.t, self.dff),
        )

    async def write_tensor(self, tensor):
        """
        Writes a tensor to the TensorStore array at the specified block index.

        Parameters:
        - tensor: The tensor to write, should be size [layers, nb, t, dff]
        - block_index: The block index (within nb) where the tensor should be written.
        """
        nb = tensor.shape[1]
        assert self.curr_b + nb <= self.b
        await self.ts_arr[:, self.curr_b:self.curr_b + nb, :, :].write(
            tensor.cpu().float().numpy().astype(bfloat16)
        )
        self.curr_b += nb
        return self.curr_b
