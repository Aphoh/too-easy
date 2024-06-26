from pathlib import Path

import tensorstore as ts
import torch


class TensorStoreWriter:
    def __init__(self, path: Path, layers: int, output_shape: tuple, output_dtype: str):
        """
        Initializes a TensorStoreWriter object to lazily write tensors to disk.

        Parameters:
        - path: The file path for the tensor store.
        - output_shape: The shape of the output tensor.
        """
        self.path = path
        self.layers = layers
        if isinstance(output_shape, int):
            output_shape = (output_shape,)
        self.ts_arr_shape = list(output_shape)
        self.output_dtype: ts.dtype = getattr(ts, output_dtype, None)
        if self.output_dtype is None:
            raise ValueError(f"Invalid tensorstore output dtype: {output_dtype}")
        elif not isinstance(self.output_dtype, ts.dtype):
            raise ValueError(f"Invalid tensorstore output dtype: {output_dtype}")

    async def init_tensorstore(self):
        """
        Initializes and returns a TensorStore array with the specified dimensions and settings.
        """
        self.ts_arr = await ts.open(
            {
                "driver": "zarr",
                "kvstore": {"driver": "file", "path": str(self.path)},
                "metadata": {  # TODO: should we allow for chunking in args?
                    "chunks": self.ts_arr_shape,
                },
            },
            open=True,
            create=True,
            shape=self.ts_arr_shape,
            dtype=self.output_dtype,
        )

    def convert_tensor(self, tensor: torch.Tensor):
        assert "cpu" in tensor.device.type, "Tensor must be on CPU"
        return tensor.numpy().astype(self.output_dtype.numpy_dtype)
