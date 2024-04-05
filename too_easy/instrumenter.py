from collections import defaultdict
from too_easy.tensor_writer import TensorStoreWriter
from torch import Tensor
import torch
import contextlib
import asyncio


class Writeable:
    def __init__(self, layer: int, tensor: Tensor, stream):
        assert isinstance(tensor, torch.Tensor)
        self.tensor = tensor
        self.stream = stream
        self.layer = layer

    def flush(self, sample_idx: int, writer: TensorStoreWriter):
        if torch.backends.mps.is_available():
            torch.mps.synchronize()
        print(f"Writing {sample_idx} layer {self.layer}")
        if isinstance(self.stream, torch.cuda.Stream):
            self.stream.synchronize()

        writer.write_layer_tensor(self.layer, sample_idx, self.tensor)


class Instrumenter:
    def __init__(self, model, fc1_pattern: str, n_layers: int):
        self.model = model
        self.fc1_pattern = fc1_pattern
        self.n_layers = n_layers
        # buffers that maps layers to tensor to write
        self.buffers = defaultdict(list)
        self._init_streams()

    def _init_streams(self):
        if torch.cuda.is_available():
            self.streams = [torch.cuda.Stream() for _ in range(self.n_layers)]
        else:
            self.streams = [contextlib.nullcontext() for _ in range(self.n_layers)]

    def instrument(self):
        left, right = self.fc1_pattern.split(".{}.")
        lefts, rights = left.split("."), right.split(".")
        module = self.model
        for latt in lefts:
            module = getattr(module, latt)

        for i in range(self.n_layers):
            i_module = module[i]
            for ratt in rights:
                i_module = getattr(i_module, ratt)
            print("Registering hook for module", i_module)

            # This copies the layer index to the closure
            def make_fwd_hook(i):
                def i_fwd_hook(_, input, output: Tensor):
                    in_d, out_d = input[0].shape[-1], output.shape[-1]
                    assert (in_d * 4 == out_d) or (
                        in_d * 8 == out_d
                    ), "FC1 layer should 4x or 8x the input."
                    with self.streams[i]:
                        self.buffers[i].append(output.to("cpu", non_blocking=True))

                return i_fwd_hook

            i_module.register_forward_hook(make_fwd_hook(i))

    def pop_writeables(self) -> list[Writeable]:
        assert (
            len(self.buffers) == self.n_layers
        ), f"len(buffers)= {len(self.buffers)} != n_layers= {self.n_layers}"
        assert (
            len(self.streams) == self.n_layers
        ), f"len(streams)= {len(self.streams)} != n_layers= {self.n_layers}"
        res = [Writeable(i, self.buffers[i][0], self.streams[i]) for i in range(self.n_layers)]
        self.buffers = defaultdict(list)
        self._init_streams()
        return res
