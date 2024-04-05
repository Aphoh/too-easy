from too_easy.tensor_writer import TensorStoreWriter
from torch import Tensor
import torch
import contextlib
from asyncio import AbstractEventLoop
from concurrent.futures import ThreadPoolExecutor
import asyncio


class Writeable:
    def __init__(self, sample_idx: int, layer: int, tensor: Tensor, stream):
        assert isinstance(tensor, torch.Tensor)
        self.tensor = tensor
        self.stream = stream
        self.layer = layer
        self.sample_idx = sample_idx

    def flush(self, writer: TensorStoreWriter):
        if torch.backends.mps.is_available():
            torch.mps.synchronize()
        print(f"Writing {self.sample_idx} layer {self.layer}")
        if isinstance(self.stream, torch.cuda.Stream):
            self.stream.synchronize()

        writer.write_layer_tensor(self.layer, self.sample_idx, self.tensor)


class Instrumenter:
    def __init__(
        self,
        model,
        loop: AbstractEventLoop,
        pool: ThreadPoolExecutor,
        writer: TensorStoreWriter,
        fc1_pattern: str,
        n_layers: int,
    ):
        self.model = model
        self.fc1_pattern = fc1_pattern
        self.n_layers = n_layers
        self.loop = loop
        self.pool = pool
        self.writer = writer
        self.write_handles = []
        self.sample_idx = 0
        self._init_streams()

    def _init_streams(self):
        if torch.cuda.is_available():
            self.streams = [torch.cuda.Stream() for _ in range(self.n_layers)]
        else:
            self.streams = [None for _ in range(self.n_layers)]

    def step(self, n_samples: int):
        self.sample_idx += n_samples
        self._init_streams()

    def _make_fwd_hook(self, layer: int):
        def i_fwd_hook(_, input, output: Tensor):
            in_d, out_d = input[0].shape[-1], output.shape[-1]
            assert (in_d * 4 == out_d) or (
                in_d * 8 == out_d
            ), "FC1 layer should 4x or 8x the input."
            ctx = contextlib.nullcontext()
            stream = self.streams[layer]
            if self.streams[layer]:
                ctx = torch.cuda.stream(stream)
            with ctx:
                w = Writeable(self.sample_idx, layer, output.to("cpu", non_blocking=True), stream)
                self.write_handles.append(
                    self.loop.run_in_executor(self.pool, w.flush, self.writer)
                )

        return i_fwd_hook

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

            i_module.register_forward_hook(self._make_fwd_hook(i))

    async def flush(self):
        await asyncio.gather(*self.write_handles)
