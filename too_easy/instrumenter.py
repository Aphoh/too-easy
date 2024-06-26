import asyncio
import contextlib
from asyncio import AbstractEventLoop, Future
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, List

import torch
import torch.distributed as dist
from torch import Tensor

from .tensor_writer import TensorStoreWriter


class Writeable:
    def __init__(self, layer: int, tensor: Tensor, stream):
        assert isinstance(tensor, torch.Tensor)
        self.tensor = tensor
        self.stream = stream
        self.layer = layer

    def flush(self, writer: TensorStoreWriter):
        if torch.backends.mps.is_available():
            torch.mps.synchronize()
        if isinstance(self.stream, torch.cuda.Stream):
            self.stream.synchronize()

        converted = writer.convert_tensor(self.tensor)
        writer.ts_arr[self.layer] += converted
        del self.tensor
        del self.stream


class Instrumenter:
    def __init__(
        self,
        model,
        layer_transform: Callable,
        loop: AbstractEventLoop,
        pool: ThreadPoolExecutor,
        writer: TensorStoreWriter,
        fc1_pattern: str,
        n_layers: int,
    ):
        self.model = model
        self.fc1_pattern = fc1_pattern
        self.n_layers = n_layers
        self.layer_transform = layer_transform
        self.loop = loop
        self.pool = pool
        self.writer = writer
        self.write_handles: List[Future[Any]] = []
        self._init_streams()

    def _init_streams(self):
        if torch.cuda.is_available():
            self.streams = [torch.cuda.Stream() for _ in range(self.n_layers)]
        else:
            self.streams = [None for _ in range(self.n_layers)]

    def step(self, n_samples: int):
        self._init_streams()

    def _make_fwd_hook(self, layer: int):
        def i_fwd_hook(_, input, output: Tensor):
            ctx = contextlib.nullcontext()
            stream = self.streams[layer]
            if self.streams[layer]:
                ctx = torch.cuda.stream(stream)
            with ctx:
                output = self.layer_transform(output)
                if dist.is_initialized():
                    dist.all_reduce(output)
                if not dist.is_initialized() or dist.get_rank() == 0:
                    w = Writeable(layer, output.to("cpu", non_blocking=True), stream)
                    self.write_handles.append(self.loop.run_in_executor(self.pool, w.flush, self.writer))

        return i_fwd_hook

    def instrument(self):
        left, right = self.fc1_pattern.split(".{}.")
        lefts, rights = left.split("."), right.split(".")
        module = self.model
        for latt in lefts:
            module = getattr(module, latt)

        for i in range(self.n_layers):
            i_module: torch.nn.Module = module[i]
            for ratt in rights:
                i_module = getattr(i_module, ratt)

            i_module.register_forward_hook(self._make_fwd_hook(i))

    async def flush(self):
        await asyncio.gather(*self.write_handles)
