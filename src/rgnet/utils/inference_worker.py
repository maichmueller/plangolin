import contextlib
import functools
import multiprocessing as mp
from typing import Any, Callable, NamedTuple, Optional, Tuple

import torch
from torch._utils import ExceptionWrapper

from rgnet.utils.data import recursively_detach, transfer_batch_to_device
from rgnet.utils.singleton import PickleSafeSingleton


class TagExceptionType(metaclass=PickleSafeSingleton):
    __slots__ = ()  # no instance dict, lighter-weight

    def __repr__(self):
        return "<TAG:EXCEPTION>"


class TagResultType(metaclass=PickleSafeSingleton):
    __slots__ = ()  # no instance dict, lighter-weight

    def __repr__(self):
        return "<TAG:RESULT>"


class TagCompletionType(metaclass=PickleSafeSingleton):
    __slots__ = ()  # no instance dict, lighter-weight

    def __repr__(self):
        return "<TAG:COMPLETION>"


class SentinelType(metaclass=PickleSafeSingleton):
    __slots__ = ()  # no instance dict, lighter-weight

    def __repr__(self):
        return "<SENTINEL>"


Sentinel = SentinelType()
TagResult = TagResultType()
TagException = TagExceptionType()


class LoadWeights(NamedTuple):
    state_dict: dict[str, Any]


class ProcessBatch(NamedTuple):
    batch: Any
    batch_idx: int
    dataloader_idx: int


class OutputBatch(NamedTuple):
    tag: TagResultType | TagExceptionType
    output: Any
    batch_idx: int
    dataloader_idx: int


def _worker_loop(
    worker_id: int,
    batch_fn: Callable[[torch.nn.Module, Any], Any],
    in_q: mp.queues.Queue,
    out_q: mp.queues.Queue,
    model: torch.nn.Module,
    device: torch.device,
) -> None:
    model.to(device).eval()
    block_put = False
    stream = torch.cuda.Stream(device=device)
    with torch.no_grad(), torch.cuda.stream(stream):
        while True:
            msg = in_q.get()
            match msg:
                case LoadWeights():
                    model.load_state_dict(msg.state_dict, strict=True)
                case ProcessBatch():
                    batch_tuple, batch_idx, dataloader_idx = (
                        msg.batch,
                        msg.batch_idx,
                        msg.dataloader_idx,
                    )
                    batch, *rest = batch_tuple
                    try:
                        output = batch_fn(model, batch)
                        output = recursively_detach(output)
                        output = transfer_batch_to_device(output, torch.device("cpu"))
                        rest = transfer_batch_to_device(rest, torch.device("cpu"))
                        out_q.put(
                            OutputBatch(
                                TagResultType(),
                                (output, *rest),
                                batch_idx,
                                dataloader_idx,
                            ),
                            block=block_put,
                        )
                    except KeyboardInterrupt:
                        raise KeyboardInterrupt
                    except Exception as e:
                        out_q.put(
                            OutputBatch(
                                TagExceptionType(),
                                ExceptionWrapper(e, where="worker loop"),
                                batch_idx,
                                dataloader_idx,
                            ),
                            block=block_put,
                        )
                case SentinelType():
                    break
                case TagCompletionType():
                    out_q.put((worker_id, TagCompletionType()), block=block_put)
                case _:
                    raise RuntimeError(f"Unknown message type: {msg!r}")


def active_check(func: Callable) -> Callable:
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        if not self._active:
            raise RuntimeError(
                "Worker not running; call start() or use as a `with`-statement context."
            )
        return func(self, *args, **kwargs)

    return wrapper


class InferenceProcessWorker(contextlib.AbstractContextManager):
    """
    A single spawned process that repeatedly takes jobs from an input queue,
    runs a user‐supplied batch_fn, and pushes results (or errors) to an output queue.
    Automatically handles startup, shutdown, and queue cleanup.

    Parameters
    ----------
    batch_fn : Callable[[Any], Any]
        A function that processes one batch and returns a result.
    ctx : Optional[mp.context.BaseContext]
        Multiprocessing context (use 'spawn' to avoid CUDA‐fork issues).
    max_qsize : int
        Max size of the in/out queues.
    join_timeout : float
        How many seconds to wait for clean shutdown before force‐killing.
    """

    def __init__(
        self,
        worker_id: int,
        model: torch.nn.Module,
        batch_fn: Callable[[torch.nn.Module, Any], Any],
        ctx: mp.context.BaseContext,
        device: torch.device,
        in_q: mp.queues.Queue | None = None,
        out_q: mp.queues.Queue | None = None,
        max_qsize: int = 0,
        join_timeout: float = 30.0,
    ):
        self.worker_id = worker_id
        self.batch_fn = batch_fn
        self.ctx: mp.context.BaseContext = ctx
        self.device = device
        self.join_timeout = join_timeout

        self.in_q: mp.queues.Queue = (
            in_q if in_q is not None else self.ctx.Queue(maxsize=max_qsize)
        )
        self.out_q: mp.queues.Queue = (
            out_q if out_q is not None else self.ctx.Queue(maxsize=max_qsize)
        )
        self.model = model
        self._proc: Optional[mp.Process] = None
        self._active = False

    def start(self) -> None:
        if self._active:
            return

        self._proc = self.ctx.Process(
            target=_worker_loop,
            args=(
                self.worker_id,
                self.batch_fn,
                self.in_q,
                self.out_q,
                self.model,
                self.device,
            ),
            daemon=True,
        )
        self._proc.start()
        self._active = True

    def __enter__(self) -> "InferenceProcessWorker":
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        # 1) Prevent new submissions
        self._active = False
        # 2) Signal shutdown
        self.in_q.put(SentinelType())
        # 3) Drain remaining outputs if you care
        #    (optional—here we just join then close)
        self._proc.join(timeout=self.join_timeout)
        if self._proc.is_alive():
            self._proc.terminate()
            self._proc.join()

        # 4) Clean up queues
        for q in (self.in_q, self.out_q):
            q.cancel_join_thread()
            q.close()

        return False  # don’t swallow errors

    @active_check
    def submit(
        self, batch: Any, block: bool = True, timeout: Optional[float] = None
    ) -> None:
        self.in_q.put(batch, block=block, timeout=timeout)

    @active_check
    def receive(
        self, block: bool = True, timeout: Optional[float] = None
    ) -> Tuple[str, Any]:
        tag, payload = self.out_q.get(block=block, timeout=timeout)
        return tag, payload
