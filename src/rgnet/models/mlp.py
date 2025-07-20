import inspect
import re
from typing import Any, Callable, Iterable, Union

import torch
from torch_geometric.nn.resolver import activation_resolver
from torchrl.data import DEVICE_TYPING
from torchrl.modules import MLP

from .residual import ResidualModule


class PaddingMixin:
    """
    Mixin that ensures any incoming tensor x has at least
    `expected_input_dim` features in its last dimension,
    padding with leading zeros if necessary.
    """

    def __init__(self, expected_input_dim: int, loc: str, *args, **kwargs):
        """
        Store expected input dim and let other base classes init.
        """
        self.expected_input_dim = expected_input_dim
        assert loc in (
            "pre",
            "post",
        ), f"PaddingMixin location must be 'pre' or 'post', got {loc=}."
        self.pre = loc == "pre"
        super().__init__(*args, **kwargs)

    def _maybe_pad(self, x: torch.Tensor) -> torch.Tensor:
        """
        If x.shape[-1] < expected_input_dim, pad with zeros in front.
        If it's > expected_input_dim, raises ValueError.
        """
        cur_dim = x.shape[-1]
        if cur_dim < self.expected_input_dim:
            pad = torch.zeros(
                *x.shape[:-1],
                self.expected_input_dim - cur_dim,
                dtype=x.dtype,
                device=x.device,
            )
            return torch.cat([pad, x] if self.pre else [x, pad], dim=-1)
        if cur_dim > self.expected_input_dim:
            raise ValueError(
                f"Tensor last dim is {cur_dim}, but expected ≤ {self.expected_input_dim}"
            )
        return x

    def forward(self, x: torch.Tensor, *args, **kwargs):
        x = self._maybe_pad(x)
        return super().forward(x, *args, **kwargs)


class ArityMLPFactory:
    class PaddingMLP(PaddingMixin, MLP): ...

    class PaddingResidualModule(PaddingMixin, ResidualModule): ...

    def __init__(
        self,
        feature_size: int,
        in_extra_features: int | None = None,
        out_extra_features: int | None = None,
        added_arity: int = 0,
        residual: bool = True,
        padding: str | None = None,
        layers: int | Iterable[int] | Iterable[str] = 3,
        activation: str = "mish",
        norm_class: type[torch.nn.Module] | Callable | None = None,
        norm_kwargs: dict | list[dict] | None = None,
        dropout: float | None = None,
        bias_last_layer: bool = True,
        single_bias_last_layer: bool = False,
        layer_class: type[torch.nn.Module] | Callable = torch.nn.Linear,
        layer_kwargs: dict | None = None,
        activate_last_layer: bool = False,
        device: DEVICE_TYPING | None = None,
        **kwargs,
    ):
        """
        Factory for creating ResidualModule instances with a PaddingMLP.
        :param feature_size: The feature size to use for the MLP.
        """
        self.feature_size = feature_size
        self.in_extra_features = in_extra_features or 0
        self.out_extra_features = out_extra_features or 0
        self.added_arity = added_arity
        self.residual = residual
        self.padding = padding
        self.layers = [layers] if isinstance(layers, int) else list(layers)
        self.activation = activation
        self.norm_class = norm_class
        self.norm_kwargs = norm_kwargs or dict()
        self.kwargs = kwargs | {
            "dropout": dropout,
            "bias_last_layer": bias_last_layer,
            "single_bias_last_layer": single_bias_last_layer,
            "layer_class": layer_class,
            "layer_kwargs": layer_kwargs,
            "activate_last_layer": activate_last_layer,
            "device": device,
        }

    def arity_feature_size(self, arity: int) -> int:
        return self.feature_size * (arity + self.added_arity)

    def __call__(self, predicate: str, arity: int) -> torch.nn.Module:
        """
        Factory method to create a ResidualModule with a PaddingMLP.
        """
        # arity + 1 to allow for entering the aggregated state information
        arity_feature_size = self.arity_feature_size(arity)
        if self.norm_class is not None:
            if self.norm_class == torch.nn.LayerNorm:
                self.norm_kwargs["normalized_shape"] = arity_feature_size
        mlp_kwargs = inspect.getfullargspec(MLP.__init__)
        # slice kwargs to only include MLP-specific arguments
        mlp_kwargs = {k: v for k, v in self.kwargs.items() if k in mlp_kwargs.args}
        outer_kwargs: dict[str, Any] = {}
        if not self.residual:
            return self._make_mlp(
                MLP if self.padding is None else ArityMLPFactory.PaddingMLP, arity
            )
        else:
            if self.padding is not None:
                outer = ArityMLPFactory.PaddingResidualModule
                outer_kwargs = {
                    "expected_input_dim": arity_feature_size,
                    "loc": self.padding,
                }
            else:
                outer = ResidualModule
            return outer(module=self._make_mlp(MLP, arity), **outer_kwargs)

    def _make_mlp(self, class_: type[MLP], arity: int) -> torch.nn.Module:
        arity_feature_size = self.arity_feature_size(arity)
        layers = []
        for layer in self.layers:
            layers.append(
                self._make_layer_size(
                    layer_mode=layer,
                    arity_feature_size=arity_feature_size,
                    prev_size=arity_feature_size if not layers else layers[-1],
                )
            )
        return class_(
            in_features=arity_feature_size + self.in_extra_features,
            num_cells=layers,
            out_features=arity_feature_size + self.out_extra_features,
            activation_class=type(activation_resolver(self.activation)),
            **self.kwargs,
        )

    @staticmethod
    def _make_layer_size(
        layer_mode: Union[str, int],
        arity_feature_size: int,
        prev_size: int | None = None,
    ) -> int:
        """
        Compute a layer size.

        layer_mode:
          - int:
              - -1 → arity_feature_size
              - >0 → that exact size
          - str:
            - "=" or "=="               → prev_size
            - "+N", "-N", "xN"/"*N", "/N" → arithmetic on prev_size
            - "//N" (integer div) and "**N" (power) also supported
        """
        # integer literal
        if isinstance(layer_mode, int):
            if layer_mode == -1:
                return arity_feature_size
            if layer_mode <= 0:
                raise ValueError(f"Layer size must be >0 (or -1), got {layer_mode}")
            return layer_mode

        # must be a string now
        assert isinstance(
            layer_mode, str
        ), f"Invalid layer_mode type: {type(layer_mode)}"

        s = layer_mode.strip()

        # same‐as‐prev
        if s in ("=", "=="):
            if prev_size is None:
                raise ValueError(f"prev_size required for mode {s!r}")
            return prev_size

        # arithmetic ops
        # first handle two‐char ops: "**" and "//"
        if s.startswith("**"):
            if prev_size is None:
                raise ValueError(f"prev_size required for mode {s!r}")
            power = float(s[2:])
            return int(prev_size**power)
        if s.startswith("//"):
            if prev_size is None:
                raise ValueError(f"prev_size required for mode {s!r}")
            divisor = float(s[2:])
            if divisor == 0:
                raise ValueError("Division by zero")
            return prev_size // int(divisor)

        # then single‐char ops
        m = re.fullmatch(r"(?P<op>[+\-x*/])(?P<num>\d+(\.\d+)?)", s)
        if m:
            if prev_size is None:
                raise ValueError(f"prev_size required for mode {s!r}")
            op = m.group("op")
            num = float(m.group("num"))
            if op in ("x", "*"):
                out = int(prev_size * num)
            elif op == "/":
                if num == 0:
                    raise ValueError("Division by zero")
                out = int(prev_size / num)
            elif op == "+":
                out = int(prev_size + num)
            elif op == "-":
                out = int(prev_size - num)
            else:
                # should never happen
                raise AssertionError
            if out <= 0:
                raise ValueError(f"Computed layer size {out} ≤ 0 for mode {s!r}")
            return out

        # nothing matched
        raise ValueError(f"Invalid layer_mode: {layer_mode!r}")
