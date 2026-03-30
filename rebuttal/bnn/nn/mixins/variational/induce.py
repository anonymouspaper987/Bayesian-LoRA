from abc import abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist

from .base import VariationalMixin
from .utils import EPS, prod, inverse_softplus, vec_to_chol
from ....distributions import MatrixNormal

__all__ = [
    "InducingDeterministicMixin",
    "InducingMixinLinearDeep",
]


def _jittered_cholesky(m: torch.Tensor) -> torch.Tensor:
    j = EPS * m.detach().diagonal().mean() * torch.eye(m.shape[-1], device=m.device)
    return m.add(j).cholesky()


class _InducingBase(VariationalMixin):
    """
    Base class for inducing weight mixins.

 module sample weight/bias
 forward F.linear Module
 Tensor copy.deepcopy
    """

    def __init__(
        self,
        *args,
        inducing_rows=None,
        inducing_cols=None,
        prior_sd: float = 1.0,
        init_lamda: float = 1e-4,
        learn_lamda: bool = True,
        max_lamda=None,
        q_inducing: str = "diagonal",
        whitened_u: bool = True,
        max_sd_u=None,
        sqrt_width_scaling: bool = False,
        cache_cholesky: bool = False,
        ensemble_size: int = 8,
        bias = False,
        **kwargs,
    ):
        super().__init__(bias = bias, *args, **kwargs)
        self.in_features = kwargs.get("in_features", None)
        self.out_features = kwargs.get("out_features", None)
        self.has_bias = bias
        self.whitened_u = whitened_u
        self._weight_shape = self.weight.shape

        # _u caches the per-layer u sample -- this will be needed for the between-layer correlations
        self._u = None

        self._caching_mode = False
        self._i = 0

        del self.weight
       
        if self.bias is not None:
            del self.bias
 

        self._d_out = self._weight_shape[0]
        self._d_in = prod(self._weight_shape[1:]) + int(self.has_bias)

        self.M_w_cache = None

        if inducing_rows is None:
            inducing_rows = int(self._d_out ** 0.5)
        if inducing_cols is None:
            inducing_cols = int(self._d_in ** 0.5)

        self.inducing_rows = inducing_rows
        self.inducing_cols = inducing_cols

        if sqrt_width_scaling:
            prior_sd /= self._d_in ** 0.5

        self.prior_sd = prior_sd
        self.max_sd_u = max_sd_u

        # augmented space dims
        self.total_cols = self._d_in + self.inducing_cols
        self.total_rows = self._d_out + self.inducing_rows

        self.z_row = nn.Parameter(
            torch.randn(inducing_rows, self._d_out) * 0.1 * inducing_rows**-0.5
        )
        self.z_col = nn.Parameter(
            torch.randn(inducing_cols, self._d_in) * 0.1 * inducing_cols**-0.5
        )

        # ---------------------------
        # ---------------------------
        in_features = self._d_in - 1 if self.has_bias else self._d_in

        self.register_buffer(
            "weight_inducing",
            torch.zeros(self._d_out, in_features)
        )
        if self.has_bias:
            self.register_buffer(
                "bias_inducing",
                torch.zeros(self._d_out)
            )
        else:
            self.register_buffer("bias_inducing", None)

        self.register_buffer(
            "noise_inducing",
            torch.zeros(self._d_out, in_features)
        )

        # ---------------------------
        # ---------------------------
        d = inducing_rows * inducing_cols
        if q_inducing == "full":
            self.inducing_mean = nn.Parameter(torch.randn(d))
            self._inducing_scale_tril = nn.Parameter(
                torch.cat(
                    [
                        torch.full((d,), inverse_softplus(1e-3)),
                        torch.zeros((d * (d - 1)) // 2),
                    ]
                )
            )
            self.register_parameter("_inducing_row_scale_tril", None)
            self.register_parameter("_inducing_col_scale_tril", None)
            self.register_parameter("_inducing_sd", None)

        elif q_inducing == "matrix":
            self.inducing_mean = nn.Parameter(torch.randn(inducing_rows, inducing_cols))
            self._inducing_row_scale_tril = nn.Parameter(
                torch.cat(
                    [
                        torch.full((inducing_rows,), inverse_softplus(1e-3)),
                        torch.zeros((inducing_rows * (inducing_rows - 1)) // 2),
                    ]
                )
            )
            self._inducing_col_scale_tril = nn.Parameter(
                torch.cat(
                    [
                        torch.full((inducing_cols,), inverse_softplus(1e-3)),
                        torch.zeros((inducing_cols * (inducing_cols - 1)) // 2),
                    ]
                )
            )
            self.register_parameter("_inducing_scale_tril", None)
            self.register_parameter("_inducing_sd", None)

        elif q_inducing == "diagonal":
            self.inducing_mean = nn.Parameter(
                torch.randn(inducing_rows, inducing_cols)
            )
            self._inducing_sd = nn.Parameter(
                torch.full((inducing_rows, inducing_cols), inverse_softplus(1e-3))
            )
            self.register_parameter("_inducing_scale_tril", None)
            self.register_parameter("_inducing_row_scale_tril", None)
            self.register_parameter("_inducing_col_scale_tril", None)

        elif q_inducing == "ensemble":
            tmp = torch.randn(inducing_rows, inducing_cols)
            self.inducing_mean = nn.Parameter(
                tmp + torch.randn(ensemble_size, inducing_rows, inducing_cols) * 0.1
            )
            self.register_parameter("_inducing_scale_tril", None)
            self.register_parameter("_inducing_row_scale_tril", None)
            self.register_parameter("_inducing_col_scale_tril", None)
            self.register_parameter("_inducing_sd", None)
        else:
            raise ValueError(
                "q_inducing must be one of 'full', 'matrix', 'diagonal', 'ensemble'."
            )
        self.q_inducing = q_inducing

        self.max_lamda = max_lamda
        if learn_lamda:
            self._lamda = nn.Parameter(torch.tensor(inverse_softplus(init_lamda)))
        else:
            self.register_buffer("_lamda", torch.tensor(inverse_softplus(init_lamda)))

        self.register_buffer(
            "_d_row",
            torch.full((inducing_rows,), inverse_softplus(0.1))
        )
        self.register_buffer(
            "_d_col",
            torch.full((inducing_cols,), inverse_softplus(0.1))
        )

  

        self._caching_mode = cache_cholesky
        self.reset_cache()

    def reset_cache(self):
        self._prior_inducing_row_scale_tril_cached = None
        self._prior_inducing_col_scale_tril_cached = None
        self._row_transform_cached = None
        self._col_transform_cached = None
        self.M_w_cache = None

    def extra_repr(self):
        s = f"in_feature={self.in_features}, out_features={self.out_features}, inducing_cols={self.inducing_cols}, inducing_rows={self.inducing_rows}"
        return s

    def init_from_deterministic_params(self, param_dict):
        raise NotImplementedError

    # ==============================
    # ==============================
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight, bias = self.sample_shaped_parameters()

        if weight.dim() != 2:
            raise NotImplementedError(
 f"InducingBase.forward Linear (weight.dim()==2) weight.dim()={weight.dim()}."
            )

        return F.linear(x, weight, bias)

    def sample_shaped_parameters(self):
        parameters = self.sample_parameters()
        if self.has_bias:
            w = parameters[:, :-1]
            b = parameters[:, -1]
        else:
            w = parameters
            b = None

        return w.view(self._weight_shape), b

    def sample_parameters(self):
        if self._u is None:
            self._u = self.sample_u()
        u = self._u
        self._u = None

        row_chol = self.prior_inducing_row_scale_tril
        col_chol = self.prior_inducing_col_scale_tril

        if self.whitened_u:
            u = row_chol @ u @ col_chol.t()

        return self.sample_conditional_parameters(u, row_chol, col_chol)

    def sample_u(self):
        if self.q_inducing == "ensemble":
            i = self._i
            self._i = (self._i + 1) % len(self.inducing_mean)
            u = self.inducing_mean[i]
        else:
            u = self.inducing_dist.rsample()
        return u.view(self.inducing_rows, self.inducing_cols)

    def sample_conditional_parameters(self, u, row_chol, col_chol):
        mean, noise = self.conditional_mean_and_noise(u, row_chol, col_chol)
        rescaled_noise = self.lamda * noise
        return self.prior_sd * (mean + rescaled_noise)

    # ==============================
    # KL[q(W,U) || p(W,U)]
    # ==============================
    def kl_divergence(self):
        if self.inducing_dist is None:
            inducing_kl = 0.0
        else:
            inducing_kl = dist.kl_divergence(
                self.inducing_dist, self.inducing_prior_dist
            ).sum()
        conditional_kl = self.conditional_kl_divergence()
        return inducing_kl + conditional_kl

    def conditional_kl_divergence(self):
        """KL[q(W|U) || p(W|U)] where covariances differ only by a lambda^2 scaling."""
        return self._d_in * self._d_out * (
            0.5 * self.lamda**2 - self.lamda.log() - 0.5
        )

    # ==============================
    # ==============================
    def compute_row_transform(self, row_chol):
        # Z_r^T (Z_r Z_r^T + D_r^2)^-1
        if self._row_transform_cached is None:
            row_transform = self.z_row.cholesky_solve(row_chol).t()
            if self.caching_mode:
                self._row_transform_cached = row_transform.detach()
        else:
            row_transform = self._row_transform_cached
        return row_transform

    def compute_col_transform(self, col_chol):
        # (Z_c Z_c^T + D_c^2)^-1 Z_c
        if self._col_transform_cached is None:
            col_transform = self.z_col.cholesky_solve(col_chol)
            if self.caching_mode:
                self._col_transform_cached = col_transform.detach()
        else:
            col_transform = self._col_transform_cached
        return col_transform

    @abstractmethod
    def conditional_mean_and_noise(self, u, row_chol, col_chol):
        pass

    # ==============================
    # constrained parameters
    # ==============================
    @property
    def caching_mode(self):
        return self._caching_mode

    @caching_mode.setter
    def caching_mode(self, mode: bool):
        self._caching_mode = mode
        if mode is False:
            self.reset_cache()

    @property
    def d_row(self):
        return F.softplus(self._d_row).diag_embed()

    @property
    def d_col(self):
        return F.softplus(self._d_col).diag_embed()

    @property
    def lamda(self):
        return (
            F.softplus(self._lamda).clamp(0.0, self.max_lamda)
            if self._lamda is not None
            else None
        )

    @property
    def prior_inducing_row_cov(self):
        return self.z_row @ self.z_row.t() + self.d_row.pow(2)

    @property
    def prior_inducing_row_scale_tril(self):
        if getattr(self, "_prior_inducing_row_scale_tril_cached", None) is not None:
            prior_inducing_row_scale_tril = self._prior_inducing_row_scale_tril_cached
        else:
            prior_inducing_row_scale_tril = _jittered_cholesky(
                self.prior_inducing_row_cov
            )
            if self.caching_mode:
                self._prior_inducing_row_scale_tril_cached = (
                    prior_inducing_row_scale_tril.detach()
                )
        return prior_inducing_row_scale_tril

    @property
    def prior_inducing_col_cov(self):
        return self.z_col @ self.z_col.t() + self.d_col.pow(2)

    @property
    def prior_inducing_col_scale_tril(self):
        if getattr(self, "_prior_inducing_col_scale_tril_cached", None) is not None:
            prior_inducing_col_scale_tril = self._prior_inducing_col_scale_tril_cached
        else:
            prior_inducing_col_scale_tril = _jittered_cholesky(
                self.prior_inducing_col_cov
            )
            if self.caching_mode:
                self._prior_inducing_col_scale_tril_cached = (
                    prior_inducing_col_scale_tril.detach()
                )
        return prior_inducing_col_scale_tril

    @property
    def inducing_scale_tril(self):
        return (
            vec_to_chol(self._inducing_scale_tril)
            if getattr(self, "_inducing_scale_tril", None) is not None
            else None
        )

    @property
    def inducing_row_scale_tril(self):
        return (
            vec_to_chol(self._inducing_row_scale_tril)
            if getattr(self, "_inducing_row_scale_tril", None) is not None
            else None
        )

    @property
    def inducing_col_scale_tril(self):
        return (
            vec_to_chol(self._inducing_col_scale_tril)
            if getattr(self, "_inducing_col_scale_tril", None) is not None
            else None
        )

    @property
    def inducing_sd(self):
        return (
            F.softplus(self._inducing_sd).clamp(0.0, self.max_sd_u)
            if getattr(self, "_inducing_sd", None) is not None
            else None
        )

    @property
    def inducing_prior_dist(self):
        if self.inducing_mean is None:
            return None

        loc = self.z_row.new_zeros(self.inducing_rows, self.inducing_cols)
        if self.whitened_u:
            if self.q_inducing == "full":
                cov = torch.eye(
                    self.inducing_rows * self.inducing_cols, device=self.device
                )
                return dist.MultivariateNormal(loc.flatten(), cov)
            elif self.q_inducing == "matrix":
                row_scale_tril = torch.eye(self.inducing_rows, device=self.device)
                col_scale_tril = torch.eye(self.inducing_cols, device=self.device)
                return MatrixNormal(loc, row_scale_tril, col_scale_tril)
            else:  # diagonal
                scale = self.z_row.new_ones(self.inducing_rows, self.inducing_cols)
                return dist.Normal(loc, scale)
        else:
            return MatrixNormal(
                loc,
                self.prior_inducing_row_scale_tril,
                self.prior_inducing_col_scale_tril,
            )

    @property
    def inducing_dist(self):
        # TODO: potentially return mixture of deltas for q_inducing == "ensemble"
        if self.inducing_mean is None or self.q_inducing == "ensemble":
            return None

        if self.q_inducing == "full":
            return dist.MultivariateNormal(
                self.inducing_mean, scale_tril=self.inducing_scale_tril
            )
        elif self.q_inducing == "matrix":
            return MatrixNormal(
                self.inducing_mean,
                self.inducing_row_scale_tril,
                self.inducing_col_scale_tril,
            )
        else:  # diagonal
            return dist.Normal(self.inducing_mean, self.inducing_sd)

    @property
    def device(self):
        return self.z_row.device


class InducingDeterministicMixin(_InducingBase):
    """Inducing model where we only sample u and deterministically transform it into the weights."""

    def conditional_mean_and_noise(self, u, row_chol, col_chol):
        row_transform = self.compute_row_transform(row_chol)
        col_transform = self.compute_col_transform(col_chol)
        M_w = row_transform @ u @ col_transform
        return M_w, torch.zeros_like(M_w)


class InducingMixinLinearDeep(_InducingBase):
    """
    Inducing mixin which uses Matheron's rule to sample from the conditional
    multivariate normal on W.
    """
    def mean_weight(self):
        return self.weight_inducing

    def noise_weight(self):
        return self.noise_inducing

    def conditional_mean_and_noise(self, u, row_chol, col_chol):
        row_transform = self.compute_row_transform(row_chol)
        col_transform = self.compute_col_transform(col_chol)

        M_w = row_transform @ u @ col_transform

        e1 = self.z_row.new_empty(self._d_out, self._d_in).normal_()
        e2, e3, e4 = self.z_row.new_empty(3, self.inducing_rows, self.inducing_cols).normal_()

        w_bar = e1
        L_r = row_chol
        L_c = col_chol

        t1 = self.z_row @ e1 @ self.z_col.t()
        t2 = L_r @ e2 @ self.d_col
        t3 = self.d_row @ e3 @ L_c.t()
        t4 = self.d_row @ e4 @ self.d_col
        u_bar = t1 + t2 + t3 + t4

        noise_term = w_bar - row_transform @ u_bar @ col_transform

        # -----------------------
        #   W = prior_sd * M_w
        # -----------------------
        W_mean = (self.prior_sd * M_w).detach()   # shape: (d_out, d_in)

        if self.has_bias:
            weight_mean = W_mean[:, :-1]        # (d_out, in_features)
            bias_mean   = W_mean[:, -1]         # (d_out,)
            self.weight_inducing.copy_(weight_mean)
            self.bias_inducing.copy_(bias_mean)

            noise_w = noise_term.detach()[:, :-1]
            self.noise_inducing.copy_(noise_w)
        else:
            self.weight_inducing.copy_(W_mean.detach())
            self.noise_inducing.copy_(noise_term.detach())

        return M_w, noise_term
