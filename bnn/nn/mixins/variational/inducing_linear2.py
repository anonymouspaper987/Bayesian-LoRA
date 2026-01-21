from abc import abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
import numpy as np
import random

from nflows.transforms.autoregressive import MaskedAffineAutoregressiveTransform
from .base import VariationalMixin
from .utils import EPS, prod, inverse_softplus, vec_to_chol
from ....distributions import MatrixNormal


__all__ = [
    "InducingDeterministicMixin",
    "InducingMixin",
]

def _safe_cholesky(m: torch.Tensor, add_jitter: bool = False):
    if add_jitter:
        I = torch.eye(m.shape[-1], device=m.device, dtype=m.dtype)
        m = m + (EPS * m.detach().diagonal().mean()) * I
    if m.dtype in (torch.float32, torch.float64):
        return torch.linalg.cholesky(m)
    L32 = torch.linalg.cholesky(m.float())
    return L32.to(dtype=m.dtype)

def _safe_cholesky_solve(B: torch.Tensor, chol: torch.Tensor, upper: bool = False):
    if B.dtype in (torch.float32, torch.float64):
        return torch.cholesky_solve(B, chol, upper=upper)
    X32 = torch.cholesky_solve(B.float(), chol.float(), upper=upper)
    return X32.to(dtype=B.dtype)

def _jittered_cholesky(m: torch.Tensor):
    return _safe_cholesky(m, add_jitter=True)


class SpectralParam(nn.Module):
    def __init__(self, out_dim, in_dim, *, device=None, dtype=None):
        super().__init__()
        factory_kwargs = dict(device=device, dtype=dtype)
        linear = nn.Linear(in_dim, out_dim, bias=False, **factory_kwargs)
        with torch.no_grad():
            torch.manual_seed(42)
            w = torch.randn(out_dim, in_dim, **factory_kwargs) * (out_dim ** -0.5)
            linear.weight.copy_(w)
        self.linear = torch.nn.utils.spectral_norm(linear)
        self.add_module("linear", self.linear)

    def forward(self):
        return self.linear.weight


class _InducingBase(VariationalMixin):
    def __init__(self, *args, inducing_rows=None, inducing_cols=None, prior_sd=1., init_lamda=1e-4, learn_lamda=True,
                 max_lamda=None, q_inducing="diagonal", whitened_u=True, max_sd_u=None, sqrt_width_scaling=False,
                 cache_cholesky=False, ensemble_size=8, bias=False, **kwargs):
       
        super().__init__(*args, **kwargs)
        self.has_bias = bias
        self.whitened_u = whitened_u
        self._weight_shape = self.weight.shape

        self._target_dtype = self.weight.dtype
        self._target_device = self.weight.device

        self._u = None
        self._caching_mode = False
        self.reset_cache()
        self._i = 0

        del self.weight
        if self.has_bias:
            del self.bias

        self._d_out = self._weight_shape[0]
        self._d_in = prod(self._weight_shape[1:]) + int(self.has_bias)

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

        self.total_cols = self._d_in + self.inducing_cols
        self.total_rows = self._d_out + self.inducing_rows
        self.alpha_col = 0.01
        self.alpha_row = 0.01

        factory = dict(device=self._target_device, dtype=self._target_dtype)

        self.z_row = nn.Parameter(
            torch.randn(inducing_rows, self._d_out, **factory) * (inducing_rows ** -0.5)
        )
        init_value = inverse_softplus(0.001)
        self.z_row_rho = nn.Parameter(torch.full((inducing_rows,), init_value, **factory), requires_grad=True)

        self.z_col = nn.Parameter(
            torch.randn(inducing_cols, self._d_in, **factory) * (inducing_cols ** -0.5)
        )
        self.z_col_rho = nn.Parameter(torch.full((inducing_cols,), inverse_softplus(0.001), **factory))

        d = inducing_rows * inducing_cols
        if q_inducing == "full":
            self.inducing_mean = nn.Parameter(torch.randn(d, **factory))
            self._inducing_scale_tril = nn.Parameter(torch.cat([
                torch.full((d,), inverse_softplus(1e-3), **factory),
                torch.zeros((d * (d - 1)) // 2, **factory),
            ]))
            self.register_parameter("_inducing_row_scale_tril", None)
            self.register_parameter("_inducing_col_scale_tril", None)
            self.register_parameter("_inducing_sd", None)
        elif q_inducing == "matrix":
            self.inducing_mean = nn.Parameter(torch.randn(inducing_rows, inducing_cols, **factory))
            self._inducing_row_scale_tril = nn.Parameter(torch.cat([
                torch.full((inducing_rows,), inverse_softplus(1e-3), **factory),
                torch.zeros((inducing_rows * (inducing_rows - 1)) // 2, **factory),
            ]))
            self._inducing_col_scale_tril = nn.Parameter(torch.cat([
                torch.full((inducing_cols,), inverse_softplus(1e-3), **factory),
                torch.zeros((inducing_cols * (inducing_cols - 1)) // 2, **factory),
            ]))
            self.register_parameter("_inducing_scale_tril", None)
            self.register_parameter("_inducing_sd", None)
        elif q_inducing == "diagonal":
            self.inducing_mean = nn.Parameter(torch.randn(inducing_rows, inducing_cols, **factory))
            self._inducing_sd = nn.Parameter(torch.full((inducing_rows, inducing_cols), inverse_softplus(1e-3), **factory))
            self.register_parameter("_inducing_scale_tril", None)
            self.register_parameter("_inducing_row_scale_tril", None)
            self.register_parameter("_inducing_col_scale_tril", None)
        elif q_inducing == "ensemble":
            tmp = torch.randn(inducing_rows, inducing_cols, **factory)
            self.inducing_mean = nn.Parameter(
                tmp + torch.randn(ensemble_size, inducing_rows, inducing_cols, **factory) * 0.1
            )
            self.register_parameter("_inducing_scale_tril", None)
            self.register_parameter("_inducing_row_scale_tril", None)
            self.register_parameter("_inducing_col_scale_tril", None)
            self.register_parameter("_inducing_sd", None)
        else:
            raise ValueError("q_inducing must be one of 'full', 'matrix', 'diagonal', 'ensemble'.")
        self.q_inducing = q_inducing

        self.max_lamda = max_lamda
        if learn_lamda:
            self._lamda = nn.Parameter(torch.tensor(inverse_softplus(init_lamda), **factory))
        else:
            self.register_buffer("_lamda", torch.tensor(inverse_softplus(init_lamda), **factory))

        self._d_row = nn.Parameter(torch.full((inducing_rows,), inverse_softplus(1e-3), **factory))
        self._d_col = nn.Parameter(torch.full((inducing_cols,), inverse_softplus(1e-3), **factory))

        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        np.random.seed(42)
        random.seed(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        self.weight, self.bias = self.sample_shaped_parameters()
        self._caching_mode = cache_cholesky

    def reset_cache(self):
        self._L_r_cached = None
        self._L_c_cached = None
        self._prior_inducing_row_scale_tril_cached = None
        self._prior_inducing_col_scale_tril_cached = None
        self._row_transform_cached = None
        self._u_middle = None
        self._col_transform_cached = None

    def extra_repr(self):
        s = super().extra_repr()
        s += f", inducing_cols={self.inducing_cols}, inducing_rows={self.inducing_rows}"
        return s

    def init_from_deterministic_params(self, param_dict):
        raise NotImplementedError

    def forward(self, x):
        self.weight, self.bias = self.sample_shaped_parameters()
        return super().forward(x)

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
        self._u_middle = u
        return self.sample_conditional_parameters(u, row_chol, col_chol)

    def sample_u(self):
        if self.q_inducing == "ensemble":
            i = self._i
            self._i = (self._i + 1) % len(self.inducing_mean)
            u = self.inducing_mean[i]
        else:
            u = self.inducing_dist.rsample()
        return u.view(self.inducing_rows, self.inducing_cols).to(device=self.z_row.device, dtype=self.z_row.dtype)

    def sample_conditional_parameters(self, u, row_chol, col_chol):
        mean, noise = self.conditional_mean_and_noise(u, row_chol, col_chol)
        rescaled_noise = self.lamda * noise
        return self.prior_sd * (mean + rescaled_noise)

    def kl_divergence(self):
        device = self.device
        dtype = self._target_dtype

        if self.inducing_dist is None:
            inducing_kl = torch.zeros((), device=device, dtype=dtype)
            base_kl = inducing_kl
        else:
            inducing_dist = self.inducing_dist
            inducing_prior_dist = self.inducing_prior_dist
            inducing_kl = dist.kl_divergence(inducing_dist, inducing_prior_dist)
            inducing_kl = inducing_kl.to(device=device, dtype=dtype)
            inducing_kl = inducing_kl.sum()
            base_kl = inducing_kl

        conditional_kl = self.conditional_kl_divergence().to(device=device, dtype=dtype)
        final_kl = base_kl + conditional_kl
        return final_kl

    def conditional_kl_divergence(self):
        result = self._d_in * self._d_out * (0.5 * self.lamda ** 2 - self.lamda.log() - 0.5)
        return result

    def compute_row_transform(self, row_chol):
        # Z_r^T (Z_r Z_r^T + D_r^2)^-1
        if self._row_transform_cached is None:
            row_transform = _safe_cholesky_solve(self.z_row, row_chol).t()
        else:
            row_transform = self._row_transform_cached

        if self.caching_mode:
            self._row_transform_cached = row_transform
        return row_transform

    def compute_col_transform(self, col_chol):
        # (Z_c Z_c^T + D_c^2)^-1 Z_c
        if self._col_transform_cached is None:
            col_transform = _safe_cholesky_solve(self.z_col, col_chol)
        else:
            col_transform = self._col_transform_cached

        if self.caching_mode:
            self._col_transform_cached = col_transform
        return col_transform

    @abstractmethod
    def conditional_mean_and_noise(self, u, row_chol, col_chol):
        pass

    @property
    def caching_mode(self):
        return self._caching_mode

    @caching_mode.setter
    def caching_mode(self, mode):
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
    def get_z_row_rho(self):
        return F.softplus(self.z_row_rho).view(-1, 1)

    @property
    def get_z_col_rho(self):
        return F.softplus(self.z_col_rho).view(-1, 1)

    @property
    def lamda(self):
        zero = torch.zeros((), device=self.device, dtype=self._target_dtype)
        out = F.softplus(self._lamda)
        if self.max_lamda is not None:
            out = out.clamp(min=zero, max=self.max_lamda)
        else:
            out = out.clamp(min=zero)
        return out

    @property
    def prior_inducing_row_cov(self):
        z_row = self.get_z_row()
        return z_row @ z_row.t() + self.d_row.pow(2)

    @property
    def prior_inducing_row_scale_tril(self):
        if self._prior_inducing_row_scale_tril_cached is not None:
            prior_inducing_row_scale_tril = self._prior_inducing_row_scale_tril_cached
        else:
            prior_inducing_row_scale_tril = _jittered_cholesky(self.prior_inducing_row_cov)

        if self.caching_mode:
            self._prior_inducing_row_scale_tril_cached = prior_inducing_row_scale_tril
        return prior_inducing_row_scale_tril

    def get_z_col(self):
        eps = torch.rand_like(self.z_col)
        output = self.z_col + self.alpha_col * eps * self.get_z_col_rho
        return output

    def get_z_row(self):
        eps = torch.rand_like(self.z_row)
        output = self.z_row + self.alpha_row * eps * self.get_z_row_rho
        return output

    @property
    def prior_inducing_col_cov(self):
        z_col = self.get_z_col()
        return z_col @ z_col.t() + self.d_col.pow(2)

    @property
    def prior_inducing_col_scale_tril(self):
        if self._prior_inducing_col_scale_tril_cached is not None:
            prior_inducing_col_scale_tril = self._prior_inducing_col_scale_tril_cached
        else:
            prior_inducing_col_scale_tril = _jittered_cholesky(self.prior_inducing_col_cov)

        if self.caching_mode:
            self._prior_inducing_col_scale_tril_cached = prior_inducing_col_scale_tril
        return prior_inducing_col_scale_tril

    @property
    def inducing_scale_tril(self):
        return vec_to_chol(self._inducing_scale_tril) if self._inducing_scale_tril is not None else None

    @property
    def inducing_row_scale_tril(self):
        return vec_to_chol(self._inducing_row_scale_tril) if self._inducing_row_scale_tril is not None else None

    @property
    def inducing_col_scale_tril(self):
        return vec_to_chol(self._inducing_col_scale_tril) if self._inducing_col_scale_tril is not None else None

    @property
    def inducing_sd(self):
        zero = torch.zeros((), device=self.device, dtype=self._target_dtype)
        if self._inducing_sd is None:
            return None
        out = F.softplus(self._inducing_sd)
        if self.max_sd_u is not None:
            out = out.clamp(min=zero, max=self.max_sd_u)
        else:
            out = out.clamp(min=zero)
        return out

    @property
    def inducing_prior_dist(self):
        if self.inducing_mean is None:
            return None

        loc = self.z_row.new_zeros(self.inducing_rows, self.inducing_cols)
        if self.whitened_u:
            if self.q_inducing == "full":
                cov = torch.eye(self.inducing_rows * self.inducing_cols, device=self.device, dtype=loc.dtype)
                return dist.MultivariateNormal(loc.flatten(), cov)
            elif self.q_inducing == "matrix":
                row_scale_tril = torch.eye(self.inducing_rows, device=self.device, dtype=loc.dtype)
                col_scale_tril = torch.eye(self.inducing_cols, device=self.device, dtype=loc.dtype)
                return MatrixNormal(loc, row_scale_tril, col_scale_tril)
            else:  # diagonal
                scale = self.z_row.new_ones(self.inducing_rows, self.inducing_cols)
                return dist.Normal(loc, scale)
        else:
            return MatrixNormal(loc, self.prior_inducing_row_scale_tril, self.prior_inducing_col_scale_tril)

    @property
    def inducing_dist(self):
        if self.inducing_mean is None or self.q_inducing == "ensemble":
            return None

        if self.q_inducing == "full":
            return dist.MultivariateNormal(self.inducing_mean, scale_tril=self.inducing_scale_tril)
        elif self.q_inducing == "matrix":
            return MatrixNormal(self.inducing_mean, self.inducing_row_scale_tril, self.inducing_col_scale_tril)
        else:  # diagonal
            return dist.Normal(self.inducing_mean, self.inducing_sd)

    @property
    def device(self):
        for attr in ['z_row', 'z_col', 'inducing_mean', '_lamda', '_d_row', '_d_col']:
            t = getattr(self, attr, None)
            if isinstance(t, torch.nn.Parameter) and t is not None:
                return t.device
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class InducingDeterministicMixin(_InducingBase):
    def conditional_mean_and_noise(self, u, row_chol, col_chol):
        row_transform = self.compute_row_transform(row_chol)
        col_transform = self.compute_col_transform(col_chol)
        M_w = row_transform @ u @ col_transform
        return M_w, torch.zeros_like(M_w)


class InducingMixinLinear(_InducingBase):
    def conditional_mean_and_noise(self, u, row_chol, col_chol):
        row_transform = self.compute_row_transform(row_chol)
        col_transform = self.compute_col_transform(col_chol)

        M_w = row_transform @ u @ col_transform

        e1 = self.z_row.new_empty(self._d_out, self._d_in).normal_()
        e2, e3, e4 = self.z_row.new_empty(3, self.inducing_rows, self.inducing_cols).normal_()

        w_bar = e1

        if self._L_r_cached is None:
            L_r = _jittered_cholesky(self.z_row.mm(self.z_row.t()))
        else:
            L_r = self._L_r_cached

        if self._L_c_cached is None:
            L_c = _jittered_cholesky(self.z_col.mm(self.z_col.t()))
        else:
            L_c = self._L_c_cached

        if self.caching_mode:
            self._L_r_cached = L_r
            self._L_c_cached = L_c

        t1 = self.z_row @ e1 @ self.z_col.t()
        t2 = L_r @ e2 @ self.d_col
        t3 = self.d_row @ e3 @ L_c.t()
        t4 = self.d_row @ e4 @ self.d_col
        u_bar = t1 + t2 + t3 + t4

        noise_term = w_bar - row_transform @ u_bar @ col_transform

        return M_w, noise_term

InducingMixin = InducingMixinLinear


def _iter_inducing_modules(module):
    yield from filter(lambda m: isinstance(m, _InducingBase), module.modules())


def register_global_inducing_weights_(module, inducing_rows, inducing_cols, cat_dim=0, **inducing_kwargs):
    """
    """
    class _GlobalInducingModule(InducingMixin, nn.Linear):
        def __init__(self, *args, **kwargs):
            kwargs["bias"] = False
            super().__init__(*args, **kwargs)

        def forward(self, input):
            return input  # identity

    if cat_dim not in [0, 1]:
        raise ValueError("cat_dim must be concatenated along dim 0 or 1")

    if len(list(_iter_inducing_modules(module))) < 2:
        raise ValueError("'module' at least contains two layers")

    non_cat_size = None
    cat_size = 0
    for m in _iter_inducing_modules(module):
        m_non_cat_size = m.inducing_rows if cat_dim == 1 else m.inducing_cols
        if non_cat_size is None:
            non_cat_size = m_non_cat_size
        elif m_non_cat_size != non_cat_size:
            raise ValueError("not the same dimension")

        if not m.whitened_u:
            raise ValueError("all inducing must use whitened u。")

        cat_size += m.inducing_rows if cat_dim == 0 else m.inducing_cols

        m.inducing_mean = None
        m._inducing_row_scale_tril = None
        m._inducing_col_scale_tril = None
        m._inducing_scale_tril = None
        m._inducing_sd = None

    if cat_dim == 0:
        num_cols = non_cat_size
        num_rows = cat_size
    else:
        num_cols = cat_size
        num_rows = non_cat_size

    module._global_inducing_module = _GlobalInducingModule(
        num_cols, num_rows, inducing_rows=inducing_rows, inducing_cols=inducing_cols, **inducing_kwargs
    )

    def inducing_sampling_hook(m, input):
        inducing_weights = m._global_inducing_module.sample_parameters()

        offset = 0
        for im in _iter_inducing_modules(m):
            if im is m._global_inducing_module:
                continue

            if cat_dim == 0:
                delta = im.inducing_rows
                im._u = inducing_weights[offset:offset + delta]
            else:
                delta = im.inducing_cols
                im._u = inducing_weights[:, offset:offset + delta]
            offset += delta

    return module.register_forward_pre_hook(inducing_sampling_hook)


class InducingDeterministicConditionalCholeskyMixin(_InducingBase):
    def conditional_mean_and_noise(self, u, row_chol, col_chol):
        row_transform = self.compute_row_transform(row_chol)
        col_transform = self.compute_col_transform(col_chol)

        M_w = row_transform @ u @ col_transform

        I_row = torch.eye(self._d_out, device=self.device, dtype=self._target_dtype)
        I_col = torch.eye(self._d_in, device=self.device, dtype=self._target_dtype)

        row_cov = (1 + EPS) * I_row - row_transform @ self.z_row
        L_r = _safe_cholesky(row_cov)
        col_cov = (1 + EPS) * I_col - self.z_col.t() @ col_transform
        L_c = _safe_cholesky(col_cov)

        noise_term = L_r @ torch.randn_like(M_w) @ L_c.t()

        return M_w, noise_term


class InducingMarginalizedCholeskyMixin(_InducingBase):
    def conditional_mean_and_noise(self, u, row_chol, col_chol):
        row_transform = self.compute_row_transform(row_chol)
        col_transform = self.compute_col_transform(col_chol)

        M_w = row_transform @ u @ col_transform

        row_cov = row_transform @ self.z_row
        col_cov = self.z_col.t() @ col_transform
        I_w = torch.eye(self._d_out * self._d_in, device=self.device, dtype=self._target_dtype)
        weight_cov = (1 + EPS) * I_w - torch.kron(row_cov, col_cov)
        L_w = _safe_cholesky(weight_cov)
        noise_term = L_w.mv(torch.randn_like(M_w).flatten()).view_as(M_w)

        return M_w, noise_term
