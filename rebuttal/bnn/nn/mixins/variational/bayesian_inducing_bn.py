# -*- coding: utf-8 -*-
"""
Bayesian Inducing BatchNorm (2D) — Fast Variant
------------------------------------------------
Major speedups vs the reference version:
  • Optional stochastic Z refresh (every N steps) + caching of transforms
  • Cheaper Matheron noise (configurable: 'none' | 'cheap' | 'full')
  • Diagonal scales applied via broadcasting (avoid diag_embed matmuls)
  • Vectorized MC evaluation helper (forward_mcmc)

API is drop-in compatible with the previous class in most places.

Usage example:
    from bayesian_inducing_bn_fast import BayesianInducingBN2dFast
    bn = BayesianInducingBN2dFast(C, noise_mode='cheap', stochastic_z=False, cache_cholesky=True)

Notes:
  - Set noise_mode='none' at inference for maximum speed.
  - Set stochastic_z=False (default) for stable caches; True enables extra randomness
    in Z_row/Z_col but will recompute transforms every `refresh_every` steps only.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist

EPS = 1e-6

# -------------------------- utils --------------------------
def _jittered_cholesky(K: torch.Tensor, jitter_mult: float = 1e-6) -> torch.Tensor:
    n = K.shape[-1]
    diag_mean = K.diagonal(dim1=-2, dim2=-1).detach().abs().mean()
    jitter = jitter_mult * (diag_mean + EPS)
    I = torch.eye(n, device=K.device, dtype=K.dtype)
    for _ in range(3):
        try:
            return torch.linalg.cholesky(K + jitter * I)
        except RuntimeError:
            jitter *= 10.0
    return torch.linalg.cholesky(K + jitter * I)

# ----------------------- fast BN module -----------------------
class BayesianInducingBN2dFast(nn.Module):
    """
    BatchNorm2d with Bayesian inducing parameters for (gamma, beta),
    with performance-oriented caches and simplified noise options.

    Key knobs:
      - noise_mode: 'none' | 'cheap' | 'full'
      - stochastic_z: whether to add noise to Z_row/Z_col
      - refresh_every: refresh Z and transforms every N forward() calls in training
    """
    def __init__(
        self,
        num_features: int,
        inducing_rows: Optional[int] = None,
        inducing_cols: Optional[int] = None,
        prior_sd: float = 1.0,
        init_lambda: float = 1e-4,
        learn_lambda: bool = True,
        max_lambda: Optional[float] = None,
        heavy_tail: bool = True,
        a0: float = 5.0,
        b0: float = 5.0,
        shared_ab: bool = False,
        momentum: float = 0.1,
        eps: float = 1e-5,
        cache_cholesky: bool = True,
        base_gamma: float = 1.0,
        base_beta: float = 0.0,
        # NEW knobs for speed
        noise_mode: str = 'cheap',   # 'none' | 'cheap' | 'full'
        stochastic_z: bool = True,
        refresh_every: int = 16,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        assert noise_mode in ('none', 'cheap', 'full')
        self.C = int(num_features)
        self.prior_sd = float(prior_sd)
        self.heavy_tail = bool(heavy_tail)
        self.a0 = float(a0)
        self.b0 = float(b0)
        self.shared_ab = bool(shared_ab)
        self.momentum = float(momentum)
        self.eps = float(eps)
        self._cache_cholesky = bool(cache_cholesky)
        self.noise_mode = noise_mode
        self.stochastic_z = bool(stochastic_z)
        self.refresh_every = max(1, int(refresh_every))

        # Inducing dims
        r_default = max(1, int(math.sqrt(self.C)))
        c_default = 2
        self.r = int(inducing_rows or r_default)
        self.c = int(inducing_cols or c_default)
        if self.c < 2:
            self.c = 2

        # Baselines
        self.register_buffer("base_gamma", torch.full((self.C,), float(base_gamma), device=device, dtype=dtype))
        self.register_buffer("base_beta", torch.full((self.C,), float(base_beta), device=device, dtype=dtype))

        # Inducing mean (r x c)
        self.inducing_mean = nn.Parameter(torch.zeros(self.r, self.c, device=device, dtype=dtype))

        # Hierarchical or Gaussian posterior
        if self.heavy_tail:
            if self.shared_ab:
                self._a = nn.Parameter(torch.full((1, 1), 5.0, device=device, dtype=dtype))
                self._b = nn.Parameter(torch.full((1, 1), 5.0, device=device, dtype=dtype))
            else:
                self._a = nn.Parameter(torch.full((self.r, self.c), 5.0, device=device, dtype=dtype))
                self._b = nn.Parameter(torch.full((self.r, self.c), 5.0, device=device, dtype=dtype))
            self._inducing_sd = None
        else:
            self._inducing_sd = nn.Parameter(torch.full((self.r, self.c), 1e-3, device=device, dtype=dtype))
            self._a = None
            self._b = None

        # Lambda for Matheron noise
        lam_init = math.log(math.expm1(init_lambda))
        lam_param = torch.tensor(lam_init, device=device, dtype=dtype)
        self._lam = nn.Parameter(lam_param) if learn_lambda else lam_param
        self.max_lambda = max_lambda if max_lambda is None else float(max_lambda)

        # Low-rank factors Z and scalings (trainable)
        self.alpha_row = 0.01
        self.alpha_col = 0.01
        self.z_row = nn.Parameter(torch.randn(self.r, self.C, device=device, dtype=dtype) * (self.r ** -0.5))
        self.z_col = nn.Parameter(torch.randn(self.c, 2, device=device, dtype=dtype) * (self.c ** -0.5))
        self.z_row_rho = nn.Parameter(torch.full((self.r,), 1e-3, device=device, dtype=dtype))
        self.z_col_rho = nn.Parameter(torch.full((self.c,), 1e-3, device=device, dtype=dtype))
        self._d_row = nn.Parameter(torch.full((self.r,), 1e-3, device=device, dtype=dtype))
        self._d_col = nn.Parameter(torch.full((self.c,), 1e-3, device=device, dtype=dtype))

        # BN running stats
        self.register_buffer("running_mean", torch.zeros(self.C, device=device, dtype=dtype))
        self.register_buffer("running_var", torch.ones(self.C, device=device, dtype=dtype))
        self.register_buffer("num_batches_tracked", torch.tensor(0, dtype=torch.long, device=device))

        # Caches
        self._L_r = None
        self._L_c = None
        self._R = None
        self._C = None
        self._Zr_cached = None
        self._Zc_cached = None
        self.register_buffer("_step", torch.tensor(0, dtype=torch.long))

    # ------------------- helpers -------------------
    def a_q(self) -> torch.Tensor:
        if self._a is None:
            raise RuntimeError("a_q requested but heavy_tail=False")
        return F.softplus(self._a) + EPS

    def b_q(self) -> torch.Tensor:
        if self._b is None:
            raise RuntimeError("b_q requested but heavy_tail=False")
        return F.softplus(self._b) + EPS

    @property
    def lam(self) -> torch.Tensor:
        if isinstance(self._lam, torch.Tensor) and self._lam.requires_grad:
            lam = F.softplus(self._lam)
        else:
            lam = F.softplus(torch.as_tensor(self._lam))
        if self.max_lambda is not None:
            lam = lam.clamp(max=self.max_lambda)
        return lam + 0.0  # keep tensor type

    def _dr_vec(self) -> torch.Tensor:
        return F.softplus(self._d_row) + EPS  # (r,)

    def _dc_vec(self) -> torch.Tensor:
        return F.softplus(self._d_col) + EPS  # (c,)

    def _noisy(self, Z: torch.Tensor, rho: torch.Tensor, alpha: float) -> torch.Tensor:
        eps = torch.randn_like(Z)
        return Z + alpha * eps * (F.softplus(rho) + EPS).view(-1, 1)
    def parameter_loss(self):
        return self.kl_divergence()

    def _maybe_refresh_Z_and_transforms(self):
        """Refresh cached Zr/Zc and transforms every `refresh_every` steps (training),
        or when caches are empty. Keeps them stable otherwise for speed.
        """
        need_refresh = (self._Zr_cached is None) or (self._Zc_cached is None)
        if self.training and self.stochastic_z:
            # refresh on schedule
            if (int(self._step.item()) % self.refresh_every) == 0:
                need_refresh = True
        if need_refresh:
            if self.stochastic_z:
                Zr = self._noisy(self.z_row, self.z_row_rho, self.alpha_row)
                Zc = self._noisy(self.z_col, self.z_col_rho, self.alpha_col)
            else:
                Zr = self.z_row
                Zc = self.z_col
            self._Zr_cached = Zr
            self._Zc_cached = Zc
            # Invalidate transforms so they recompute with the new Z
            self._R = None; self._C = None; self._L_r = None; self._L_c = None

        # Compute transforms if missing
        if self._L_r is None or self._L_c is None:
            Kr = self._Zr_cached @ self._Zr_cached.t() + torch.diag(self._dr_vec() ** 2)
            Kc = self._Zc_cached @ self._Zc_cached.t() + torch.diag(self._dc_vec() ** 2)
            self._L_r = _jittered_cholesky(Kr)
            self._L_c = _jittered_cholesky(Kc)
        if self._R is None:
            X = torch.cholesky_solve(self._Zr_cached, self._L_r)  # (r x C)
            self._R = X.t()  # (C x r)
        if self._C is None:
            self._C = torch.cholesky_solve(self._Zc_cached, self._L_c)  # (c x 2)

    def reset_cache(self):
        self._L_r = None; self._L_c = None; self._R = None; self._C = None
        self._Zr_cached = None; self._Zc_cached = None

    # ------------------- sampling -------------------
    def _sample_u(self) -> torch.Tensor:
        if self.heavy_tail:
            aq, bq = self.a_q(), self.b_q()
            tau = dist.Gamma(aq, bq).rsample()
            sigma = (1.0 / (tau + EPS)).sqrt()
            eps = torch.randn_like(self.inducing_mean)
            u = self.inducing_mean + sigma * eps
        else:
            sd = F.softplus(self._inducing_sd) + EPS
            u = dist.Normal(self.inducing_mean, sd).rsample()
        return u  # (r x c)

    # ------------------- mean + noise -------------------
    def _conditional_mean_and_noise(self, u: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        self._maybe_refresh_Z_and_transforms()
        R, Cmat, Lr, Lc = self._R, self._C, self._L_r, self._L_c
        M_w = R @ u @ Cmat  # (C x 2)

        if self.noise_mode == 'none':
            noise = torch.zeros_like(M_w)
            return M_w, noise

        # cheap noise: just e1
        e1 = torch.randn(self.C, 2, device=u.device, dtype=u.dtype)
        if self.noise_mode == 'cheap':
            return M_w, e1

        # full Matheron noise with fast diag broadcasting
        e2 = torch.randn(self.r, self.c, device=u.device, dtype=u.dtype)
        e3 = torch.randn(self.r, self.c, device=u.device, dtype=u.dtype)
        e4 = torch.randn(self.r, self.c, device=u.device, dtype=u.dtype)
        Zr = self._Zr_cached; Zc = self._Zc_cached
        dr = self._dr_vec(); dc = self._dc_vec()

        t1 = Zr @ e1 @ Zc.t()                    # (r x c)
        t2 = Lr @ e2                             # (r x c)
        t2 = t2 * dc.view(1, -1)                 # right diag
        t3 = e3 @ Lc.t()                         # (r x c)
        t3 = dr.view(-1, 1) * t3                 # left diag
        t4 = dr.view(-1, 1) * e4 * dc.view(1, -1)
        U_bar = t1 + t2 + t3 + t4
        noise = e1 - R @ U_bar @ Cmat
        return M_w, noise

    def _sample_gamma_beta(self) -> Tuple[torch.Tensor, torch.Tensor]:
        u = self._sample_u()
        M_w, noise = self._conditional_mean_and_noise(u)
        params = self.prior_sd * (M_w + self.lam * noise)
        gamma = self.base_gamma + params[:, 0]
        beta = self.base_beta + params[:, 1]
        return gamma, beta

    # ------------------- KL terms -------------------
    def _kl_hierarchical(self) -> torch.Tensor:
        if not self.heavy_tail:
            return torch.zeros((), device=self.inducing_mean.device, dtype=self.inducing_mean.dtype)
        aq = self.a_q(); bq = self.b_q()
        a0 = torch.as_tensor(self.a0, device=aq.device, dtype=aq.dtype)
        b0 = torch.as_tensor(self.b0, device=aq.device, dtype=aq.dtype)
        if self.shared_ab:
            aq = aq.expand_as(self.inducing_mean)
            bq = bq.expand_as(self.inducing_mean)
            a0 = a0.expand_as(self.inducing_mean)
            b0 = b0.expand_as(self.inducing_mean)
        kl_var = (
            a0 * torch.log(bq / (b0 + EPS) + EPS)
            - (torch.lgamma(aq) - torch.lgamma(a0))
            + (aq - a0) * torch.digamma(aq)
            + aq * (b0 / (bq + EPS) - 1.0)
        )
        e_invvar = aq / (bq + EPS)
        kl_mean = 0.5 * (self.inducing_mean ** 2) * e_invvar
        return (kl_var + kl_mean).sum()

    def _kl_gaussian(self) -> torch.Tensor:
        if self.heavy_tail:
            return torch.zeros((), device=self.inducing_mean.device, dtype=self.inducing_mean.dtype)
        sd = F.softplus(self._inducing_sd) + EPS
        kl = -torch.log(sd) + 0.5 * (sd ** 2 + self.inducing_mean ** 2) - 0.5
        return kl.sum()

    def conditional_kl_divergence(self) -> torch.Tensor:
        lam = self.lam.clamp_min(1e-8)
        term = 0.5 * lam ** 2 - torch.log(lam + EPS) - 0.5
        count = self.C * 2
        return count * term

    def kl_divergence(self) -> torch.Tensor:
        return self._kl_hierarchical() + self._kl_gaussian() + self.conditional_kl_divergence()

    # ------------------- BN forward -------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 4 and x.size(1) == self.C, "Expect NCHW input"
        if self.training:
            mean = x.mean(dim=(0, 2, 3))
            var = x.var(dim=(0, 2, 3), unbiased=False)
            with torch.no_grad():
                self.running_mean.lerp_(mean, 1.0 - self.momentum)
                self.running_var.lerp_(var, 1.0 - self.momentum)
                self.num_batches_tracked += 1
                self._step += 1
        else:
            mean = self.running_mean
            var = self.running_var
        x_hat = (x - mean[None, :, None, None]) / (var[None, :, None, None] + self.eps).sqrt()
        gamma, beta = self._sample_gamma_beta()
        return gamma[None, :, None, None] * x_hat + beta[None, :, None, None]

    # ------------------- extras -------------------
    @torch.no_grad()
    def expected_gamma_beta(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Cheap deterministic approximation (noise_mode='none', E[u]=m)."""
        self._maybe_refresh_Z_and_transforms()
        R, Cmat = self._R, self._C
        M_w = R @ self.inducing_mean @ Cmat
        params = self.prior_sd * M_w
        gamma = self.base_gamma + params[:, 0]
        beta = self.base_beta + params[:, 1]
        return gamma, beta

    @torch.no_grad()
    def forward_mcmc(self, x: torch.Tensor, K: int = 8) -> torch.Tensor:
        """Vectorized MC evaluation: returns (K,N,C,H,W)."""
        assert x.dim() == 4 and x.size(1) == self.C
        mean = self.running_mean
        var = self.running_var
        x_hat = (x - mean[None, :, None, None]) / (var[None, :, None, None] + self.eps).sqrt()
        # sample K sets of (gamma,beta)
        gammas = []
        betas = []
        for _ in range(K):
            g, b = self._sample_gamma_beta()
            gammas.append(g)
            betas.append(b)
        gamma = torch.stack(gammas, 0)[:, None, :, None, None]  # (K,1,C,1,1)
        beta = torch.stack(betas, 0)[:, None, :, None, None]
        y = gamma * x_hat[None, ...] + beta
        return y

# ----------------------- sanity test -----------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    C = 64
    m = BayesianInducingBN2dFast(C, inducing_rows=8, inducing_cols=2,
                                 heavy_tail=True, noise_mode='cheap',
                                 stochastic_z=False, cache_cholesky=True)
    m.train()
    x = torch.randn(32, C, 8, 8)
    y = m(x)
    print("train forward ok:", y.shape)
    kl = m.kl_divergence()
    print("KL:", float(kl))
    m.eval()
    with torch.no_grad():
        y1 = m(x)
        yk = m.forward_mcmc(x, K=8)
    print("eval single:", y1.shape, "MC:", yk.shape)
