import numpy as np
import torch
from torch import nn, Tensor
import torch.nn.functional as F


class DSSSoftmaxKernel(nn.Module):
    def __init__(self, n: int, h: int):
        # h copies of softmax kernel

        super().__init__()

        self.n = n
        self.h = h

        self.lambda_re = nn.Parameter(torch.zeros(h, n))
        self.lambda_im = nn.Parameter(torch.zeros(h, n))
        self.w = nn.Parameter(torch.zeros([h, n, 2]))
        self.log_delta = nn.Parameter(torch.zeros(h))

        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            self.w.normal_()

            self.log_delta.uniform_(np.log(0.001), np.log(0.1))
            self.log_delta.exp_()

            lambda_re, lambda_im = self._init_hippo(self.n)
            lambda_re = lambda_re.repeat(self.h, 1)
            lambda_im = lambda_im.repeat(self.h, 1)
            self.lambda_re.copy_(lambda_re)
            self.lambda_im.copy_(lambda_im)
            assert self.lambda_re.shape == (self.h, self.n)
            assert self.lambda_im.shape == (self.h, self.n)

    def _init_hippo(self, n: int) -> tuple[Tensor, Tensor]:
        hippo_matrix = torch.zeros([2 * n, 2 * n])
        for i in range(2 * n):
            for j in range(2 * n):
                if i < j:
                    hippo_matrix[i, j] = (2 * i + 1) ** 0.5 * (2 * j + 1) ** 0.5 / 2
                elif i > j:
                    hippo_matrix[i, j] = -((2 * i + 1) ** 0.5) * (2 * j + 1) ** 0.5 / 2
                else:
                    hippo_matrix[i, j] = -0.5

        L, _ = torch.linalg.eig(hippo_matrix)
        L_positive = torch.cat(
            [lmda.unsqueeze(0) for lmda in L if torch.imag(lmda) > 0]
        )
        assert len(L_positive) == n

        return torch.real(L_positive), torch.imag(L_positive)

    def _row_softmax_eps(self, x: Tensor, eps: float = 1e-7) -> Tensor:
        x_argmax = torch.argmax(x.real, dim=-1, keepdim=True)
        x_max = torch.gather(x, -1, x_argmax).detach()
        x = x - x_max

        x_sum = torch.sum(x.exp(), dim=-1)
        recipr = x_sum.conj() / (x_sum * x_sum.conj() + eps)

        return x.exp() * recipr.unsqueeze(-1)

    def _convolve(self, K: Tensor, u: Tensor) -> Tensor:
        L = K.shape[-1]
        K_ft = torch.fft.rfft(K, n=2 * L)
        u_ft = torch.fft.rfft(u, n=2 * L)
        res = torch.fft.irfft(K_ft * u_ft, n=2 * L)[..., :L]
        return res

    def forward(self, u: Tensor) -> Tensor:
        # TODO: figure out batching
        # u: (H x L)

        u = u.permute(1, 0)

        assert len(u.shape) == 2
        assert u.shape[0] == self.h
        L = u.shape[1]

        lmbda = torch.complex(self.lambda_re, self.lambda_im)
        delta = torch.exp(self.log_delta).unsqueeze(-1)

        P_left = (delta * lmbda).view(self.h, self.n, 1)
        P_right = torch.arange(0, L, 1).expand(self.h, 1, L).to(torch.complex64)
        P = P_left @ P_right
        assert P.shape == (self.h, self.n, L)

        S = self._row_softmax_eps(P)

        K_bar = torch.real(
            (torch.view_as_complex(self.w) / lmbda).view(self.h, 1, self.n) @ S
        )
        assert K_bar.shape == (self.h, 1, L)

        res = self._convolve(K_bar.squeeze(), u)
        return res.permute(1, 0)


class DSSLayer(nn.Module):
    def __init__(self, n: int, h: int):
        super().__init__()

        self.kernel = DSSSoftmaxKernel(n, h)
        self.proj = nn.Linear(h, h)

    def forward(self, x: Tensor) -> Tensor:
        # x: (H x L)
        x = x + self.kernel(x)
        x = self.proj(F.gelu(x))
        return x
