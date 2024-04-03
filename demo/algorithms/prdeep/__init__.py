import torch
from demo.algorithms.fasta import *

__all__ = ['prdeep', 'ProxOpts']

class ProxOpts:
    def __init__(self, width, height, denoiser, prox_iters, sigma_w, lb, device="cpu"):
        self.width = width
        self.height = height
        self.denoiser = denoiser
        self.prox_iter = prox_iters
        self.sigma_w = sigma_w
        self.lb = lb
        self.device = device


def denoise(noisy, width, height, denoiser, device="cpu", rgb=False):
    if ~rgb:
        noisy_hat = torch.reshape(noisy, (1, 1, height, width))
    else:
        noisy_hat = torch.reshape(noisy, (1, 3, height, width))

    denoiser.eval()
    with torch.no_grad():
        noisy_hat = noisy_hat.to(device)
        noisy_hat = denoiser(noisy_hat)
        clean = noisy_hat.cpu()
        del noisy_hat

    return clean.reshape(-1, 1)


def iterative_prox_map(z, t, denoi, opts: ProxOpts):
    lb = opts.lb
    x = z
    prox_iter = opts.prox_iter
    for _ in range(prox_iter):
        x = (1.0 / (1 + t * lb)) * (z + t * lb * denoi(x))
    return x


def prdeep(A, At, b, x0, opts: fastaOpts, prox_opts: ProxOpts):
    """
    param A  : A matrix or function handle
    param At  : The adjoint/transpose of A
    b   : A column vector of measurements
    x0  : Initial guess of solution, often just a vector of zeros
    opts: Optional inputs to FASTA
    """
    # Define ingredienets for FASTA
    # note: fasta solves min f(Ax)+g(x)

    #  f(z) = 1/(2*sigma_w^2)|| abs(z) - b||^2
    f = lambda z: 1 / (2 * prox_opts.sigma_w ** 2) * torch.linalg.norm(torch.abs(z) - b, ord='fro') ** 2
    subgrad = lambda z: 1 / (prox_opts.sigma_w ** 2) * (z - b * z / torch.abs(z))

    denoi = lambda noisy: denoise(noisy, prox_opts.height, prox_opts.width, prox_opts.denoiser, prox_opts.device)

    # g(x) = 1/2*||y-|Ax|||^2 + lambda*x.T*(x-D(x))
    g = lambda x: prox_opts.lb / 2 * torch.matmul(torch.real(x).T, torch.real(x) - denoi(x))

    # proxg(z,t) = argmin .5||x-z||^2+t*g(x)
    prox = lambda z, t: iterative_prox_map(z, t, denoi, prox_opts)

    res = fasta(A, At, f, subgrad, g, prox, x0, opts)

    return res
