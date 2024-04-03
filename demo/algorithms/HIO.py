import numpy as np


def HIO(
        mag, mask=None, x_init=None, beta=0.6, steps=200
):
    """
    Implementation of Fienup's phase-retrieval methods with support.
    This function implements the hybrid input-output method.

    Parameters:
        mag: Measured magnitudes of Fourier transform
        mask: Binary array indicating where the image should be
        if padding is known
        x_init: input image,random init if x is None
        beta: Positive step size
        steps: Number of iterations

    Returns:
        x: reconstructed image
    """
    assert beta > 0
    assert steps > 0

    if mask is None:
        mask = np.ones(mag.shape)
    assert mask.shape == mag.shape

    G, g = None, None
    # sample random phase and initialize image x if x is None
    if x_init is None:
        random_phase = np.random.rand(*mag.shape)
        G = mag * np.exp(1j * random_phase * 2 * np.pi)
        g = np.real(np.fft.ifftn(G))
    else:
        G = np.fft.fftn(x_init)
        g = np.real(x_init)

    for i in range(steps):
        G_prime = mag * np.exp(1j * np.angle(G))
        g_prime = np.real(np.fft.ifftn(G_prime))

        indices = np.logical_and(mask, g_prime > 0)
        g[indices] = g_prime[indices]
        g[~indices] = g[~indices] - beta * g_prime[~indices]
        G = np.fft.fftn(g)

    g = np.real(np.fft.ifftn(G))

    return g
