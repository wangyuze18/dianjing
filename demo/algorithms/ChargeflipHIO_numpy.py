import numpy as np
import cv2
import scipy.io as sio
from matplotlib import pyplot as plt
import time


def normalize(image: np.ndarray):
    max_pixels = np.max(image, axis=(-1, -2))
    min_pixels = np.min(image, axis=(-1, -2))
    return (image - min_pixels) / (max_pixels - min_pixels)


def estimate_Rfactor(mag, x_mag):
    Rfactor = np.sum(np.abs(mag * temp_R - x_mag * temp_R)) / mag.sum()
    return Rfactor


def hio(
        mag, x_init=None, beta=0.6, percent=0.32, steps=200, verbose=True
):
    """
    Implementation of Fienup's phase-retrieval methods with support.
    This function implements the hybrid input-output method.

    Parameters:
        mag: Measured magnitudes of Fourier transform
        if padding is known
        x_init: input image,random init if x is None
        beta: Positive step size
        steps: Number of iterations
        verbose: If True, progress is shown

    Returns:
        x: reconstructed image
        R_hist: the record of Rfactor
    """
    assert beta > 0
    assert steps > 0

    # sample random phase and initialize image x if x is None
    if x_init is None:
        y_hat = mag * np.exp(1j * 2 * np.pi * np.random.rand(mag.shape[0], mag.shape[1]))
        x = np.real(np.fft.ifftn(y_hat))
    else:
        x = x_init

    if verbose:
        R_hist = np.array([0.0] * steps)
    else:
        R_hist = None

    # main loop
    for i in range(0, steps):

        # fourier transform
        x_hat = np.fft.fftn(x)

        if verbose:
            R = estimate_Rfactor(mag, np.abs(x_hat))
            R_hist[i] = R
            print(f'HIO step {i + 1} of {steps} Rfactor: {R}')

        # satisfy fourier domain constraints
        # (replace magnitude with input magnitude)
        y_hat = mag * np.exp(1j * np.angle(x_hat))

        # replace current (000)
        y_hat[temp_0] = x_hat[temp_0]

        # inverse fourier transform
        y = np.real(np.fft.ifftn(y_hat))

        x_p = x

        # updates for elements that satisfy object domain constraints
        x = y

        indices = apply_dynamic_support_constraints(y, percent)

        # updates for elements that violate object domain constraints
        x[indices] = x_p[indices] - beta * y[indices]

    return x, R_hist


def error_reduction(
        mag, x_init=None, steps=200, percent=0.17, verbose=True
):
    """
    Implementation of error-reduction algorithm,a modification of the Gerchberg-Saxton (GS) algorithm

    Parameters:
        mag: Measured magnitudes of Fourier transform
        mask: Binary array indicating where the image should be
        if padding is known
        x_init: input image,random init if x is None
        steps: Number of iterations
        verbose: If True, progress is shown

    Returns:
        x: reconstructed image
        R_hist: the record of Rfactor
    """
    assert steps > 0

    # sample random phase and initialize image x if x is None
    if x_init is None:
        y_hat = mag * np.exp(1j * 2 * np.pi * np.random.rand(mag.shape[0], mag.shape[1]))
        x = np.real(np.fft.ifftn(y_hat))
    else:
        x = x_init

    if verbose:
        R_hist = np.array([0.0] * steps)
    else:
        R_hist = None

    # main loop
    for i in range(0, steps):

        # fourier transform
        x_hat = np.fft.fftn(x)

        if verbose:
            R = estimate_Rfactor(mag, np.abs(x_hat))
            R_hist[i] = R
            print(f'ER step {i + 1} of {steps} Rfactor: {R}')

        # satisfy fourier domain constraints
        # (replace magnitude with input magnitude)
        y_hat = mag * np.exp(1j * np.angle(x_hat))

        # replace current (000)
        y_hat[temp_0] = x_hat[temp_0]

        # inverse fourier transform
        y = np.real(np.fft.ifftn(y_hat))

        # updates for elements that satisfy object domain constraints
        x = y

        indices = np.logical_or(apply_dynamic_support_constraints(y, percent), y < 0)
        # updates for elements that violate object domain constraints
        x[indices] = 0

    return x, R_hist


def charge_flipping(
        mag, x_init=None, percent=0.17, steps=200, verbose=True
):
    """
    Implementation of charge-flipping algorithm

    Parameters:
        mag: Measured magnitudes of Fourier transform
        if padding is known
        x_init: input image,random init if x is None
        steps: Number of iterations
        percent: Pixels above the threshold(top percent) should be unchanged at each iteration
        verbose: If True, progress is shown

    Returns:
        x: reconstructed image
        R_hist: the record of Rfactor
    """
    assert steps > 0

    # sample random phase and initialize image x if x is None
    if x_init is None:
        y_hat = mag * np.exp(1j * 2 * np.pi * np.random.rand(mag.shape[0], mag.shape[1]))
        x = np.real(np.fft.ifftn(y_hat))
    else:
        x = x_init

    if verbose:
        R_hist = np.array([0.0] * steps)
    else:
        R_hist = None

    # main loop
    for i in range(0, steps):

        # fourier transform
        x_hat = np.fft.fftn(x)

        if verbose:
            R = estimate_Rfactor(mag, np.abs(x_hat))
            R_hist[i] = R
            print(f'charge flipping step {i + 1} of {steps} Rfactor: {R}')

        # satisfy fourier domain constraints
        # (replace magnitude with input magnitude)
        y_hat = mag * np.exp(1j * np.angle(x_hat))

        # replace current (000)
        y_hat[temp_0] = x_hat[temp_0]

        # inverse fourier transform
        y = np.real(np.fft.ifftn(y_hat))

        # updates for elements that satisfy object domain constraints
        x = np.abs(y)

        # get the indices of elements that violate object domain constraints
        indices = apply_dynamic_support_constraints(y, percent)

        # updates for elements that violate object domain constraints
        x[indices] = -np.real(y[indices])

    return x, R_hist


def apply_dynamic_support_constraints(y, percent):
    size = y.size
    support_num = round(size * (1 - percent))
    ima_array = y.flatten()
    indices = np.argpartition(ima_array, support_num - 1)
    thresold = ima_array[indices[support_num - 1]]
    flip = (y < thresold)
    return flip


def solver(ima, sol, verbose=True):
    """
        solver of three algorithms include chargeflip(CF),hio(HIO),error_reduction(ER)

        Parameters:
            sol:a list of (algorithm name,eliminate percent,iteration step)

        Returns:
            y: reconstructed image
            R_hist: the record of Rfactor
    """
    # get the starting amptitude
    mag = np.abs(np.fft.fftn(ima))

    phi = np.random.rand(*mag.shape) * 2 * np.pi  # random selected initial phases

    # find the (000) position
    global temp_0
    temp_0 = np.where(mag == np.max(mag))

    # write down the position to calculate the R in our draft
    global temp_R
    temp_R = mag > 0

    phase = np.exp(1j * phi)
    y_hat = mag * phase  # initial diff pattern - known amps and random phases.
    y = np.real(np.fft.ifftn(y_hat))  # first estimate of image

    x = y
    x_hat = np.fft.fftn(x)
    phase = np.exp(1j * np.angle(x_hat))
    y_hat = mag * np.exp(1j * phase)  # replace with known fourier modulus (diffracted intensities)
    y_hat[temp_0] = x_hat[temp_0]  # use the current (000) beam
    y = np.real(np.fft.ifftn(y_hat))  # next estimate of image.
    Rhist = None
    if verbose:
        Rhist = []
    for sol_name, sol_percent, sol_steps in sol:
        if sol_name == 'CF':
            y, r = charge_flipping(mag=mag, percent=sol_percent, x_init=y, steps=sol_steps)
        elif sol_name == 'HIO':
            y, r = hio(mag=mag, percent=sol_percent, x_init=y, steps=sol_steps)
        elif sol_name == 'ER':
            y, r = error_reduction(mag=mag, percent=sol_percent, x_init=y, steps=sol_steps)
        if verbose:
            Rhist.extend(r)

    return y, Rhist
