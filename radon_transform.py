import numpy as np
from scipy.fftpack import fft, ifft

# -----------------------------------------
# Utility Functions
# -----------------------------------------
def next_power_of_2(n):
    """Return next power of 2 for FFT padding."""
    return 2 ** (n - 1).bit_length()


# -----------------------------------------
# Core Radon Functions
# -----------------------------------------
def Radon_Transform(d, h, dt, qmin, qmax, nq, mode='adj', mu=1, maxiter=10):
    """
    Forward Radon Transform (time-offset → time-slowness).

    Args:
        d (ndarray): Data (nt x nx).
        h (ndarray): Offset vector.
        dt (float): Sampling rate (s).
        qmin, qmax (float): Slowness range.
        nq (int): Number of slowness samples.
        mode (str): 'adj' (Adjoint), 'LS' (Least Squares), 'IRLS' (Sparse).
        mu (float): Regularization parameter.
        maxiter (int): Iterations for IRLS.

    Returns:
        m (ndarray): Radon panel (nt x nq).
    """
    nt, nx = d.shape
    q = np.linspace(qmin, qmax, nq)
    nfft = next_power_of_2(nt)

    # FFT on time axis
    D = fft(d, n=nfft, axis=0)
    M = np.zeros((nfft, nq), dtype=np.complex128)

    ilow, ihigh = 0, nfft // 2
    f = 2 * np.pi / nfft / dt

    # Initialize operator and identity
    op = np.exp(-1j * f * h[:, None] @ q[None, :])
    L = np.ones((nx, nq))
    I = np.eye(nq)

    # Loop over positive frequencies
    for ifreq in range(ilow, ihigh):
        p = D[ifreq, :][:, None]

        if mode == 'adj':
            y = L.conj().T @ p
        elif mode == 'LS':
            rhs = L.conj().T @ p
            y = np.linalg.inv(L.conj().T @ L + mu * I) @ rhs
        elif mode == 'IRLS':
            y = IRLS(L, p, gamma=mu, maxiter=maxiter)
        else:
            raise ValueError("Invalid mode. Use 'adj', 'LS', or 'IRLS'.")

        M[ifreq, :] = y.flatten()
        L *= op  # Update operator for frequency

    m = 2 * ifft(M, n=nfft, axis=0).real
    return m[:nt, :]


def Inverse_Radon_Transform(m, h, dt, qmin, qmax):
    """
    Inverse Radon Transform (time-slowness → time-offset).
    """
    nt, nq = m.shape
    nx = len(h)
    q = np.linspace(qmin, qmax, nq)
    nfft = next_power_of_2(nt)

    M = fft(m, n=nfft, axis=0)
    D = np.zeros((nfft // 2, nx), dtype=np.complex128)

    f = 2 * np.pi / nfft / dt
    op = np.exp(1j * f * h[:, None] @ q[None, :]).conj()
    L = np.ones((nx, nq))

    for ifreq in range(0, nfft // 2):
        p = M[ifreq, :][:, None]
        y = L @ p
        D[ifreq, :] = y.flatten()
        L *= op

    d = 2 * ifft(D, n=nfft, axis=0).real
    return d[:nt, :]


# -----------------------------------------
# Auxiliary Functions
# -----------------------------------------
def IRLS(A, b, gamma, maxiter=10):
    """
    Iteratively Reweighted Least Squares solver.
    """
    n1, n2 = A.shape
    I = np.eye(n2)
    W = np.eye(n1)

    for _ in range(maxiter):
        rhs = gamma * A.conj().T @ W @ b
        x = np.linalg.inv(gamma * A.conj().T @ W @ A + I) @ rhs
        e = A @ x - b
        W = np.diag(1 / (np.sqrt(np.abs(e) ** 2 + 1e-5))).astype(float) ** 2
        I = np.diag(1 / (np.sqrt(np.abs(x) ** 2 + 1e-5 * np.max(np.abs(x))))).astype(float) ** 2

    return x


def PhVelEst(m, dt):
    """
    Estimate phase velocity distribution from Radon panel.
    """
    nt, nq = m.shape
    nfft = 2 * next_power_of_2(nt)
    M = fft(m, n=nfft, axis=0)
    ihigh = nfft // 2
    f = np.arange(ihigh) / (nfft * dt)
    M = np.abs(M[:ihigh, :].T)

    # Normalize per frequency
    for i in range(ihigh):
        M[:, i] /= (np.max(M[:, i]) + 1e-8)

    return M, f
