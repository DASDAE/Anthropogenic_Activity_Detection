import numpy as np
from scipy.interpolate import interp1d
from scipy.ndimage import maximum_filter

# -----------------------------------------
# Utility Functions
# -----------------------------------------
def normalize(data):
    """
    Normalize data to range [-1, 1].
    """
    dmin, dmax = np.min(data), np.max(data)
    return 2 * (data - dmin) / (dmax - dmin) - 1


# -----------------------------------------
# Core Semblance Computation
# -----------------------------------------
def compute_semblance(data, h, dt, velocities):
    """
    Compute semblance in the velocity-time domain.
    
    Args:
        data (ndarray): 2D array of shape (nx, nt).
        h (ndarray): Offsets (length nx).
        dt (float): Time sampling interval (seconds).
        velocities (ndarray): Trial velocities.

    Returns:
        semblance (ndarray): 2D array of shape (nv, nt).
    """
    nx, nt = data.shape
    nv = len(velocities)
    semblance = np.zeros((nv, nt))
    t_idx = np.arange(nt)

    for iv, v in enumerate(velocities):
        if np.abs(v) < 1e-6:
            continue

        delays = -h / v / dt  # fractional delays in samples
        shifted = np.zeros((nx, nt))

        # Time-shift traces
        for ix in range(nx):
            t_shift = t_idx - delays[ix]
            f = interp1d(t_idx, data[ix, :], bounds_error=False, fill_value=0)
            shifted[ix, :] = f(t_shift)

        # Compute semblance
        num = np.square(np.sum(shifted, axis=0))
        den = shifted.var(axis=0) * nx + 1e-10
        semblance[iv] = num / (den * nx)

    return semblance


def compute_semblance_with_correlation_filter(data, h, dt, velocities, corr_threshold=0.9):
    """
    Compute semblance only if spatial coherence (cross-correlation) exceeds threshold.
    
    Args:
        data (ndarray): 2D array (nx, nt).
        h (ndarray): Offsets (length nx).
        dt (float): Sampling interval.
        velocities (ndarray): Trial velocities.
        corr_threshold (float): Minimum average correlation to compute semblance.

    Returns:
        semblance (ndarray): 2D array (nv, nt), or zeros if skipped.
    """
    nx, nt = data.shape
    nv = len(velocities)
    semblance = np.zeros((nv, nt))
    t_idx = np.arange(nt)

    # --- Compute average correlation across adjacent channels ---
    cross_corrs = []
    for i in range(nx - 1):
        trace1, trace2 = data[i], data[i + 1]
        std1, std2 = np.std(trace1), np.std(trace2)

        corr = np.corrcoef(trace1, trace2)[0, 1] if (std1 != 0 and std2 != 0) else 0.0
        cross_corrs.append(np.clip(corr, -1, 1))

    avg_corr = np.mean(cross_corrs)
    print(f"Average cross-correlation: {avg_corr:.3f}")

    # Skip if coherence is low
    if avg_corr < corr_threshold:
        print(f"Avg. corr = {avg_corr:.3f} < threshold = {corr_threshold}. Skipping semblance.")
        return semblance

    # --- Compute semblance ---
    for iv, v in enumerate(velocities):
        if np.abs(v) < 1e-6:
            continue

        delays = -h / v / dt
        shifted = np.zeros((nx, nt))

        for ix in range(nx):
            t_shift = t_idx - delays[ix]
            f = interp1d(t_idx, data[ix], bounds_error=False, fill_value=0)
            shifted[ix] = f(t_shift)

        num = np.square(np.sum(shifted, axis=0))
        den = shifted.var(axis=0) * nx + 1e-10
        semblance[iv] = num / (den * nx)

    return semblance


# -----------------------------------------
# Peak Detection in Velocity-Time Domain
# -----------------------------------------
def find_2d_peaks(beam, velocities, dt, threshold_ratio=0.5, window=5):
    """
    Identify local peaks in a 2D velocity-time semblance/beamforming map.
    
    Args:
        beam (ndarray): 2D array (nv x nt).
        velocities (ndarray): Velocity axis.
        dt (float): Time sampling interval.
        threshold_ratio (float): Relative threshold for peak detection.
        window (int): Neighborhood size for local maximum detection.

    Returns:
        peaks (list of tuples): [(velocity, time), ...].
        peak_mask (ndarray): Boolean mask of detected peaks.
    """
    # Local max detection
    local_max = maximum_filter(beam, size=(window, window)) == beam
    threshold = threshold_ratio * np.max(beam)
    peak_mask = (beam > threshold) & local_max

    # Extract indices
    peak_indices = np.argwhere(peak_mask)
    peak_velocities = velocities[peak_indices[:, 0]]
    peak_times = peak_indices[:, 1] * dt

    return list(zip(peak_velocities, peak_times)), peak_mask
