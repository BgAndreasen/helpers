import numpy as np

def calc_principal_heading(u, v, tidal_mode=True):
    """
    This is an adaption of the function in the MHKit python package,
    https://github.com/MHKiT-Software/MHKiT-Python

    u, v : 2D numpy arrays with shape (time, depth)
    Returns:
      1D numpy array with shape (depth,)
    """
    # combine into complex
    dt = u + 1j * v

    if tidal_mode:
        # flip negative imag
        mask_neg = np.imag(dt) <= 0
        dt = np.where(mask_neg, -dt, dt)

        # double angles
        angles = np.angle(dt)
        dt = dt * np.exp(1j * angles)

        # mask invalid
        dt = np.where(np.isfinite(dt), dt, np.nan)

        # mean over time‑axis (axis=0)
        mean_dt = np.nanmean(dt, axis=0)
        pang = np.angle(mean_dt) / 2
    else:
        mean_dt = np.nanmean(dt, axis=0)
        pang = np.angle(mean_dt)

    # to degrees, 0–360
    p_heading = (np.rad2deg(pang) % 360).round(4)
    return p_heading.astype(np.float32)