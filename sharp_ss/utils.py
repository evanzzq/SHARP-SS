import numpy as np

def generate_arr(
    timeRange: np.ndarray,
    existing_arr: np.ndarray,
    existing_wid: np.ndarray,
    min_space: float,
    wid_range: tuple
) -> tuple[float, float]:
    """
    Generate a random arrival time and width within the time range,
    ensuring it does not overlap with existing phases and maintains a minimum spacing.

    Parameters:
    - timeRange: np.ndarray, the global time vector (only first and last used)
    - existing_arr: np.ndarray, current list of arrival times
    - existing_wid: np.ndarray, current list of phase widths
    - min_space: float, minimum spacing required between arrivals
    - wid_range: tuple, range of possible widths for the new phase

    Returns:
    - tuple (float, float): the new valid arrival time and width
    """
    tmin, tmax = timeRange[0], timeRange[-1]
    max_attempts = 1000

    for _ in range(max_attempts):
        candidate = np.random.uniform(tmin, tmax)
        wid = np.random.uniform(*wid_range)

        if existing_arr.size > 0:
            dist = np.abs(existing_arr - candidate)
            overlap_thresh = (existing_wid + wid) / 2
            if np.any(dist < np.maximum(min_space, overlap_thresh)):
                continue  # Too close or overlapping
        return candidate, wid

    raise ValueError("Could not generate a valid arrival time after many attempts.")