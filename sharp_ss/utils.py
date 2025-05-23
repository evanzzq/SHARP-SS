import numpy as np

def generate_arr(timeRange: np.ndarray, existing_arr: np.ndarray, min_space: float) -> float:
    """
    Generate a random arrival time within the time range,
    ensuring it is at least `min_space` away from all existing arrivals.

    Parameters:
    - time: np.ndarray, the global time vector
    - existing_arr: np.ndarray, current list of arrival times
    - min_space: float, minimum spacing required between arrivals

    Returns:
    - float, the new valid arrival time
    """
    tmin, tmax = timeRange[0], timeRange[-1]
    max_attempts = 1000

    for _ in range(max_attempts):
        candidate = np.random.uniform(tmin, tmax)
        if np.all(np.abs(existing_arr - candidate) >= min_space):
            return candidate

    raise ValueError("Could not generate a valid arrival time after many attempts.")
