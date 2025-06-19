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

def generate_arr_PP_SS_mars(
    timeRange: np.ndarray,
    rhoRange: tuple,
    existing_arr_PP: np.ndarray,
    existing_wid_PP: np.ndarray,
    existing_wid_SS: np.ndarray,
    existing_rho: np.ndarray,
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
    existing_arr_SS = existing_arr_PP * existing_rho
    max_attempts = 1000

    for _ in range(max_attempts):
        candidate_rho = np.random.uniform(rhoRange[0], rhoRange[1])
        candidate_PP = np.random.uniform(tmin, tmax/candidate_rho) # make sure candidate_SS within range
        candidate_SS = candidate_PP * candidate_rho
        wid_PP = np.random.uniform(*wid_range)
        wid_SS = np.random.uniform(*wid_range)

        if existing_arr_PP.size > 0:
            dist_PP = np.abs(existing_arr_PP - candidate_PP)
            dist_SS = np.abs(existing_arr_SS - candidate_SS)
            overlap_thresh_PP = (existing_wid_PP + wid_PP) / 2
            overlap_thresh_SS = (existing_wid_SS + wid_SS) / 2
            if np.any(dist_PP < np.maximum(min_space, overlap_thresh_PP)) or np.any(dist_SS < np.maximum(min_space, overlap_thresh_SS)):
                continue  # Too close or overlapping
        return candidate_PP, wid_PP, wid_SS, candidate_rho

    raise ValueError("Could not generate a valid arrival time after many attempts.")