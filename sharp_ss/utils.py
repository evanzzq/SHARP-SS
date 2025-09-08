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

def prepare_experiment(exp_vars):
    """
    Load data and prior for one experiment, based on exp_vars.
    Modifies and returns exp_vars dict.
    """

    import os, pickle
    from sharp_ss.model import Prior

    filedir   = exp_vars["filedir"]
    event_name = exp_vars["event_name"]
    data_type = exp_vars["data_type"]
    runname   = exp_vars["runname"]
    negOnly   = exp_vars["negOnly"]

    PPdir = event_name + "_PP"
    SSdir = event_name + "_SS"
    syndir = event_name

    CDinv = CDinv_PP = CDinv_SS = None

    # --- Load data ---
    if data_type in ["PP", "SS", "syn"]:
        if data_type == "PP": datadir = os.path.join(filedir, "data", PPdir)
        if data_type == "SS": datadir = os.path.join(filedir, "data", SSdir)
        if data_type == "syn": datadir = os.path.join(filedir, "data", syndir)

        data = np.load(os.path.join(datadir, "data.npz"))
        exp_vars["P"], exp_vars["D"], time = data["P"], data["D"], data["time"]

        if exp_vars["useCD"]:
            CD = np.loadtxt(os.path.join(datadir, "CD.csv"), delimiter=",")
            if negOnly:
                half_len = CD.shape[0] // 2
                CD = CD[:half_len, :half_len]
            CDinv = np.linalg.pinv(CD)
            exp_vars["CDinv"] = CDinv

    elif data_type == "joint":
        data_PP = np.load(os.path.join(filedir, "data", PPdir, "data.npz"))
        P_PP, D_PP, time = data_PP["P"], data_PP["D"], data_PP["time"]

        data_SS = np.load(os.path.join(filedir, "data", SSdir, "data.npz"))
        P_SS, D_SS, time_SS = data_SS["P"], data_SS["D"], data_SS["time"]

        if len(time_SS) != len(time) or (time_SS[1] - time_SS[0]) != (time[1] - time[0]):
            raise ValueError("Time vector for PP and SS don't match!")

        exp_vars["P_PP"], exp_vars["D_PP"] = P_PP, D_PP
        exp_vars["P_SS"], exp_vars["D_SS"] = P_SS, D_SS

        if exp_vars["useCD"]:
            CD_PP = np.loadtxt(os.path.join(filedir, "data", PPdir, "CD.csv"), delimiter=",")
            CD_SS = np.loadtxt(os.path.join(filedir, "data", SSdir, "CD.csv"), delimiter=",")
            if negOnly:
                half_len = CD_PP.shape[0] // 2
                CD_PP = CD_PP[:half_len, :half_len]
                half_len = CD_SS.shape[0] // 2
                CD_SS = CD_SS[:half_len, :half_len]
            exp_vars["CDinv_PP"] = np.linalg.pinv(CD_PP)
            exp_vars["CDinv_SS"] = np.linalg.pinv(CD_SS)

    elif data_type == "synDL":
        P_tmp = np.loadtxt(os.path.join(filedir, exp_vars["Pfile"]), delimiter=",", skiprows=1)
        time, P = P_tmp[:, 0], P_tmp[:, 1]
        datadir = os.path.join(filedir, "data", syndir)
        D_tmp = f"data_denoised_l{exp_vars['DLmod'][0]}x{exp_vars['DLmod'][0]}_s{exp_vars['DLmod'][1]}x{exp_vars['DLmod'][1]}_layers{exp_vars['DLmod'][2]}_ep{exp_vars['DLmod'][3]}.csv"
        D = np.loadtxt(os.path.join(datadir, "DL_denoise", D_tmp), delimiter=",")
        exp_vars["P"], exp_vars["D"] = P, D

    # --- Prior ---
    if data_type in ["syn", "synDL"]:
        with open(os.path.join(datadir, "prior.pkl"), "rb") as f:
            prior = pickle.load(f)
    else:
        dt = time[1] - time[0]
        tlen = (len(time) // 2) * dt
        prior = Prior(
            stdP=exp_vars["stdP"], maxN=exp_vars["maxN"],
            tlen=tlen, dt=dt,
            ampRange=tuple(exp_vars["ampRange"]),
            widRange=tuple(exp_vars["widRange"]),
            negOnly=exp_vars["negOnly"], align=exp_vars["align"]
        )
        sharedDir = os.path.join(filedir, "run", exp_vars["modname"], runname)
        os.makedirs(sharedDir, exist_ok=True)
        with open(os.path.join(sharedDir, "prior.pkl"), "wb") as f:
            pickle.dump(prior, f)

    exp_vars["prior"] = prior
    return exp_vars
