import os, pickle
import numpy as np
from multiprocessing import Process
from sharp_ss.rjmcmc import rjmcmc_run
from sharp_ss.rjmcmc_PP_SS_mars import rjmcmc_run_PP_SS_mars
from sharp_ss.model import Bookkeeping, Prior
from parameter_setup import *

def run_chain(chain_id):
    # --- Create save directory ---
    if num_chains == 1:
        if data_type in ["PP", "SS", "joint", "syn"]:
            saveDir = os.path.join(filedir, "run", modname, runname)
        elif data_type == "synDL":
            DLname = f"{DLmod[0]}_{DLmod[1]}_{DLmod[2]}_{DLmod[3]}"
            saveDir = os.path.join(filedir, "run", modname, DLname, runname)
    else:
        if data_type in ["PP", "SS", "joint", "syn"]:
            saveDir = os.path.join(filedir, "run", modname, runname, f"chain_{chain_id}")
        elif data_type == "synDL":
            DLname = f"{DLmod[0]}_{DLmod[1]}_{DLmod[2]}_{DLmod[3]}"
            saveDir = os.path.join(filedir, "run", modname, DLname, runname, f"chain_{chain_id}")
    os.makedirs(saveDir, exist_ok=True)

    # --- Initialize CDinv as None ---
    CDinv = None
    CDinv_PP = None
    CDinv_SS = None

    # --- Load data ---
    if data_type in ["PP", "SS", "syn"]:
        if data_type == "PP": datadir = os.path.join(filedir, "data", PPdir)
        if data_type == "SS": datadir = os.path.join(filedir, "data", SSdir)
        if data_type == "syn": datadir = os.path.join(filedir, "data", syndir)
        data = np.load(os.path.join(datadir, "data.npz"))
        P = data["P"]
        D = data["D"]
        time = data["time"]
        if useCD:
            CD = np.loadtxt(os.path.join(datadir, "CD.csv"), delimiter=",")
            if negOnly:
                half_len = CD.shape[0] // 2
                CD = CD[:half_len, :half_len]
            CDinv = np.linalg.pinv(CD)
    elif data_type == "joint":
        data_PP = np.load(os.path.join(filedir, "data", PPdir, "data.npz"))
        P_PP = data_PP["P"]
        D_PP = data_PP["D"]
        time = data_PP["time"]
        data_SS = np.load(os.path.join(filedir, "data", SSdir, "data.npz"))
        P_SS = data_SS["P"]
        D_SS = data_SS["D"]
        time_SS = data_SS["time"]
        if len(time_SS) != len(time) or (time_SS[1] - time_SS[0]) != (time[1] - time[0]):
            raise ValueError("Time vector for PP and SS don't match!")
        if useCD:
            CD_PP = np.loadtxt(os.path.join(PPdir, "CD.csv"), delimiter=",")
            CD_SS = np.loadtxt(os.path.join(SSdir, "CD.csv"), delimiter=",")
            if negOnly:
                half_len = CD_PP.shape[0] // 2
                CD_PP = CD_PP[:half_len, :half_len]
                half_len = CD_SS.shape[0] // 2
                CD_SS = CD_SS[:half_len, :half_len]
            CDinv_PP = np.linalg.pinv(CD_PP)
            CDinv_SS = np.linalg.pinv(CD_SS)

    elif data_type == "synDL":
        P_tmp = np.loadtxt(os.path.join(filedir, Pfile), delimiter=",", skiprows=1)
        time = P_tmp[:, 0]
        P = P_tmp[:, 1]
        datadir = os.path.join(filedir, "data", syndir)
        D_tmp = f"data_denoised_l{DLmod[0]}x{DLmod[0]}_s{DLmod[1]}x{DLmod[1]}_layers{DLmod[2]}_ep{DLmod[3]}.csv"
        D = np.loadtxt(os.path.join(datadir, "DL_denoise", D_tmp), delimiter=",")

    # --- Load or create Prior ---
    if data_type in ["syn", "synDL"]:
        with open(os.path.join(datadir, "prior.pkl"), "rb") as f:
            prior = pickle.load(f)
    else:
        prior = Prior(
            stdP=stdP, maxN=maxN, tlen=time[-1], dt=time[1] - time[0],
            ampRange=ampRange, widRange=widRange,
            negOnly=negOnly, align=align
        )
        # Save only once (chain 0) to shared parent directory
        if chain_id == 0:
            sharedDir = os.path.join(filedir, "run", modname, runname)
            os.makedirs(sharedDir, exist_ok=True)
            with open(os.path.join(sharedDir, "prior.pkl"), "wb") as f:
                pickle.dump(prior, f)

    # --- Bookkeeping ---
    bookkeeping = Bookkeeping(
        totalSteps=totalSteps,
        burnInSteps=burnInSteps,
        nSaveModels=nSaveModels,
        actionsPerStep=actionsPerStep
    )

    # --- Run RJMCMC ---
    if data_type in ["PP", "SS", "syn", "synDL"]:
        ensemble, _ = rjmcmc_run(P, D, prior, bookkeeping, saveDir, CDinv=CDinv)
    elif data_type == "joint":
        ensemble, _ = rjmcmc_run_PP_SS_mars(P_PP, P_SS, D_PP, D_SS, prior, bookkeeping, saveDir, CDinv_PP=CDinv_PP, CDinv_SS=CDinv_SS)

    # --- Save results ---
    with open(os.path.join(saveDir, "ensemble.pkl"), "wb") as f:
        pickle.dump(ensemble, f)

# --- Run ---
if __name__ == "__main__":
    if num_chains == 1:
        run_chain(0)
    else:
        processes = []
        for i in range(num_chains):
            p = Process(target=run_chain, args=(i,))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
