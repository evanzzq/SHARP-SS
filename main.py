import os, pickle
import time as pytime
import numpy as np
import multiprocessing as mp
from sharp_ss.rjmcmc import rjmcmc_run
from sharp_ss.rjmcmc_PP_SS_mars import rjmcmc_run_PP_SS_mars
from sharp_ss.model import Bookkeeping, Prior
from parameter_setup import *

# -------- Chain function --------
def run_chain(chain_id, exp_vars):
    filedir   = exp_vars["filedir"]
    modname   = exp_vars["modname"]
    runname   = exp_vars["runname"]
    data_type = exp_vars["data_type"]
    num_chains = exp_vars["num_chains"]

    # --- Save directory ---
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

    # --- Extract experiment variables ---
    prior        = exp_vars["prior"]
    bookkeeping  = exp_vars["bookkeeping"]
    CDinv        = exp_vars.get("CDinv", None)
    CDinv_PP     = exp_vars.get("CDinv_PP", None)
    CDinv_SS     = exp_vars.get("CDinv_SS", None)

    if data_type in ["PP", "SS", "syn", "synDL"]:
        P = exp_vars["P"]
        D = exp_vars["D"]
        ensemble, logL_trace = rjmcmc_run(P, D, prior, bookkeeping, saveDir, CDinv=CDinv)
    elif data_type == "joint":
        P_PP, P_SS = exp_vars["P_PP"], exp_vars["P_SS"]
        D_PP, D_SS = exp_vars["D_PP"], exp_vars["D_SS"]
        ensemble, logL_trace = rjmcmc_run_PP_SS_mars(
            P_PP, P_SS, D_PP, D_SS, prior, bookkeeping, saveDir,
            CDinv_PP=CDinv_PP, CDinv_SS=CDinv_SS
        )

    # --- Save results ---
    with open(os.path.join(saveDir, "ensemble.pkl"), "wb") as f:
        pickle.dump(ensemble, f)
    np.savetxt(os.path.join(saveDir, "log_likelihood.txt"), logL_trace)


# -------- Main run --------
if __name__ == "__main__":
    # --- Load static data ONCE ---
    CDinv = CDinv_PP = CDinv_SS = None

    if data_type in ["PP", "SS", "syn"]:
        if data_type == "PP": datadir = os.path.join(filedir, "data", PPdir)
        if data_type == "SS": datadir = os.path.join(filedir, "data", SSdir)
        if data_type == "syn": datadir = os.path.join(filedir, "data", syndir)
        data = np.load(os.path.join(datadir, "data.npz"))
        P, D, time = data["P"], data["D"], data["time"]
        if useCD:
            CD = np.loadtxt(os.path.join(datadir, "CD.csv"), delimiter=",")
            if negOnly:
                half_len = CD.shape[0] // 2
                CD = CD[:half_len, :half_len]
            CDinv = np.linalg.pinv(CD)

    elif data_type == "joint":
        data_PP = np.load(os.path.join(filedir, "data", PPdir, "data.npz"))
        P_PP, D_PP, time = data_PP["P"], data_PP["D"], data_PP["time"]

        data_SS = np.load(os.path.join(filedir, "data", SSdir, "data.npz"))
        P_SS, D_SS, time_SS = data_SS["P"], data_SS["D"], data_SS["time"]

        if len(time_SS) != len(time) or (time_SS[1] - time_SS[0]) != (time[1] - time[0]):
            raise ValueError("Time vector for PP and SS don't match!")

        if useCD:
            CD_PP = np.loadtxt(os.path.join(filedir, "data", PPdir, "CD.csv"), delimiter=",")
            CD_SS = np.loadtxt(os.path.join(filedir, "data", SSdir, "CD.csv"), delimiter=",")
            if negOnly:
                half_len = CD_PP.shape[0] // 2
                CD_PP = CD_PP[:half_len, :half_len]
                half_len = CD_SS.shape[0] // 2
                CD_SS = CD_SS[:half_len, :half_len]
            CDinv_PP = np.linalg.pinv(CD_PP)
            CDinv_SS = np.linalg.pinv(CD_SS)

    elif data_type == "synDL":
        P_tmp = np.loadtxt(os.path.join(filedir, Pfile), delimiter=",", skiprows=1)
        time, P = P_tmp[:, 0], P_tmp[:, 1]
        datadir = os.path.join(filedir, "data", syndir)
        D_tmp = f"data_denoised_l{DLmod[0]}x{DLmod[0]}_s{DLmod[1]}x{DLmod[1]}_layers{DLmod[2]}_ep{DLmod[3]}.csv"
        D = np.loadtxt(os.path.join(datadir, "DL_denoise", D_tmp), delimiter=",")

    # --- Prior (shared across chains) ---
    if data_type in ["syn", "synDL"]:
        with open(os.path.join(datadir, "prior.pkl"), "rb") as f:
            prior = pickle.load(f)
    else:
        prior = Prior(
            stdP=stdP, maxN=maxN, tlen=(len(time) // 2) * (time[1] - time[0]), dt=time[1] - time[0],
            ampRange=ampRange, widRange=widRange,
            negOnly=negOnly, align=align
        )
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

    # --- Multiprocessing setup ---
    try:
        import psutil
        cpu_cores = psutil.cpu_count(logical=False) or mp.cpu_count()
    except ImportError:
        cpu_cores = mp.cpu_count()

    print(f"Detected {cpu_cores} physical cores.")

    if num_chains >= 2:
        threads_per_chain = max(1, cpu_cores // min(num_chains, cpu_cores))
        os.environ["OMP_NUM_THREADS"] = str(threads_per_chain)
        os.environ["MKL_NUM_THREADS"] = str(threads_per_chain)
        print(f"Setting {threads_per_chain} threads per chain to avoid oversubscription.")
    else:
        print("Single chain: allow full multithreading.")

    ctx = mp.get_context("spawn")
    batch_size = cpu_cores
    total_batches = (num_chains + batch_size - 1) // batch_size

    # --- Package experiment variables ---
    exp_vars = {
        "filedir": filedir, "modname": modname, "runname": runname,
        "data_type": data_type, "num_chains": num_chains,
        "prior": prior, "bookkeeping": bookkeeping,
        "CDinv": CDinv, "CDinv_PP": CDinv_PP, "CDinv_SS": CDinv_SS
    }
    if data_type in ["PP", "SS", "syn", "synDL"]:
        exp_vars["P"], exp_vars["D"] = P, D
    elif data_type == "joint":
        exp_vars["P_PP"], exp_vars["P_SS"] = P_PP, P_SS
        exp_vars["D_PP"], exp_vars["D_SS"] = D_PP, D_SS

    # --- Run all chains in batches ---
    start = pytime.time()
    for batch_idx in range(total_batches):
        start_chain = batch_idx * batch_size
        end_chain = min(start_chain + batch_size, num_chains)
        batch_chain_ids = list(range(start_chain, end_chain))
        print(f"Running batch {batch_idx+1}/{total_batches} with chains {batch_chain_ids}")

        with ctx.Pool(processes=len(batch_chain_ids)) as pool:
            pool.starmap(run_chain, [(cid, exp_vars) for cid in batch_chain_ids])

    end = pytime.time()
    print(f"Total elapsed time: {end - start:.2f} seconds")
