import os
from sharp_ss.visualization import plot_rjmcmc_results, plot_rjmcmc_results_PP_SS_mars
from parameter_setup import *

# ---- Load data dir ----
# PP or SS or Syn
if data_type in ["PP", "SS", "syn"]:
    if data_type == "PP": datadir = os.path.join(filedir, "data", PPdir)
    if data_type == "SS": datadir = os.path.join(filedir, "data", SSdir)
    if data_type == "syn": datadir = os.path.join(filedir, "data", syndir)
    npz = os.path.join(datadir, "data.npz")
# Joint
if data_type == "joint":
    npz_PP = os.path.join(filedir, "data", PPdir, "data.npz")
    npz_SS = os.path.join(filedir, "data", SSdir, "data.npz")

saveDir = os.path.join(filedir, "run", modname, runname)
if data_type in ["PP", "SS", "syn"]:
    plot_rjmcmc_results(
        saveDir, num_chains=num_chains, npz_filename=npz
        )
elif data_type == "joint":
    plot_rjmcmc_results_PP_SS_mars(
        saveDir,
        npz_PP=npz_PP,
        npz_SS=npz_SS,
        num_chains=num_chains
        )