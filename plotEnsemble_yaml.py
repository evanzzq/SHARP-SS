import os, yaml, pickle
from sharp_ss.visualization import plot_rjmcmc_results, plot_rjmcmc_results_PP_SS_mars

# ==== Config ====
yaml_file = "parameter_setup.yaml"
ensemble_filename = "ensemble.pkl"   # or whatever name you use
logL_filename = "log_likelihood.txt"

# ---- Load YAML config ----
with open(yaml_file, "r") as f:
    config = yaml.safe_load(f)

common = config["common"]
experiments = config["experiments"]

# ---- Prompt user to choose an experiment ----
print("Available experiments:")
for i, exp in enumerate(experiments):
    display_name = f"{exp['event_name']}_{exp['data_type']}_{exp['runname']}"
    print(f"{i}: {display_name}")

choice = int(input("Select experiment index: "))
exp_params = experiments[choice]

# ---- Combine common + experiment params ----
params = {**common, **exp_params}

# ---- Construct directories ----
filedir = params["filedir"]
event_name = params["event_name"]
data_type = params["data_type"]
runname = params["runname"]
num_chains = params["num_chains"]

PPdir = f"{event_name}_PP"
SSdir = f"{event_name}_SS"
syndir = f"{event_name}"

saveDir = os.path.join(filedir, "run", f"{event_name}_{data_type}", runname)
with open(os.path.join(saveDir, "prior.pkl"), "rb") as f:
    prior = pickle.load(f)

# ---- Load data files ----
if data_type in ["PP", "SS", "syn"]:
    if data_type == "PP":
        datadir = os.path.join(filedir, "data", PPdir)
    elif data_type == "SS":
        datadir = os.path.join(filedir, "data", SSdir)
    else:
        datadir = os.path.join(filedir, "data", syndir)
    npz_file = os.path.join(datadir, "data.npz")

elif data_type == "joint":
    npz_PP = os.path.join(filedir, "data", PPdir, "data.npz")
    npz_SS = os.path.join(filedir, "data", SSdir, "data.npz")

# ---- Collect ensembles from chains ----
ensemble_all = []
if num_chains > 1:
    chain_logLs = []
    for i in range(num_chains):
        chain_dir = os.path.join(saveDir, f"chain_{i}")
        logL_file = os.path.join(chain_dir, logL_filename)
        if not os.path.exists(logL_file):
            continue
        with open(logL_file, "r") as f:
            lines = f.readlines()
            if len(lines) == 0:
                continue
            logL = float(lines[-1].strip())  # last row = final logL
        chain_logLs.append((i, logL))

    # Sort chains by final logL (descending)
    chain_logLs.sort(key=lambda x: x[1], reverse=True)
    
    # Plot histogram of final logL values
    import matplotlib.pyplot as plt
    plt.hist([x[1] for x in chain_logLs], bins=10, edgecolor="k")
    plt.xlabel("Final log-likelihood")
    plt.ylabel("Number of chains")
    plt.title("Histogram of chain final logL")
    plt.show()

    # Ask user for number of top chains
    user_input = input("Enter number of top chains to use (press Enter to use all): ")
    if user_input.strip() == "":
        top_chains = None
    else:
        top_chains = int(user_input)

    top_ids = [i for i, _ in chain_logLs]
    if top_chains is not None:
        top_ids = top_ids[:top_chains]
    
    print(f"Selected chains (sorted by final logL): {top_ids}")

    # Load only top chains
    for i in top_ids:
        chain_dir = os.path.join(saveDir, f"chain_{i}")
        with open(os.path.join(chain_dir, ensemble_filename), "rb") as f:
            ensemble = pickle.load(f)
        ensemble_all.extend(ensemble)

else:
    # Single-chain case
    with open(os.path.join(saveDir, ensemble_filename), "rb") as f:
        ensemble_all = pickle.load(f)

# ---- Plot ----
if data_type in ["PP", "SS", "syn"]:
    plot_rjmcmc_results(ensemble_all, prior, npz_filename=npz_file)
elif data_type == "joint":
    plot_rjmcmc_results_PP_SS_mars(ensemble_all, prior, npz_PP=npz_PP, npz_SS=npz_SS)
