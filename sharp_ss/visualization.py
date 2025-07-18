import matplotlib.pyplot as plt
import numpy as np
import pickle, os
from sharp_ss.forward import create_G_from_model, convolve_P_G

def plot_G_2dhist(ax, G_array, time, title="Model Ensemble"):
    nbins_time = len(time)
    nbins_amp = 100
    amp_min = np.min(G_array)
    amp_max = np.max(G_array)
    amp_bins = np.linspace(amp_min, amp_max, nbins_amp)
    hist2d = np.zeros((nbins_amp - 1, nbins_time))

    for G in G_array:
        inds = np.digitize(G, amp_bins) - 1
        for i, ind in enumerate(inds):
            if 0 <= ind < nbins_amp - 1:
                hist2d[ind, i] += 1

    hist2d /= np.max(hist2d)  # Normalize

    extent = [time[0], time[-1], amp_bins[0], amp_bins[-1]]
    ax.imshow(hist2d, aspect='auto', extent=extent, origin='lower', cmap='hot')
    ax.plot(time, np.mean(G_array, axis=0), color="cyan", linewidth=0.5, label="Mean")
    ax.set_title(title)
    ax.set_ylabel("Amplitude")
    ax.grid(True, linestyle='--', linewidth=0.4)
    ax.legend()

def plot_rjmcmc_results(base_dir, num_chains=1, ensemble_filename="ensemble.pkl", npz_filename="data.npz"):
    """
    Plot RJMCMC G ensemble and D predictions from one or multiple chains.

    Args:
        base_dir (str): Directory containing prior and .npz file, and either ensemble.pkl or chain subfolders.
        num_chains (int): Number of chains (if 1, assumes single ensemble.pkl in base_dir).
        ensemble_filename (str): Filename of the ensemble.pkl in each chain folder.
        npz_filename (str): Name of the .npz file with P, D, and time.
    """
    
    # --- Load ensemble(s) ---
    ensemble_all = []

    if num_chains == 1:
        with open(os.path.join(base_dir, ensemble_filename), "rb") as f:
            ensemble_all = pickle.load(f)
    else:
        for i in range(num_chains):
            chain_dir = os.path.join(base_dir, f"chain_{i}")
            with open(os.path.join(chain_dir, ensemble_filename), "rb") as f:
                ensemble = pickle.load(f)
            ensemble_all.extend(ensemble)

    # --- Load prior and data ---
    with open(os.path.join(base_dir, "prior.pkl"), "rb") as f:
        prior = pickle.load(f)

    data = np.load(npz_filename)
    P = data["P"]
    D_obs = data["D"]
    time_PD = data["time"]

    # --- Time vector from prior ---
    Ntime = int(prior.tlen / prior.dt) * 2 - 1
    time = np.linspace(-prior.tlen, prior.tlen, Ntime, endpoint=False)

    # --- Forward G and D = P * G ---
    G_list = [create_G_from_model(model, prior) for model in ensemble_all]
    G_array = np.vstack(G_list)

    D_pred_list = [convolve_P_G(P, G) for G in G_array]
    D_pred_array = np.vstack(D_pred_list)
    D_pred_mean = np.mean(D_pred_array, axis=0)

    # --- Plot ---
    plt.figure(figsize=(12, 8))

    # Plot 2D histogram
    ax = plt.subplot(2, 1, 1)
    plot_G_2dhist(ax, G_array, time)

    # D plot
    plt.subplot(2, 1, 2)
    for D_pred in D_pred_array:
        plt.plot(time_PD, D_pred, color="gray", alpha=0.2, linewidth=0.1)
    plt.plot(time_PD, D_pred_mean, color="black", linewidth=1.5, label="Predicted Mean")
    plt.plot(time_PD, D_obs, linestyle="--", color="red", linewidth=1.2, label="Observed D")
    plt.title("Predicted D = P * G vs Observed D")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.minorticks_on()
    plt.legend()

    plt.tight_layout()
    plt.show()

def plot_rho_distribution(ax, ensemble, tlim, bins=(100, 100)):
    """
    Visualize rho distribution across ensemble using locPP vs rho.

    Args:
        ax: matplotlib Axes object to plot on
        ensemble: list of Model2 objects
        bins: tuple of (xbins, ybins) for 2D histogram
    """
    loc_all = []
    rho_all = []

    for model in ensemble:
        if model.Nphase == 0:
            continue
        loc_all.extend(model.locPP)
        rho_all.extend(model.rho)

    loc_all = np.array(loc_all)
    rho_all = np.array(rho_all)

    # 2D histogram
    h = ax.hist2d(loc_all, rho_all, bins=bins, cmap='hot')

    # # Colorbar
    # cbar = ax.figure.colorbar(h[3], ax=ax, label="Count")

    # Labels
    ax.set_xlabel("locPP (s)")
    ax.set_ylabel("ρ (rho)")
    ax.set_title("Ensemble Distribution of ρ vs locPP")
    ax.set_xlim(tlim)

def plot_rjmcmc_results_PP_SS_mars(
    base_dir,
    npz_PP,
    npz_SS,
    num_chains=1,
    rho=1.8
):
    """
    Plot ensemble diagnostics (G and D predictions) for joint PP and SS inversion.
    Supports both single-chain and multi-chain experiments.
    """

    from sharp_ss.model import Model

    # --- Load ensemble(s) ---
    ensemble_all = []
    if num_chains == 1:
        with open(os.path.join(base_dir, "ensemble.pkl"), "rb") as f:
            ensemble_all = pickle.load(f)
    else:
        for i in range(num_chains):
            chain_dir = os.path.join(base_dir, f"chain_{i}")
            with open(os.path.join(chain_dir, "ensemble.pkl"), "rb") as f:
                ensemble = pickle.load(f)
            ensemble_all.extend(ensemble)

    # Load prior
    with open(os.path.join(base_dir, "prior.pkl"), "rb") as f:
        prior = pickle.load(f)

    # Load PP data
    data_PP = np.load(npz_PP)
    P_PP = data_PP["P"]
    D_PP = data_PP["D"]
    time_PD_PP = data_PP["time"]

    # Load SS data
    data_SS = np.load(npz_SS)
    P_SS = data_SS["P"]
    D_SS = data_SS["D"]
    time_PD_SS = data_SS["time"]

    # Time vector for Gs
    Ntime = int(prior.tlen / prior.dt) * 2 - 1
    time = np.linspace(-prior.tlen, prior.tlen, Ntime, endpoint=False)

    # Prepare containers
    G_PP_list, G_SS_list = [], []
    D_pred_PP_list, D_pred_SS_list = [], []

    for model in ensemble_all:       
        # PP
        model_PP = Model(Nphase=model.Nphase, loc=model.locPP, amp=model.ampPP, wid=model.widPP)
        G_PP = create_G_from_model(model_PP, prior)
        # SS
        model_SS = Model(Nphase=model.Nphase, loc=model.locPP*model.rho, amp=model.ampSS, wid=model.widSS)
        G_SS = create_G_from_model(model_SS, prior)
        # Append
        G_PP_list.append(G_PP)
        G_SS_list.append(G_SS)
        D_pred_PP_list.append(convolve_P_G(P_PP, G_PP))
        D_pred_SS_list.append(convolve_P_G(P_SS, G_SS))

    # Stack and average
    G_PP_array = np.vstack(G_PP_list)
    G_SS_array = np.vstack(G_SS_list)
    D_PP_array = np.vstack(D_pred_PP_list)
    D_SS_array = np.vstack(D_pred_SS_list)

    G_PP_mean = np.mean(G_PP_array, axis=0)
    G_SS_mean = np.mean(G_SS_array, axis=0)
    D_PP_mean = np.mean(D_PP_array, axis=0)
    D_SS_mean = np.mean(D_SS_array, axis=0)

    # --- Plotting ---
    fig, axs = plt.subplots(4, 1, figsize=(8, 12), sharex=False)

    # 1. G_PP as 2D histogram
    plot_G_2dhist(axs[0], G_PP_array, time, "PP: Ensemble of G Models (2D Histogram)")

    # 2. D_PP predictions
    axs[1].set_title("PP: Predicted D vs Observed D")
    axs[1].plot(time_PD_PP, D_PP_array.T, color="gray", alpha=0.2, linewidth=0.8)
    axs[1].plot(time_PD_PP, D_PP_mean, color="black", linewidth=1.5, label="Predicted Mean")
    axs[1].plot(time_PD_PP, D_PP, linestyle="--", color="red", linewidth=1.2, label="Observed D")
    axs[1].set_ylabel("Amplitude")
    axs[1].grid(True)
    axs[1].legend()

    # # 3. Rho distribution vs. locPP
    # plot_rho_distribution(axs[2], ensemble_all, [time[0], time[-1]])

    # 4. G_SS as 2D histogram
    plot_G_2dhist(axs[2], G_SS_array, time, "SS: Ensemble of G Models (2D Histogram)")

    # 5. D_SS predictions
    axs[3].set_title("SS: Predicted D vs Observed D")
    axs[3].plot(time_PD_SS, D_SS_array.T, color="gray", alpha=0.2, linewidth=0.8)
    axs[3].plot(time_PD_SS, D_SS_mean, color="black", linewidth=1.5, label="Predicted Mean")
    axs[3].plot(time_PD_SS, D_SS, linestyle="--", color="red", linewidth=1.2, label="Observed D")
    axs[3].set_ylabel("Amplitude")
    axs[3].set_xlabel("Time (s)")
    axs[3].grid(True)
    axs[3].legend()

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()