import copy, time, os, datetime
import numpy as np
import matplotlib.pyplot as plt
from sharp_ss.utils import generate_arr
from sharp_ss.model import Model, Model2
from sharp_ss.forward import create_G_from_model, convolve_P_G, align_D

def calc_like_prob(P, D, model, prior, sigma=None, CDinv=None):
    """
    Calculate likelihood probability and associated matrices.
    
    Args:
        P: (n,) array
        D: (n, m) array, where n is the number of sample points and m is the number of traces
        model: object with model parameters
        prior: object with prior settings
        
    Returns:
        logL
    """
    if sigma is None: sigma = prior.stdP

    # Forward calculation: D = P * G
    G = create_G_from_model(model, prior)
    D_model = convolve_P_G(P, G)

    # Align to max amplitude if preferred (determined by prior.align - send in sample points)
    if prior.align:
        D_model, D = align_D(D_model, D, prior.align/prior.dt)

    if D.ndim == 1:
        D = D[:, np.newaxis]  # (npts, 1)
    if D_model.ndim == 1:
        D_model = D_model[:, np.newaxis]  # (npts, 1)

    # Calculate Diff without expanding D_model
    Diff = (D_model - D)  # (npts, ntraces)

    # Only take negative side if negOnly
    if prior.negOnly:
        Diff = Diff[:len(Diff) // 2, :]
    
    # Compute log likelihood
    if CDinv is None:
        logL = -0.5 * np.sum((Diff / sigma) ** 2)
    else:
        logL = -0.5 * np.trace(Diff.T @ CDinv @ Diff)

    return logL

def birth(model, prior):
    model_new = copy.deepcopy(model)
    if model_new.Nphase < prior.maxN:
        model_new.Nphase += 1
        new_loc, new_wid = generate_arr(
                np.array([prior.minSpace, prior.tlen]), model_new.loc, model_new.wid, prior.minSpace, prior.widRange
                )
        model_new.loc = np.append(model_new.loc, new_loc)
        model_new.wid = np.append(model_new.wid, new_wid)
        model_new.amp = np.append(model_new.amp, np.random.uniform(prior.ampRange[0], prior.ampRange[1]))
        success = True
    else:
        success = False
    return model_new, success

def death(model, prior):
    model_new = copy.deepcopy(model)
    if model_new.Nphase > 0:
        idx = np.random.randint(model_new.Nphase)
        model_new.Nphase -= 1
        model_new.loc = np.delete(model_new.loc, idx)
        model_new.amp = np.delete(model_new.amp, idx)
        model_new.wid = np.delete(model_new.wid, idx)
        success = True
    else:
        success = False
    return model_new, success

def update_loc(model, prior):
    # Birth if no phase in model
    if model.Nphase == 0:
        return birth(model, prior)
    # Copy model
    model_new = copy.deepcopy(model)
    # Select a phase to update
    idx = np.random.randint(model_new.Nphase)
    # Propose a new location
    model_new.loc[idx] += prior.locStd * np.random.randn()
    # Check if new location is in allowed range
    if not (prior.minSpace <= model_new.loc[idx] <= prior.tlen):
        return model, False
    # Get other locations and widths
    loc_others = np.delete(model_new.loc, idx)
    wid_others = np.delete(model_new.wid, idx)
    loc_new = model_new.loc[idx]
    wid_new = model_new.wid[idx]
    # Check for spacing and overlap with other phases
    dist = np.abs(loc_others - loc_new)
    min_dists = np.maximum(prior.minSpace, (wid_others + wid_new) / 2)
    if np.any(dist < min_dists):
        return model, False
    # Success
    return model_new, True

def update_amp(model, prior):
    # Birth if no phase in model
    if model.Nphase == 0:
        return birth(model, prior)
    # Copy model
    model_new = copy.deepcopy(model)
    # Select a phase and update
    idx = np.random.randint(model_new.Nphase)
    model_new.amp[idx] += prior.ampStd * np.random.randn()
    # Check range
    if not (prior.ampRange[0] <= model_new.amp[idx] <= prior.ampRange[1]):
        return model, False
    # Success, return
    return model_new, True

def update_wid(model, prior):
    # Birth if no phase in model
    if model.Nphase == 0:
        return birth(model, prior)
    # Copy model
    model_new = copy.deepcopy(model)
    # Select a phase to update
    idx = np.random.randint(model_new.Nphase)
    # Propose a new width
    model_new.wid[idx] += prior.widStd * np.random.randn()
    # Check if new width is within allowed range
    wid_new = model_new.wid[idx]
    if not (prior.widRange[0] <= wid_new <= prior.widRange[1]):
        return model, False
    # Check overlap with other phases
    loc_new = model_new.loc[idx]
    loc_others = np.delete(model_new.loc, idx)
    wid_others = np.delete(model_new.wid, idx)
    if loc_others.size > 0:
        dist = np.abs(loc_others - loc_new)
        overlap_thresh = (wid_others + wid_new) / 2
        if np.any(dist < np.maximum(prior.minSpace, overlap_thresh)):
            return model, False
    # Success
    return model_new, True

def update_sig(model, prior):
    # Copy model
    model_new = copy.deepcopy(model)
    # Update
    eps = np.finfo(float).eps
    model_new.sig = np.maximum(eps, model_new.sig + prior.sigStd * np.random.randn() * prior.stdP)
    # Return
    return model_new, True

def update_nc(model, prior):
    # Copy model
    model_new = copy.deepcopy(model)
    # Update
    eps = np.finfo(float).eps
    model_new.nc1 += prior.nc1Std * np.random.randn()
    model_new.nc1 = np.clip(model_new.nc1, eps, 1)
    model_new.nc2 += prior.nc2Std * np.random.randn()
    # Return
    return model_new, True

def rjmcmc_run(P, D, prior, bookkeeping, saveDir, CDinv=None):
    
    totalSteps = bookkeeping.totalSteps
    burnInSteps = bookkeeping.burnInSteps
    nSaveModels = bookkeeping.nSaveModels
    save_interval = (totalSteps - burnInSteps) // nSaveModels
    actionsPerStep = bookkeeping.actionsPerStep

    # Start from an empty model
    model = Model.create_empty(prior=prior)

    # Initial likelihood
    logL = calc_like_prob(P, D, model, prior, CDinv=CDinv)

    start_time = time.time()
    checkpoint_interval = totalSteps // 100

    ensemble = []
    logL_trace = []

    for iStep in range(totalSteps):

        actions = np.random.choice(5, size=actionsPerStep)
        model_new = model

        for action in actions:
            if action == 0:
                model_new, _ = birth(model_new, prior)
            elif action == 1:
                model_new, _ = death(model_new, prior)
            elif action == 2:
                model_new, _ = update_loc(model_new, prior)
            elif action == 3:
                model_new, _ = update_amp(model_new, prior)
            elif action == 4:
                model_new, _ = update_wid(model_new, prior)

        # Compute likelihood
        new_logL = calc_like_prob(P, D, model_new, prior, CDinv=CDinv)

        # Acceptance probability
        log_accept_ratio = new_logL - logL

        if np.log(np.random.rand()) < log_accept_ratio:
            model = model_new
            logL = new_logL
        
        logL_trace.append(logL)

        # Save only selected models after burn-in
        if iStep >= burnInSteps and (iStep - burnInSteps) % save_interval == 0:
            ensemble.append(model)
        
        # Checkpoint log/plot every 1%
        if (iStep + 1) % checkpoint_interval == 0:
            # Save (overwrite) log-likelihood plot
            fig, ax = plt.subplots()
            ax.plot(logL_trace, 'k-')
            ax.set_xlabel("Step")
            ax.set_ylabel("log Likelihood")
            fig.tight_layout()
            fig.savefig(os.path.join(saveDir, "logL.png"))  # overwrites each time
            plt.close(fig)

            # Overwrite progress log
            elapsed = time.time() - start_time
            now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(os.path.join(saveDir, "progress_log.txt"), "a") as f:
                f.write(f"[{now}] Step {iStep+1}/{totalSteps}, Elapsed: {elapsed:.2f} sec\n")

    # Save final log-likelihood trace
    logL_file = os.path.join(saveDir, "logL.txt")
    np.savetxt(logL_file, logL_trace, fmt="%.8f")

    return ensemble, logL_trace