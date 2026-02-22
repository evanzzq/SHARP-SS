import copy, time, os, datetime
import numpy as np
import matplotlib.pyplot as plt
from sharp_ss.utils import generate_arr_PP_SS_mars
from sharp_ss.model import Model, Model2
from sharp_ss.forward import create_G_from_model, convolve_P_G, align_D

def calc_like_prob_PP_SS_mars(P_PP, P_SS, D_PP, D_SS, model, prior, sigma=None, CDinv_PP=None, CDinv_SS=None):
    """
    Calculate likelihood probability using both PP and SS with an assumed vp/vs value.
    
    Args:
        P: (n,) array
        D_PP and D_SS: (n,) array, where n is the number of sample points
        model: object with model parameters, timing is for PP (t_SS = t_PP * (vp/vs))
        prior: object with prior settings
        
    Returns:
        logL
    """

    # Forward calculation: D = P * G
    # PP
    model_PP = Model(Nphase=model.Nphase, loc=model.locPP, amp=model.ampPP, wid=model.widPP, loge=0.)
    G_PP = create_G_from_model(model_PP, prior)
    D_PP_model = convolve_P_G(P_PP, G_PP)
    # SS
    #### temp: hard-coded rayp and vs ####
    vs = 3.5 # km/s

    # S0976a
    # Li2022: 0.08181493934, 0.1712366396
    # Syn TauP SS @1877: 0.0862, 0.1375
    # Syn TauP SS @1883: 0.0862, 0.1662/0.1715/0.1723
    # Syn TauP SS @1888: 0.0862, 0.1527
    rayp_PP, rayp_SS = 0.0862, 0.1527
    
    loc_SS = model.locPP * model.rho * np.sqrt(1 - (rayp_PP * vs * model.rho)**2) / np.sqrt(1 - (rayp_SS * vs)**2)
    model_SS = Model(Nphase=model.Nphase, loc=loc_SS, amp=model.ampSS, wid=model.widSS, loge=0.)
    G_SS = create_G_from_model(model_SS, prior)
    D_SS_model = convolve_P_G(P_SS, G_SS)

    # Align to max amplitude if preferred (determined by prior.align - send in sample points)
    if prior.align:
        D_PP_model, D_PP = align_D(D_PP_model, D_PP, prior.align/prior.dt)
        D_SS_model, D_SS = align_D(D_SS_model, D_SS, prior.align/prior.dt)

    # If PP/SS length doesn't match, correct to the shorter one
    if len(D_PP) != len(D_SS):
        target_len = min(len(D_PP), len(D_SS))
        if len(D_PP) > len(D_SS): 
            D_PP = D_PP[(len(D_PP)-target_len)//2:(len(D_PP)-target_len)//2+target_len]
            D_PP_model = D_PP_model[(len(D_PP)-target_len)//2:(len(D_PP)-target_len)//2+target_len]
        if len(D_PP) < len(D_SS): 
            D_SS = D_SS[(len(D_PP)-target_len)//2:(len(D_PP)-target_len)//2+target_len]
            D_SS_model = D_SS_model[(len(D_PP)-target_len)//2:(len(D_PP)-target_len)//2+target_len]
    
    if D_PP.ndim == 1:
        D_PP = D_PP[:, np.newaxis]  # (npts, 1)
    if D_PP_model.ndim == 1:
        D_PP_model = D_PP_model[:, np.newaxis]  # (npts, 1)
    if D_SS.ndim == 1:
        D_SS = D_SS[:, np.newaxis]  # (npts, 1)
    if D_SS_model.ndim == 1:
        D_SS_model = D_SS_model[:, np.newaxis]  # (npts, 1)

    # Calculate Diff
    Diff_PP = D_PP_model - D_PP
    Diff_SS = D_SS_model - D_SS

    # Only take negative side if negOnly
    if prior.negOnly:
        Diff_PP = Diff_PP[:len(Diff_PP) // 2, :]
        Diff_SS = Diff_SS[:len(Diff_SS) // 2, :]
    
    # Compute log likelihood
    if CDinv_PP is None or CDinv_SS is None:
        sigma_PP, sigma_SS = prior.std1 * np.exp(0.5 * model.loge1), prior.std2 * np.exp(0.5 * model.loge2)
        logL = -0.5 * np.sum((Diff_PP / sigma_PP) ** 2) - 0.5 * np.sum((Diff_SS / sigma_SS) ** 2)
    else:
        CDinv_PP, CDinv_SS = CDinv_PP * np.exp(-model.loge1), CDinv_SS * np.exp(-model.loge2)
        logL = -0.5 * (np.trace(Diff_PP.T @ CDinv_PP @ Diff_PP) + np.trace(Diff_SS.T @ CDinv_SS @ Diff_SS))

    return logL

def birth(model, prior):
    model_new = copy.deepcopy(model)
    if model_new.Nphase < prior.maxN:
        model_new.Nphase += 1
        new_locPP, new_widPP, new_widSS, new_rho = generate_arr_PP_SS_mars(
                np.array([prior.minSpace, prior.tlen]), prior.rhoRange, model_new.locPP, model_new.widPP, model_new.widSS, model_new.rho, prior.minSpace, prior.widRange
                )
        model_new.locPP = np.append(model_new.locPP, new_locPP)
        model_new.widPP = np.append(model_new.widPP, new_widPP)
        model_new.widSS = np.append(model_new.widSS, new_widSS)
        model_new.ampPP = np.append(model_new.ampPP, np.random.uniform(prior.ampRange[0], prior.ampRange[1]))
        model_new.ampSS = np.append(model_new.ampSS, np.random.uniform(prior.ampRange[0], prior.ampRange[1]))
        model_new.rho   = np.append(model_new.rho,   new_rho)
        success = True
    else:
        success = False
    return model_new, success

def death(model, prior):
    model_new = copy.deepcopy(model)
    if model_new.Nphase > 0:
        idx = np.random.randint(model_new.Nphase)
        model_new.Nphase -= 1
        model_new.locPP = np.delete(model_new.locPP, idx)
        model_new.ampPP = np.delete(model_new.ampPP, idx)
        model_new.widPP = np.delete(model_new.widPP, idx)
        model_new.ampSS = np.delete(model_new.ampSS, idx)
        model_new.widSS = np.delete(model_new.widSS, idx)
        model_new.rho   = np.delete(model_new.rho,   idx)
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
    model_new.locPP[idx] += prior.locStd * np.random.randn()
    # Check if new location is in allowed range
    if not (prior.minSpace <= model_new.locPP[idx] <= prior.tlen):
        return model, False
    # Get other locations and widths
    locPP_others = np.delete(model_new.locPP, idx)
    widPP_others = np.delete(model_new.widPP, idx)
    widSS_others = np.delete(model_new.widSS, idx)
    rho_others   = np.delete(model_new.rho,   idx)
    locPP_new = model_new.locPP[idx]
    widPP_new = model_new.widPP[idx]
    widSS_new = model_new.widSS[idx]
    # Check for spacing and overlap with other phases
    distPP = np.abs(locPP_others - locPP_new)
    distSS = distPP * rho_others
    min_distsPP = np.maximum(prior.minSpace, (widPP_others + widPP_new) / 2)
    min_distsSS = np.maximum(prior.minSpace, (widSS_others + widSS_new) / 2)
    if np.any(distPP < min_distsPP) or np.any(distSS < min_distsSS):
        return model, False
    # Success
    return model_new, True

def update_ampPP(model, prior):
    # Birth if no phase in model
    if model.Nphase == 0:
        return birth(model, prior)
    # Copy model
    model_new = copy.deepcopy(model)
    # Select a phase and update
    idx = np.random.randint(model_new.Nphase)
    model_new.ampPP[idx] += prior.ampStd * np.random.randn()
    # Check range
    if not (prior.ampRange[0] <= model_new.ampPP[idx] <= prior.ampRange[1]):
        return model, False
    # Success, return
    return model_new, True

def update_ampSS(model, prior):
    # Birth if no phase in model
    if model.Nphase == 0:
        return birth(model, prior)
    # Copy model
    model_new = copy.deepcopy(model)
    # Select a phase and update
    idx = np.random.randint(model_new.Nphase)
    model_new.ampSS[idx] += prior.ampStd * np.random.randn()
    # Check range
    if not (prior.ampRange[0] <= model_new.ampSS[idx] <= prior.ampRange[1]):
        return model, False
    # Success, return
    return model_new, True

def update_widPP(model, prior):
    # Birth if no phase in model
    if model.Nphase == 0:
        return birth(model, prior)
    # Copy model
    model_new = copy.deepcopy(model)
    # Select a phase to update
    idx = np.random.randint(model_new.Nphase)
    # Propose a new width
    model_new.widPP[idx] += prior.widStd * np.random.randn()
    # Check if new width is within allowed range
    wid_new = model_new.widPP[idx]
    if not (prior.widRange[0] <= wid_new <= prior.widRange[1]):
        return model, False
    # Check overlap with other phases
    loc_new = model_new.locPP[idx]
    loc_others = np.delete(model_new.locPP, idx)
    wid_others = np.delete(model_new.widPP, idx)
    if loc_others.size > 0:
        dist = np.abs(loc_others - loc_new)
        overlap_thresh = (wid_others + wid_new) / 2
        if np.any(dist < np.maximum(prior.minSpace, overlap_thresh)):
            return model, False
    # Success
    return model_new, True

def update_widSS(model, prior):
    # Birth if no phase in model
    if model.Nphase == 0:
        return birth(model, prior)
    # Copy model
    model_new = copy.deepcopy(model)
    # Select a phase to update
    idx = np.random.randint(model_new.Nphase)
    # Propose a new width
    model_new.widSS[idx] += prior.widStd * np.random.randn()
    # Check if new width is within allowed range
    wid_new = model_new.widSS[idx]
    if not (prior.widRange[0] <= wid_new <= prior.widRange[1]):
        return model, False
    # Check overlap with other phases
    loc_new = model_new.locPP[idx] * model_new.rho[idx]
    loc_others = np.delete(model_new.locPP*model_new.rho, idx)
    wid_others = np.delete(model_new.widSS, idx)
    if loc_others.size > 0:
        dist = np.abs(loc_others - loc_new)
        overlap_thresh = (wid_others + wid_new) / 2
        if np.any(dist < np.maximum(prior.minSpace, overlap_thresh)):
            return model, False
    # Success
    return model_new, True

def update_rho(model, prior):
    # Birth if no phase in model
    if model.Nphase == 0:
        return birth(model, prior)
    # Copy model
    model_new = copy.deepcopy(model)
    # Select a phase to update
    idx = np.random.randint(model_new.Nphase)
    # Propose a new rho
    model_new.rho[idx] += prior.rhoStd * np.random.randn()
    # Check range
    if not (prior.rhoRange[0] <= model_new.rho[idx] <= prior.rhoRange[1]):
        return model, False
    # Success, return
    return model_new, True

def update_loge1(model, prior):
    # Copy model
    model_new = copy.deepcopy(model)
    # Update
    model_new.loge1 += prior.logeStd * np.random.randn()
    # Check range and return
    if prior.logeRange[0] <= model_new.loge1 <= prior.logeRange[1]:
        return model_new, True
    return model, False

def update_loge2(model, prior):
    # Copy model
    model_new = copy.deepcopy(model)
    # Update
    model_new.loge2 += prior.logeStd * np.random.randn()
    # Check range and return
    if prior.logeRange[0] <= model_new.loge2 <= prior.logeRange[1]:
        return model_new, True
    return model, False

def rjmcmc_run_PP_SS_mars(P_PP, P_SS, D_PP, D_SS, prior, bookkeeping, saveDir, CDinv_PP=None, CDinv_SS=None):
    
    totalSteps = bookkeeping.totalSteps
    burnInSteps = bookkeeping.burnInSteps
    nSaveModels = bookkeeping.nSaveModels
    save_interval = (totalSteps - burnInSteps) // nSaveModels
    actionsPerStep = bookkeeping.actionsPerStep

    n_len = P_PP.shape[0]
    if prior.negOnly: n_len //= 2

    # Start from an empty model
    model = Model2.create_empty()

    # Initial likelihood
    logL = calc_like_prob_PP_SS_mars(
        P_PP, P_SS, D_PP, D_SS, model, prior, CDinv_PP=CDinv_PP, CDinv_SS=CDinv_SS
        )

    start_time = time.time()
    checkpoint_interval = totalSteps // 100

    ensemble = []
    logL_trace = []

    for iStep in range(totalSteps):

        actions = np.random.choice(10, size=actionsPerStep)
        model_new = model

        for action in actions:
            if action == 0:
                model_new, _ = birth(model_new, prior)
            elif action == 1:
                model_new, _ = death(model_new, prior)
            elif action == 2:
                model_new, _ = update_loc(model_new, prior)
            elif action == 3:
                model_new, _ = update_ampPP(model_new, prior)
            elif action == 4:
                model_new, _ = update_ampSS(model_new, prior)
            elif action == 5:
                model_new, _ = update_widPP(model_new, prior)
            elif action == 6:
                model_new, _ = update_widSS(model_new, prior)
            elif action == 7:
                model_new, _ = update_rho(model_new, prior)
            elif action == 8:
                model_new, _ = update_loge1(model_new, prior)
            elif action == 9:
                model_new, _ = update_loge2(model_new, prior)

        # Compute likelihood
        new_logL = calc_like_prob_PP_SS_mars(
            P_PP, P_SS, D_PP, D_SS, model_new, prior, CDinv_PP=CDinv_PP, CDinv_SS=CDinv_SS
            )

        # Acceptance probability
        log_accept_ratio = (new_logL - logL) + n_len * ((model.loge1 - model_new.loge1) + (model.loge2 - model_new.loge2))

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

    return ensemble, logL_trace