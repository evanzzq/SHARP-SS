import copy, time, os, datetime
import numpy as np
import matplotlib.pyplot as plt
from sharp_ss.utils import generate_arr
from sharp_ss.model import Model, Prior
from sharp_ss.forward import create_G_from_model, calc_like_prob

def birth(model, prior):
    model_new = copy.deepcopy(model)
    if model_new.Nphase < prior.maxN:
        model_new.Nphase += 1
        model_new.loc = np.append(
            model_new.loc, generate_arr(np.array([prior.minSpace, prior.tlen]), model_new.loc, prior.minSpace)
        )
        model_new.amp = np.append(model_new.amp, np.random.uniform(prior.ampRange[0], prior.ampRange[1]))
        model_new.wid = np.append(model_new.wid, np.random.uniform(prior.widRange[0], prior.widRange[1]))
    else:
        success = False
    return model_new, success

def death(model, prior):
    model_new = copy.deepcopy(model)
    if model_new.Nphase > 0:
        model_new.Nphase -= 1
        idx = np.random.randint(model_new.Nphase)
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
    # Select a phase and update
    idx = np.random.randint(model_new.Nphase)
    model_new.loc[idx] += prior.locStd * np.random.randn()
    # Check range
    if not (prior.minSpace <= model_new.loc[idx] <= prior.tlen):
        return model, False
    # Check spacing with other phases
    loc_others = np.delete(model_new.loc, idx)
    if np.any(np.abs(loc_others - model_new.loc[idx]) < prior.minSpace):
        return model, False
    # Success, return
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
    if not (prior.ampRange[0] <= model_new.loc[idx] <= prior.ampRange[1]):
        return model, False
    # Success, return
    return model_new, True

def update_wid(model, prior):
    # Birth if no phase in model
    if model.Nphase == 0:
        return birth(model, prior)
    # Copy model
    model_new = copy.deepcopy(model)
    # Select a phase and update
    idx = np.random.randint(model_new.Nphase)
    model_new.wid[idx] += prior.widStd * np.random.randn()
    # Check range
    if not (prior.widRange[0] <= model_new.loc[idx] <= prior.widRange[1]):
        return model, False
    # Success, return
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

def choose_actions(actionsPerStep, fitNoise, ncProb=0.025):
    
    actionPool = [0, 1, 2, 3, 4]
    if fitNoise: actionPool.extend([5, 6])
    actionPool = np.array(actionPool)

    if 6 in actionPool:
        base_actions = actionPool[actionPool != 6]
        base_weight = (1-ncProb) / len(base_actions)
        weights = np.array([ncProb if a == 5 else base_weight for a in actionPool])
    else:
        weights = np.full(len(actionPool), 1.0 / len(actionPool))

    return np.random.choice(actionPool, size=actionsPerStep, replace=True, p=weights)

def rjmcmc_run(P, D, prior, bookkeeping, saveDir):
    
    totalSteps = bookkeeping.totalSteps
    burnInSteps = bookkeeping.burnInSteps
    nSaveModels = bookkeeping.nSaveModels
    save_interval = (totalSteps - burnInSteps) // nSaveModels
    actionsPerStep = bookkeeping.actionsPerStep
    fitNoise = bookkeeping.fitNoise

    # Start from an empty model
    model = Model.create_empty(prior=prior)
    changed_corr = True  # need to compute correlation matrix at first step
    R_LT = R_UT = R_P = LogDetR = None

    # Initial likelihood
    logL, R_LT, R_UT, R_P, LogDetR = calc_like_prob(
        P, D, model, prior, CdInv_opt=True
    )

    start_time = time.time()
    checkpoint_interval = totalSteps // 100

    ensemble = []
    logL_trace = []

    for iStep in range(totalSteps):

        actions = choose_actions(fitNoise, actionsPerStep)
        model_new = model

        changed_corr = False

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
            elif action == 5:
                model_new, _ = update_sig(model_new, prior)
            elif action == 6:
                model_new, nc_success = update_nc(model_new, prior)
        
        changed_corr &= nc_success

        # Forward model
        G_new = create_G_from_model(model_new, prior)

        # Compute likelihood
        if changed_corr:
            new_logL, new_R_LT, new_R_UT, new_R_P, new_LogDetR = \
                calc_like_prob(P, D, G_new, model_new, prior, CdInv_opt=True)
        else:
            new_logL, _, _, _, _ = \
                calc_like_prob(P, D, G_new, model_new, prior, CdInv_opt=False,
                               R_LT=R_LT, R_UT=R_UT, R_P=R_P, LogDetR=LogDetR)

        # Acceptance probability
        log_accept_ratio = new_logL - logL

        if np.log(np.random.rand()) < log_accept_ratio:
            model = model_new
            logL = new_logL
            if changed_corr:
                R_LT, R_UT, R_P, LogDetR = new_R_LT, new_R_UT, new_R_P, new_LogDetR
        
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

        changed_corr = False  # reset after each iteration

    return ensemble, logL_trace

    