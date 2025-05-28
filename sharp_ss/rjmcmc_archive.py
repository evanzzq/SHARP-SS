def calc_like_prob_archive(P, D, model, prior, CdInv_opt, R_LT=None, R_UT=None, R_P=None, LogDetR=None):
    """
    Calculate likelihood probability and associated matrices.
    
    Args:
        P: (n,) array
        D: (n, m) array, where n is the number of sample points and m is the number of traces
        model: object with model parameters
        prior: object with prior settings
        CdInv_opt: bool, whether to build CdInv
        R_LT, R_UT, R_P, LogDetR: optional, for efficiency
        
    Returns:
        logL, D_model, R_LT, R_UT, R_P, LogDetR
    """

    from scipy.linalg import toeplitz, lu, cholesky, solve_triangular
    
    # Forward calculation: D = P * G
    G = create_G_from_model(model, prior)
    D_model = convolve_P_G(P, G)

    # Align to max amplitude if preferred (determined by prior.align - send in sample points)
    if prior.align:
        D_model, D = align_D(D_model, D, prior.align/prior.dt)

    if CdInv_opt:
        Rrow = np.zeros(len(D))
        if prior.negOnly:
            Rrow = Rrow[:len(Rrow) // 2]

        for i in range(len(Rrow)):
            t = (i) * prior.dt
            Rrow[i] = np.exp(-model.nc1 * t) * np.cos(model.nc2 * np.pi * model.nc1 * t)
        
        R = toeplitz(Rrow)
        
        # Cholesky for logdet
        L = cholesky(R, lower=True)
        LogDetR = 2 * np.sum(np.log(np.diag(L)))

        # LU decomposition
        R_P, R_LT, R_UT = lu(R)

    if D.ndim == 1:
        D = D[:, np.newaxis]  # (npts, 1)
    if D_model.ndim == 1:
        D_model = D_model[:, np.newaxis]  # (npts, 1)

    # Calculate Diff without expanding D_model
    Diff = (D_model - D)  # (npts, ntraces)

    if prior.negOnly:
        Diff = Diff[:len(Diff) // 2, :]

    # Solve triangular systems
    y = solve_triangular(R_LT, R_P @ Diff, lower=True)
    y = solve_triangular(R_UT, y, lower=False)

    # Mahalanobis distance per trace
    MahalDist = np.sum(Diff * y, axis=0) / (model.sig**2)

    # Log determinant
    LogCdDeterm = Diff.shape[0] * np.log(model.sig) + 0.5 * LogDetR

    # Total logL
    logL = np.sum(-LogCdDeterm - MahalDist / 2)

    return logL, R_LT, R_UT, R_P, LogDetR

def rjmcmc_run_archive(P, D, prior, bookkeeping, saveDir):
    
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

def choose_actions_archive(actionsPerStep, fitNoise, ncProb=0.025):
    
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