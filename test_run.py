import os, pickle
import numpy as np
from sharp_ss.rjmcmc import rjmcmc_run
from sharp_ss.model import Bookkeeping

datadir = "./synthetic_0/"


data = np.load(os.path.join(datadir, "synthetic_0.npz"))
P = data["P"]
D = data["D"]
G = data["G"]

with open(os.path.join(datadir, "prior.pkl"), "rb") as f:
    prior = pickle.load(f)

with open(os.path.join(datadir, "model.pkl"), "rb") as f:
    model = pickle.load(f)

bookkeeping = Bookkeeping()

saveDir = "./synthetic_0/"

ensemble, logL_trace = rjmcmc_run(P, D, prior, bookkeeping, saveDir)