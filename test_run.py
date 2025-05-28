import os, pickle
import numpy as np
from sharp_ss.rjmcmc import rjmcmc_run
from sharp_ss.model import Bookkeeping
from sharp_ss.forward import create_G_from_model
import matplotlib.pyplot as plt

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

# Save emsemble
with open("./synthetic_0/ensemble.pkl", "wb") as f:
    pickle.dump(ensemble, f)

# Visualize
plt.figure(figsize=(10, 4))
for model in ensemble:
    G = create_G_from_model(model, prior)
    plt.plot(G)
plt.show()