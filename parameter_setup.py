# ---- Parameter setup ----
# filedir = "H:\My Drive\Research\SharpSSPy"
filedir = "/Users/evanzhang/zzq@umd.edu - Google Drive/My Drive/Research/SharpSSPy"

event_name = "S0976a"
data_type = "joint" # "PP", "SS", "joint", "syn"

PPdir = event_name+"_PP"
SSdir = event_name+"_SS"
syndir = event_name+"_syn"

modname = event_name+"_"+data_type
runname = "run2"

ampRange = (-0.3, 0.)
widRange = (0.1, 2.)
negOnly = True
align = 3.
maxN = 2
stdP = 0.1

totalSteps     = int(4e5)
burnInSteps    = None
nSaveModels    = 100
actionsPerStep = 2

num_chains = 8