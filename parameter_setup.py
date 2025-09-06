# ---- Parameter setup ----
# filedir = "H:\My Drive\Research\SharpSSPy"
filedir = "/Users/evanzhang/zzq@umd.edu - Google Drive/My Drive/Research/SharpSSPy"

event_name = "S1000a"
data_type = "PP" # "PP", "SS", "joint", "syn", "synDL"
useCD = True

PPdir = event_name+"_PP"
SSdir = event_name+"_SS"
syndir = event_name

# For "synDL" type only
Pfile = "misc/ocean_stack.csv" # relative to filedir
DLmod = [16, 8, 5, 5] # patch size (l1/l2), patch size (s1/s2), no. of layer, epochs

modname = event_name+"_"+data_type
runname = "run2"

ampRange = (-0.5, 0.5)
widRange = (0.1, 2.)
negOnly = True
align = False
maxN = 5
stdP = 0.1

totalSteps     = int(2e5)
burnInSteps    = int(1.5e5)
nSaveModels    = 100
actionsPerStep = 2

num_chains = 1