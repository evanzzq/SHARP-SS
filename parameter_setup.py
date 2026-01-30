# ---- Parameter setup ----
filedir = "H:\My Drive\Res÷earch\SharpSSPy"
# filedir = "/Users/evanzh÷ang/zzq@umd.edu - Google Drive/My Drive/Research/SharpSSPy"

event_name = "S0976a_src3s"
data_type = "joint" # "PP", "SS", "joint", "syn", "synDL"
useCD = True

PPdir = event_name+"_PP"
SSdir = event_name+"_SS"
syndir = event_name

# For "synDL" type only
Pfile = "misc/ocean_stack.csv" # relative to filedir
DLmod = [16, 8, 5, 5] # patch size (l1/l2), patch size (s1/s2), no. of layer, epochs

modname = event_name+"_"+data_type
runname = "run2"

ampRange = (0., 0.5)
widRange = (0.1, 2.)
negOnly = True
align = False
maxN = 10
stdP = 0.1

totalSteps     = int(1e6)
burnInSteps    = int(8e5)
nSaveModels    = 500
actionsPerStep = 2

num_chains = 96