
# variable names
target = 'IsSignal'
mass = 'Muons_Minv_MuMu'
weight = 'GlobalWeight'

# channels and datasets
jet0 = 'jet0'
jet1 = 'jet1'
jet2 = 'jet2'
channels = [jet0, jet1, jet2]
sig = 'sig'
data = 'data'
ss = 'ss'
datasets = [sig, data, ss]

n_classes = 2 # sig vs bkg, could be 3 for ggf, vbf, bkg

# plotting
bins = 100
mlow = 110
mhigh = 160

# misc
XYZW = ['X', 'Y', 'Z', 'W']
