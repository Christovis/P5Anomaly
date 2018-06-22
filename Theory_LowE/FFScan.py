import numpy as np
from numpy import ma
import matplotlib.pyplot as plt
from collections import OrderedDict
import matplotlib.patches as patches
import scipy.integrate as integrate
import flavio
import flavio.plots as fpl
from matplotlib import rc
from pypet import Environment, cartesian_product
import P5p_anomaly

SMpred = P5p_anomaly.P5p_binned() #Central value prediction(!!Run only once, then comment!!)


# Create an environment
env = Environment(overwrite_file=True)
# Get the trajectory from the environment
traj = env.traj
# Add parameters
traj.f_add_parameter('ksi_ort',  1., comment = 'First dimension')
traj.f_add_parameter('ksi_par',  1., comment = 'Second dimension')
traj.f_explore(cartesian_product ({'ksi_ort' : [0.234, 0.298 ],
                                   'ksi_par' : [0.11, 0.126]}))

# Define the observable in the par. space
def scan(traj):
    P5p_anomaly.ksi_ort_0=traj.ksi_ort
    P5p_anomaly.ksi_par_0=traj.ksi_par
    return P5p_anomaly.P5p_binned()
    
# Find the maximum and minimum value for each bin

Result = env.run(scan)

def FindMaxMin():
    res = []
    Max_values = []
    Min_values = []
    for j in range(len(Result[0][1])):
        for i in range(len(Result)):
            res.append(Result[i][1][j])
        Max_values.append(max(res))
        Min_values.append(min(res))
        res=[]
    return(Max_values, Min_values)


# Find Error bars

def FindErrBar():
    MaxMin = FindMaxMin()
    bar_max = []
    bar_min = []
    for i in range(len(MaxMin[0])):
        bar_max.append( np.absolute(MaxMin[0][i] - SMpred[i]))
        bar_min.append( np.absolute(SMpred[i] - MaxMin[1][i]))
    return(bar_max, bar_min)

#for i in range(len(FindErrBar()[0])):
#    print('%i bin: ' %i, SMpred[i], '+', FindErrBar()[0][i], '-', FindErrBar()[1][i])
