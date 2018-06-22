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

SMpred = P5p_anomaly.P5p_binned() #Central value prediction (!!Run only once then comment)


# Create an environment
env = Environment(overwrite_file=True)
# Get the trajectory from the environment
traj = env.traj
# Add parameters
traj.f_add_parameter('m_b',  1., comment = 'First dimension')
traj.f_add_parameter('m_c',  1., comment = 'Second dimension')
traj.f_add_parameter('alpha_s_b',  1., comment = 'Third dimension')

traj.f_explore(cartesian_product ({'m_b' : [4.1, 4.3 ],
                                   'm_c' : [1.27, 1.33],
                                   'alpha_s_b': [0.192, 0.235]
                                   }))


# Define the observable in the par. space
def scan(traj):
    P5p_anomaly.m_b=traj.m_b
    P5p_anomaly.m_c=traj.m_c
    P5p_anomaly.alpha_s_b=traj.alpha_s_b
    return P5p_anomaly.P5p_binned()

Result=env.run(scan)

# Find the maximum and minimum value for each bin
def FindMaxMin():
    res=[]
    Max_values=[]
    Min_values=[]
    for j in range(len(Result[0][1])):
        for i in range(len(Result)):
            res.append(Result[i][1][j])
        Max_values.append(max(res))
        Min_values.append(min(res))
        res=[]
    return(Max_values, Min_values)

#print('Max Values: ',  FindMaxMin()[0], '\n', 'Min values: ',  FindMaxMin()[1])


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

