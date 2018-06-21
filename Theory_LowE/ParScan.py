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

#SM_res = P5p_anomaly.P5p_binned() #Central value prediction
#print(SM_res)


# Create an environment
env = Environment(overwrite_file=True)
# Get the trajectory from the environment
traj = env.traj
# Add parameters
traj.f_add_parameter('m_b',  1., comment = 'First dimension')
traj.f_add_parameter('m_c',  1., comment = 'Second dimension')

traj.f_explore(cartesian_product ({'m_b' : [4.1, 4.3 ],
                                   'm_c' : [1.27, 1.33],
                                   }))


# Define the observable in the par. space
def scan(traj):
    P5p_anomaly.m_b=traj.m_b
    P5p_anomaly.m_c=traj.m_c
    return P5p_anomaly.P5p_binned()

#Result=env.run(scan)

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

#print(len(Result[0][1]))

# Find Error bars

def FindErrBar():
    bar_max = []
    bar_min = []
    for i in range(len(FindMaxMin()[0])):
        bar_max.append( np.absolute(FindMaxMin()[0][i] - SM_res[i]))
        bar_min.append( np.absolute(SM_res[i] - FindMaxMin()[1][i]))
    return(bar_max, bar_min)

#for i in range(len(FindErrBar()[0])):
#   print('%i bin: ' %i, SM_res[i], '+', FindErrBar()[0][i], '-', FindErrBar()[1][i])


# Bin Plot with error bars
def BinPlot():
    rc('font',**{'family':'serif','serif':['Palatino']})
    rc('text', usetex=True)
    
    bins=[(0.1, 0.98),(1.1, 2.5), (2.5, 4.), (4., 6.0)]
    #bins.tolist() #needed for Flavio th-prediction
    bins=[tuple(entry) for entry in bins]
    
    
    Max_plt = np.array(FindMaxMin()[0])
    Max_plt = np.append(Max_plt, -1)
    Min_plt = np.array(FindMaxMin()[1])
    Min_plt = np.append(Min_plt, -1)
    
    ax=plt.gca()
    ax.set_xlim([0, 6.1])
    ax.set_ylim([-1, 1.])
    for i in range(len(bins)):
        label= 'SM-th'
        if i>0:
            label=None
        ax.add_patch(patches.Rectangle((bins[i][0], Min_plt[i]),
                                       bins[i][1]-bins[i][0],          # width
                                       Max_plt[i]-Min_plt[i],   # height
                                       ec='c', fill= False, lw=True, hatch= '///',
                                       label=label, capstyle= 'butt'))
            
            # Falvio experimental data
    measur=['LHCb B->K*mumu 2015 P 0.1-0.98',
            'LHCb B->K*mumu 2015 P 1.1-2.5',
            'LHCb B->K*mumu 2015 P 2.5-4',
            'LHCb B->K*mumu 2015 P 4-6']
    #'ATLAS B->K*mumu 2017 P5p' ]
    fpl.bin_plot_exp('<P5p>(B0->K*mumu)',
                     col_dict= {'ATLAS': 'c', 'LHCb': 'g' },  #'#EB70AA' for light pink
                     divide_binwidth=False,
                     include_measurements=measur)
    
    # Flavio theoretical prediction
    #fpl.bin_plot_th( '<P5p>(B0->K*mumu)', bins,
    #                 label='SM-th-Flavio', divide_binwidth=False,
    #                N=50,threads=2)
    
    plt.xlabel('$q^2 \hspace{2pt} (GeV^2)$')
    plt.ylabel('$P5\' \hspace{2pt} (q^2)$')
    plt.legend()
    plt.title('Parameters corrections')
    #plt.title('$P_5\'$ prediction with $ (\delta C_7, \delta C_9, \delta C_{10}) = (.1, .1, .1)$')
    plt.show()
    #plt.ylim(-1.2, 0.7)
    #plt.savefig('Fig1_NewP5.png', bbox_inches='tight')
    return(0)

#BinPlot()

