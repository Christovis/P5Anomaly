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

def scan(traj):
    P5p_anomaly.m_b=traj.m_b
    P5p_anomaly.m_c=traj.m_c
    P5p_anomaly.C2=traj.C2
    P5p_anomaly.C7_eff=traj.C7
    P5p_anomaly.C9=traj.C9
    P5p_anomaly.C10=traj.C10
    return P5p_anomaly.P5p_binned()

# create an environment
env = Environment()
# get the trajectory from the environment
traj = env.traj
# Add parameters
traj.f_add_parameter('m_b',  1., comment = 'First dimension')
traj.f_add_parameter('m_c',  1., comment = 'Second dimension')
traj.f_add_parameter('C2',  1., comment = 'Third dimension')
traj.f_add_parameter('C7',  1., comment = 'Fourth dimension')
traj.f_add_parameter('C9',  1., comment = 'Fifth dimension')
traj.f_add_parameter('C10',  1., comment = 'Sixt dimension')
traj.f_explore(cartesian_product ({'m_b' : [4.6, 4.8 , 5.],
                                   'm_c' : [1.3, 1.5, 1.6],
                                   'C2' : [1.106, 1.108, 1.09],
                                   'C7' : [-0.378, -0.365, -0.351],
                                   'C9' : [4.111, 4.334, 4.550],
                                   'C10' : [-4.321, -4.513, -4.666]}))

result=env.run(scan)
print(result[0][1][1])

#Max_values=[]
#Min_values=[]
res = []
#for i in range(len(result)):
 #   res.append(result[i][1][2]) # list of all second bin results
#Max = max(res)
#Min = min(res)
#Max_values.append(Max)
#Min_values.append(Min)
#print('Max value: ', Max_values, '\n' , 'Min value: ',  Min_values)



'''
   #res_plt_np = np.array(results_NP)
    #res_plt_np = np.append(res_plt_np, -1)
    
    
    
    axes = plt.gca()
    axes.set_xlim([0, 6.1])
    axes.set_ylim([-1.6, 1])
    plt.step( bins_lim, res_plt_sm,
              'c', where='post', label='SM')
    
    fpl.bin_plot_th( '<P5p>(B0->K*mumu)', bins,
                     label='SM-th-Flavio', divide_binwidth=False,
                     N=50,threads=2)
    
    plt.legend()
    plt.show()
'''

'''
###############################################################################
# BIN PLOT with ERROR BARS (coming only from FF)

rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)

bins=[(0.1, 2.0), (2.0, 4.3), (4.3, 6.0)]
#bins.tolist() #needed for Flavio th-prediction
bins=[tuple(entry) for entry in bins]
   

Max_plt = np.array(Max_values)
Max_plt = np.append(Max_plt, -1)
Min_plt = np.array(Min_values)
Min_plt = np.append(Min_plt, -1)
 
ax=plt.gca()
ax.set_xlim([0, 6.1])
ax.set_ylim([-1, 1.])
for i in range(len(bins)):
    label= 'SM'
    if i>0:
       label=None
    ax.add_patch(patches.Rectangle((bins[i][0], Min_plt[i]),
                                   bins[i][1]-bins[i][0],          # width
                                   Max_plt[i]-Min_plt[i],   # height
                                   ec='#EB70AA', fill= False, lw=True, hatch= 'o',
                                   label=label, capstyle= 'butt'))
'''
'''
for i in range(len(bins)):
    label= 'NP'
    if i>0:
       label=None
    ax.add_patch(patches.Rectangle((bins[i][0], res_plt_np[i]),
                                   bins[i][1]-bins[i][0],           # width
                                   (res_plt_np[i]+0.01)-res_plt_np[i],   # height
                                   ec='y', fill= False, lw=True, hatch = 'x',
                                   label=label, capstyle= 'butt'))
'''
# Falvio experimental data
measur=['LHCb B->K*mumu 2015 P 0.1-0.98',
        'LHCb B->K*mumu 2015 P 1.1-2.5',
        'LHCb B->K*mumu 2015 P 2.5-4',
        'LHCb B->K*mumu 2015 P 4-6']
        #'ATLAS B->K*mumu 2017 P5p' ]
fpl.bin_plot_exp('<P5p>(B0->K*mumu)',
                 col_dict= {'ATLAS': 'c', 'LHCb': 'm'},
                 divide_binwidth=False,
                 include_measurements=measur)

# Flavio theoretical prediction
#fpl.bin_plot_th( '<P5p>(B0->K*mumu)', bins,
#                 label='SM-th-Flavio', divide_binwidth=False,
#                 N=50,threads=2)

plt.xlabel('$q^2 \hspace{2pt} (GeV^2)$')
plt.ylabel('$P5 \hspace{2pt} (q^2)$')
plt.legend()
plt.title('$P_5\'$ prediction with $ (\delta C_7, \delta C_9, \delta C_{10}) = (.1, .1, .1)$')
plt.show()
#plt.ylim(-1.2, 0.7)
#plt.savefig('Fig1_NewP5.png', bbox_inches='tight')

