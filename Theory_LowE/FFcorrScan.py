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
    P5p_anomaly.ksi_ort_0=traj.ksi_ort
    P5p_anomaly.ksi_par_0=traj.ksi_par
    return P5p_anomaly.P5p_binned()

# create an environment
env = Environment()
# get the trajectory from the environment
traj = env.traj
# Add parameters
traj.f_add_parameter('ksi_ort',  1., comment = 'First dimension')
traj.f_add_parameter('ksi_par',  1., comment = 'Second dimension')
traj.f_explore(cartesian_product ({'ksi_ort' : [0.234, 0.266 , 0.298 ],
                                   'ksi_par' : [0.11, 0.118, 0.126]}))

#result=env.run(scan)
Result = result
#print(Result[0][1][0])

res=[]
#Max_values=[]
#Min_values=[]
'''
for j in range(3):
    for i in range(len(Result)):
        res.append(Result[i][1][j])
    Max_values.append(max(res))
    Min_values.append(min(res))
    res=[]
'''
#SM_res = [0.4714213878670856, -0.37291853867782426, -0.8540693764591013]
#for i in range(len(Max_values)):
#    err_max = np.absolute(np.absolute(Max_values[i]) - np.absolute(SM_res[i]))
#    err_min = np.absolute(SM_res[i] - Min_values[i]) 
#    print('central: ', SM_res[i], 'Max', err_max, ' ', 'Min',  err_min)


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
    label= 'SM-th'
    if i>0:
       label=None
    ax.add_patch(patches.Rectangle((bins[i][0], Min_plt[i]),
                                   bins[i][1]-bins[i][0],          # width
                                   Max_plt[i]-Min_plt[i],   # height
                                   ec='c', fill= False, lw=True, hatch= 'xxx',
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
#plt.title('$SM$' ''prediction)
#plt.title('$P_5\'$ prediction with $ (\delta C_7, \delta C_9, \delta C_{10}) = (.1, .1, .1)$')
plt.show()
#plt.ylim(-1.2, 0.7)
#plt.savefig('Fig1_NewP5.png', bbox_inches='tight')
