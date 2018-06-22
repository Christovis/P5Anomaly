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
import FFScan
import ParScan

SMpred = P5p_anomaly.P5p_binned() #Central value prediction

FFerrBar = FFScan.FindErrBar() #Scan over the FF parameters uncertainties phace space
ParErrBar = ParScan.FindErrBar() #Scan over some parameters (masses, alpha_s) phace space
TotBarUp = []
TotBarDown = []
for i in range(len(FFerrBar[0])):
    TotBarUp.append(np.sqrt(FFerrBar[0][i]**2 + ParErrBar[0][i]**2)) #Add errors in quadrature
    TotBarDown.append(np.sqrt(FFerrBar[1][i]**2 + ParErrBar[1][i]**2))

    
# Bin Plot with error bars
def BinPlot():
    rc('font',**{'family':'serif','serif':['Palatino']})
    rc('text', usetex=True)
    
    bins=[(0.1, 0.98), (1.1, 2.5), (2.5, 4.), (4., 6.0)]
        
    ax=plt.gca()
    ax.set_xlim([0, 6.1])
    ax.set_ylim([-1, 1.])
    for i in range(len(bins)):
        label= 'SM'
        if i>0:
            label=None
        ax.add_patch(patches.Rectangle((bins[i][0], (SMpred[i]-TotBarDown[i])), #bottom left point
                                       bins[i][1]-bins[i][0],          # width
                                       (TotBarUp[i]+TotBarDown[i]),   # height
                                       ec='b', fill= True, fc= 'c',
                                       lw=True, hatch= '///',
                                       label=label, capstyle= 'butt'))
            
    # Falvio experimental data
    measur=['LHCb B->K*mumu 2015 P 0.1-0.98',
            'LHCb B->K*mumu 2015 P 1.1-2.5',
            'LHCb B->K*mumu 2015 P 2.5-4',
            'LHCb B->K*mumu 2015 P 4-6']
            #'ATLAS B->K*mumu 2017 P5p' ]
    fpl.bin_plot_exp('<P5p>(B0->K*mumu)',
                     col_dict= {'ATLAS': 'r', 'LHCb': 'k' },  #'#EB70AA' for light pink
                     divide_binwidth=False,
                     include_measurements=measur)

    plt.xlabel('$q^2 \hspace{2pt} (GeV^2)$')
    plt.ylabel('$P5\' \hspace{2pt} (q^2)$')
    plt.legend()
    plt.title('FF+Parameter corrections added in quadrature')
    plt.show()
    return(0)

BinPlot()

