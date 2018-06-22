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
import FFcorrScan
import ParScan

#SM predictions
SMpred = P5p_anomaly.P5p_binned() #don't call twice, Pypet scan will change the parameter values 


# LHCb measurements for bins (From Flavio, LHCb 2015)
ExpMeasur = [0.387, 0.289, -0.066, -0.3]
ExpBarMax = [0.184, 0.243, 0.366, 0.181]
ExpBarMin = [0.185, 0.225, 0.387, 0.182]


# Use FFcorrScan to generate maximum and minimum values (FF corrections only)
FFCorr = FFcorrScan.FindMaxMin()
ParCorr = ParScan.FindMaxMin()

#Generate error bars
def FindErrBar():
    FFbarMax = []
    FFbarMin = []
    ParbarMax = []
    ParbarMin = []
    TotBarMax = []
    TotBarMin = []
    for i in range(len(FFCorr[0])):
        FFbarMax.append( np.absolute(FFCorr[0][i] - SMpred[i]))
        FFbarMin.append( np.absolute(SMpred[i] - FFCorr[1][i]))
        ParbarMax.append( np.absolute(ParCorr[0][i] - SMpred[i]))
        ParbarMin.append( np.absolute(SMpred[i] - ParCorr[1][i]))
        TotBarMax.append(FFbarMax+ParbarMax)
        TotBarMin.append(ParbarMin+FFbarMin)
    return(TotBarMax, TotBarMin) #return list of max-err-bar and min-err-bar


for i in range(len(FindErrBar())):
    print(FindErrBar()[i])


#Generate the NP only contribution
def FindNPpred():
    Difference = []
    MaxErrBar = []
    MinErrBar = []
    for i in range(len(ExpMeasur)):
        diff = ExpMeasur[i] - SMpred[i] #Difference between Exp. and SM predictions
        Difference.append(diff)
        err_max = np.sqrt(ExpBarMax[i]**2 + FindErrBar()[0][i]**2) #Add in quadrature the up err bars
        MaxErrBar.append(err_max)
        err_min = np.sqrt(ExpBarMin[i]**2 + FindErrBar()[1][i]**2) #Add in quadrature the down err bars
        MinErrBar.append(err_min)
    return(Difference, MaxErrBar, MinErrBar)

#for i in range(len(bins)):
#    print('%i bin' %i, 'Diff',  FindNPpred()[0][i], ' ', 'Errmax' , FindNPpred()[1][i], ' ', 'Errmin' , FindNPpred()[2][i])
    


# Bin Plot with error bars

def BinPlot():
    rc('font',**{'family':'serif','serif':['Palatino']})
    rc('text', usetex=True)
    
    bins=[(0.1, 0.98),(1.1, 2.5), (2.5, 4.), (4., 6.0)]
    #bins.tolist() #needed for Flavio th-prediction
    #bins=[tuple(entry) for entry in bins]

    BinsLim = [] #needed fot the scatter plot
    for i in range(len(bins)):
        res = bins[i][0]+(bins[i][1]-bins[i][0])/2
        BinsLim.append(res)
    
    Max_plt = np.array(FindNPpred()[1])
    Max_plt = np.append(Max_plt, -1)
    Min_plt = np.array(FindNPpred()[2])
    Min_plt = np.append(Min_plt, -1)
    Central_plt = np.array(FindNPpred()[0])
    Central_plt = np.append(Central_plt, -1)

    Central = FindNPpred()[0] #needed for the scatter plot of the central values

    ax=plt.gca()
    ax.set_xlim([0, 6.1])
    ax.set_ylim([-1, 1.])
    for i in range(len(bins)):
        label= 'NP prediction'
        if i>0:
            label=None
        ax.add_patch(patches.Rectangle((bins[i][0], (Central_plt[i]-Min_plt[i])),
                                       bins[i][1]-bins[i][0],          # width
                                       (Max_plt[i] + Min_plt[i]),   # height
                                       ec='c', fill= False, lw=True, hatch= '///',
                                       label=label, capstyle= 'butt'))

    plt.scatter(BinsLim, Central, c = 'c')
    
    # Falvio experimental data
    measur=['LHCb B->K*mumu 2015 P 0.1-0.98',
            'LHCb B->K*mumu 2015 P 1.1-2.5',
            'LHCb B->K*mumu 2015 P 2.5-4',
            'LHCb B->K*mumu 2015 P 4-6']
    #'ATLAS B->K*mumu 2017 P5p' ]
    
   # fpl.bin_plot_exp('<P5p>(B0->K*mumu)',
    #                 col_dict= {'ATLAS': 'c', 'LHCb': 'y' },  #'#EB70AA' for light pink
     #                divide_binwidth=False,
      #               include_measurements=measur)
    
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
    return(0)

#BinPlot()
