import numpy as np
from numpy import ma
from collections import OrderedDict
import scipy.integrate as integrate
from sklearn import linear_model
import flavio
import flavio.plots as fpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import rc
import P5_funcs as P5f

###############################################################################
# Define Parameters
m_l= 0. #lepton mass (muon mass =0.10565 GeV)
m_b= 4.8 #GeV from 1207.2753 pg.13
m_c= 1.5
m_B= 5.27950 #GeV from 1207.2753 pg.13
m_Ks= 0.895 #GeV from 1207.2753 pg.13

Mass = {'m_l' : m_l,     # lepton mass (muon mass =0.10565 GeV)
        'm_b' : m_b,     # GeV from 1207.2753 pg.13
        'm_c' : m_c,
        'm_B' : m_B,     # GeV from 1207.2753 pg.13
        'm_Ks' : m_Ks,   # GeV from 1207.2753 pg.13
        'm_b_hat' : m_b/m_B,
        'm_Ks_hat' : m_Ks/m_B}

Param = {'alpha_em' : 1/128,         # at m_Z +- 0.0007
         'V_tbV_ts' : 0.0428,
         'G_f' : 1.166378*1e-5}  # GeV**-2

# Wilson-Coefficienc (WC)
# Ali, Ball et.al., at scale= m_b= 4.8 Gev.
# Standard Model
C7_eff = -0.313
C7_eff_p = 0.0
C9 = 4.344
C9_eff_p = 0.0
C10 = -4.669
C10_p = 0.0
SM_WC = {'C1' : -0.267,
         'C2' : 1.107,
         'C3' : -0.011,
         'C4' : -0.026,
         'C5' : 0.007,
         'C6' : 0.031,
         'C7' : C7_eff + C7_eff_p,
         'C9' : C9 + C9_eff_p,
         'C10' : C10 + C10_p}
# New Physics
dC9=0.1; dC7=0.1; dC10=0.1
NP_WC = {'C1' : -0.267,
         'C2' : 1.107,
         'C3' : -0.011,
         'C4' : -0.026,
         'C5' : 0.007,
         'C6' : 0.031,
         'C7' : C7_eff + C7_eff_p + dC7,
         'C9' : C9 + C9_eff_p + dC9,
         'C10' : C10 + C10_p + dC10}

# New parametrization of form factors from Patricia Bell, Ali et. al.
# Central vlaues of the FF parameters for A_1, A_2 and V
FormFac = {'F_0' : [0.337 , 0.282 , 0.457],
           'c1' : [0.602 , 1.172 , 1.482],
           'c2' : [0.258 , 0.567 , 1.015],
           'c3' : [0.0 , 0.0 , 0.0]}
#Maximum allowed FF parameters
FormFacmax = {'F_0' : [0.385 , 0.320 , 0.548],
              'c1' : [0.557 , 1.083 , 1.462],
              'c2' :[0.068 , 0.393 , 0.953],
              'c3' : [0.0 , 0.0 , 0.0]}
# Minimumallowed FF parameters
FormFacmin = {'F_0' : [0.294 , 0.246 , 0.399],
              'c1' : [0.656 , 1.237 , 1.537],
              'c2' : [0.456 , 0.822 , 1.123],
              'c3' : [0.0 , 0.0 , 0.0]}

###############################################################################
# Run Experiment (Finding the integrated values)

bins = np.array([[0.1, 2.0], [2.0, 4.3], [4.3, 6]])
bins_lim = np.array([0.1])
for i in range(len(bins)):
    bins_lim = np.append(bins_lim, bins[i][1])


def c_4(q, FF, WC, M, P):
    c4 = (1. - P5f.FL(q, FF, WC, M, P))*(P5f.DecayRate(q, FF, WC, M, P) + \
                                         P5f.DecayRate_bar(q, FF, WC, M, P))
    return c4


def c_0(q, FF, WC, M, P):
    c0 = P5f.DecayRate(q, FF, WC, M, P) + P5f.DecayRate_bar(q, FF, WC, M , P)
    return c0

def J_5_(q, FF, WC, M, P):
    j5 = P5f.J_5(q, FF, WC, M, P) + P5f.J_5_bar(q, FF, WC, M , P)
    return j5


results_SM = []
results_NP = [] #central values
for bin in range(len(bins)):
    min = bins[bin][0]
    max = bins[bin][1]
    J5_bin_np = integrate.quad(J_5_, min, max, args=(FormFac, NP_WC, Mass, Param))
    c_0_bin_np = integrate.quad(c_0, min, max, args=(FormFac, NP_WC, Mass, Param))
    c_4_bin_np = integrate.quad(c_4, min, max, args=(FormFac, NP_WC, Mass, Param))
    P_5p_bin_np = J5_bin_np[0]/np.sqrt(c_4_bin_np[0] * (c_0_bin_np[0]-c_4_bin_np[0]))
    results_NP.append(P_5p_bin_np)
    J5_bin_sm = integrate.quad(J_5_, min, max, args=(FormFac, SM_WC, Mass, Param))
    c_0_bin_sm = integrate.quad(c_0, min, max, args=(FormFac, SM_WC, Mass, Param))
    c_4_bin_sm = integrate.quad(c_4, min, max, args=(FormFac, SM_WC, Mass, Param))
    P_5p_bin_sm = J5_bin_sm[0]/np.sqrt(c_4_bin_sm[0] * (c_0_bin_sm[0]-c_4_bin_sm[0]))
    results_SM.append(P_5p_bin_sm)                              
print('SM values= ', results_SM, '\n', 'NP values= ', results_NP)

#max values
results_max_SM=[]
results_max_NP=[]
for bin in range(len(bins)):
    min=bins[bin][0]
    max=bins[bin][1]
    J5_bin_np = integrate.quad(J_5_, min, max, args=(FormFacmax, NP_WC, Mass, Param))
    c_0_bin_np = integrate.quad(c_0, min, max, args=(FormFacmax, NP_WC, Mass, Param))
    c_4_bin_np = integrate.quad(c_4, min, max, args=(FormFacmax, NP_WC, Mass, Param))
    P_5p_bin_max_np = J5_bin_np[0]/np.sqrt(c_4_bin_np[0] * (c_0_bin_np[0]-c_4_bin_np[0]))
    results_max_NP.append(P_5p_bin_max_np)
    J5_bin_sm = integrate.quad(J_5_, min, max, args=(FormFacmax, SM_WC, Mass, Param))
    c_0_bin_sm = integrate.quad(c_0, min, max, args=(FormFacmax, SM_WC, Mass, Param))
    c_4_bin_sm = integrate.quad(c_4, min, max, args=(FormFacmax, SM_WC, Mass, Param))
    P_5p_bin_max_sm = J5_bin_sm[0]/np.sqrt(c_4_bin_sm[0] * (c_0_bin_sm[0]-c_4_bin_sm[0]))
    results_max_SM.append(P_5p_bin_max_sm)
print('SM-max values= ', results_max_SM, '\n', 'NP-max values= ', results_max_NP)

res_plt_max_sm = np.array(results_max_SM)
res_plt_max_sm = np.append(res_plt_max_sm, -1)
res_plt_max_np = np.array(results_max_NP)
res_plt_max_np = np.append(res_plt_max_np, -1)

#min values
results_min_SM=[]
results_min_NP=[]
for bin in range(len(bins)):
    min=bins[bin][0]
    max=bins[bin][1]
    J5_bin_np = integrate.quad(J_5_, min, max, args=(FormFacmin, NP_WC, Mass, Param))
    c_0_bin_np = integrate.quad(c_0, min, max, args=(FormFacmin, NP_WC, Mass, Param))
    c_4_bin_np = integrate.quad(c_4, min, max, args=(FormFacmin, NP_WC, Mass, Param))
    P_5p_bin_min_np = J5_bin_np[0]/np.sqrt(c_4_bin_np[0] * (c_0_bin_np[0]-c_4_bin_np[0]))
    results_min_NP.append(P_5p_bin_min_np)
    J5_bin_sm = integrate.quad(J_5_, min, max, args=(FormFacmin, SM_WC, Mass, Param))
    c_0_bin_sm = integrate.quad(c_0, min, max, args=(FormFacmin, SM_WC, Mass, Param))
    c_4_bin_sm = integrate.quad(c_4, min, max, args=(FormFacmin, SM_WC, Mass, Param))
    P_5p_bin_min_sm = J5_bin_sm[0]/np.sqrt(c_4_bin_sm[0] * (c_0_bin_sm[0]-c_4_bin_sm[0]))
    results_min_SM.append(P_5p_bin_min_sm)
print('SM-min values= ', results_min_SM, '\n', 'NP-min values= ', results_min_NP)

res_plt_min_sm=np.array(results_min_SM)
res_plt_min_sm= np.append(res_plt_min_sm, -1)
res_plt_min_np=np.array(results_min_NP)
res_plt_min_np= np.append(res_plt_min_np, -1)

bins.tolist() #needed for Flavio th-prediction
bins=[tuple(entry) for entry in bins]


###############################################################################
# Parameter Fitting
reg = linear_model.LinearRegression()
reg.fit([[0, 0], [1, 1], [2, 2]], [0, 1, 2])
reg.coef_

###############################################################################
# BIN PLOT with ERROR BARS (coming only from FF)

rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)

ax=plt.gca()
ax.set_xlim([0, 6.1])
ax.set_ylim([-1, 1.])
for i in range(len(bins)):
    label= 'SM'
    if i>0:
       label=None
    ax.add_patch(patches.Rectangle((bins[i][0], res_plt_min_sm[i]),
                                   bins[i][1]-bins[i][0],           # width
                                   res_plt_max_sm[i]-res_plt_min_sm[i],   # height
                                   ec='m', fc='m', lw=True,
                                   label=label, capstyle= 'butt'))
for i in range(len(bins)):
    label= 'NP'
    if i>0:
       label=None
    ax.add_patch(patches.Rectangle((bins[i][0], res_plt_min_np[i]),
                                   bins[i][1]-bins[i][0],           # width
                                   res_plt_max_np[i]-res_plt_min_np[i],   # height
                                   ec='c', fc='c', lw=True,
                                   label=label, capstyle= 'butt'))

# Falvio experimental data
measur=['LHCb B->K*mumu 2015 P 0.1-0.98',
        'LHCb B->K*mumu 2015 P 1.1-2.5',
        'LHCb B->K*mumu 2015 P 2.5-4',
        'LHCb B->K*mumu 2015 P 4-6']
        #'ATLAS B->K*mumu 2017 P5p' ]
fpl.bin_plot_exp('<P5p>(B0->K*mumu)',
                 col_dict= {'ATLAS': 'y', 'LHCb': 'g'},
                 divide_binwidth=False,
                 include_measurements=measur)

# Flavio theoretical prediction
fpl.bin_plot_th( '<P5p>(B0->K*mumu)', bins,
                 label='SM-th-Flavio', divide_binwidth=False,
                 N=50,threads=2)

plt.xlabel('q2 (GeV2)')
plt.ylabel('P5 (q2)')
plt.legend()
#plt.title('$P_5\'$ prediction with $ (\delta C_7, \delta C_9, \delta C_{10}) = (.1, .1, .1)$')
plt.savefig('Fig1_NewP5.png', bbox_inches='tight')
