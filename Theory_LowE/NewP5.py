import numpy as np
from numpy import ma
import matplotlib.pyplot as plt
from collections import OrderedDict
import matplotlib.patches as patches
import scipy.integrate as integrate
import flavio
import flavio.plots as fpl
from flavio.statistics import fits
from matplotlib import rc

###############################################################################
# Define Parameters
m_l= 0. #lepton mass (muon mass =0.10565 GeV)
m_b= 4.8 #GeV from 1207.2753 pg.13
m_c= 1.5
m_B= 5.27950 #GeV from 1207.2753 pg.13
m_Ks= 0.895 #GeV from 1207.2753 pg.13
m_b_hat=m_b/m_B
m_Ks_hat= m_Ks/m_B

alpha_em= 1/128 #at m_Z +- 0.0007
V_tbV_ts= 0.0428
G_f= 1.166378* 10**(-5) #GeV**-2

# WC  taken from Ali, Ball et.al., at scale= m_b= 4.8 Gev.
C1 = -0.267
C2 = 1.107
C3 = -0.011
C4 = -0.026
C5 = 0.007
C6 = 0.031
C7_eff = -0.313
C9 = 4.344
C10 = -4.669

C7_eff_p = 0.0
C9_eff_p = 0.0
C10_p = 0.0

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
# Defina Functions

#In the low q regime (q<< m_B^2) we have: NB we use q in place of q^2.

def E_Ks(q):
    return ((m_B/2) * (1- (q/(m_B**2))))

def s_hat(q):
    return(q/(m_B**2))
   
###Functions needed for C9_eff(q dC7)

def h(q, m):
    z = 4*m**2/q
    if m==0:
        con = -4/9*(-np.log(m_b**2)-2/3-z) - \
              4/9*(2+z)*np.sqrt(np.absolute(z-1)) * \
              (np.log(2) - 1/2*np.log(4/q) - np.pi * 1j/2)
        return con
    else:
        con = -4/9*(np.log(m**2/m_b**2)-2/3-z) - \
              4/9*(2+z)*np.sqrt(np.absolute(z-1))*np.arctan(1/np.sqrt(z-1))
        return con

def Y(q, m_b, m_c):
    con = h(q, m_c)*(4/3*C1 + C2 + 6*C3 + 60*C5) - \
          1/2*h(q, m_b)*(7*C3 + 4/3*C4 + 76*C5 + 64/3*C6) - \
          1/2*(h(q, 0)*(C3 + 4/3*C4 + 16*C5 + 64/3*C6)) + \
          4/3*C3 + 64/9*C5 + 64/27*C6
    return con

def C9_eff(q):
    return (C9 + Y(q, m_b, m_c))

###From factors: central, min and max values

def V(q, FF):  # s=m_B**2
    V = FF['F_0'][2] * np.exp(FF['c1'][2]*s_hat(q) + \
                              FF['c2'][2]*s_hat(q)**2 + \
                              FF['c3'][2]*s_hat(q)**3)
    return V

def A_1(q, FF): # s=m_B**2
    A = FF['F_0'][0] * np.exp(FF['c1'][0]*s_hat(q) + \
                              FF['c2'][0]*s_hat(q)**2 + \
                              FF['c3'][0]*s_hat(q)**3)
    return A

def A_2(q, FF): # s=m_B**2
    A = FF['F_0'][1] * np.exp(FF['c1'][1]*s_hat(q) + \
                              FF['c2'][1]*s_hat(q)**2 + \
                              FF['c3'][1]*s_hat(q)**3)
    return A


def ksi_ort(q, FF):
    return(m_B*V(q, FF)/(m_B + m_Ks))

def ksi_par(q, FF):
    ksi = ((m_B + m_Ks)/(2*E_Ks(q)))*A_1(q, FF) - ((m_B - m_Ks)/(m_B))*A_2(q, FF)
    return ksi

##Factors needed for the amplitudes

def lmb(q):
    return(m_B**4 + m_Ks**4 + q**2 - 2*((m_B*m_Ks)**2 + q*(m_Ks**2) + q*(m_B**2)))

def beta_l(q):
    return(np.sqrt(1 - (4*m_l**2)/q))

def N(q):
   con = V_tbV_ts*(np.sqrt(G_f**2*alpha_em**2*q*np.sqrt(lmb(q))*beta_l(q) / \
                   (3*2.**10*(np.pi)**5*m_B**3)))
   return con


##Amplitudes 
def A_ort(q, chir, FF):
    if chir == 'L':
        con = np.sqrt(2)*N(q)*m_B*(1 - s_hat(q)) * \
              ((C9_eff(q)+C9_eff_p) - (C10+C10_p) + ((2*m_b_hat)/s_hat(q)) * \
               (C7_eff+C7_eff_p))
        return con*ksi_ort(q, FF)
    elif chir == 'R':
        con = np.sqrt(2)*N(q)*m_B*(1 - s_hat(q)) * \
              ((C9_eff(q)+C9_eff_p) + (C10+C10_p) + ((2*m_b_hat)/s_hat(q)) * \
               (C7_eff+C7_eff_p))
        return con*ksi_ort(q, FF)
    else:
        print("Invalid chirality argument")

        
def A_par(q, chir, FF):
    if chir == 'L':
        A = -np.sqrt(2)*N(q)*m_B*(1-s_hat(q)) * \
            ((C9_eff(q)-C9_eff_p) - (C10-C10_p) + ((2*m_b_hat)/s_hat(q)) * \
             (C7_eff-C7_eff_p)) * ksi_ort(q, FF)
        return A
    elif chir == 'R':
        A = -np.sqrt(2)*N(q)*m_B*(1-s_hat(q)) * \
            ((C9_eff(q)-C9_eff_p) + (C10 - C10_p) + ((2*m_b_hat)/s_hat(q)) * \
             (C7_eff-C7_eff_p)) * ksi_ort(q, FF)
        return A
    else:
        print("Invalid chirality argument")


def A_0(q, chir, FF):
    if chir == 'L':
        A = (-N(q)*m_B**2)/(2.*m_Ks*np.sqrt(s_hat(q))) * \
            (1-s_hat(q))**2 * ((C9_eff(q)+C9_eff_p) - (C10+C10_p) + \
                               (2*m_b_hat)*(C7_eff-C7_eff_p)) * \
            ksi_par(q, FF)
        return A
    elif chir == 'R':
        A = (-N(q)*m_B**2)/(2.*m_Ks*np.sqrt(s_hat(q)) )* \
            (1-s_hat(q))**2 * ((C9_eff(q)+C9_eff_p) + (C10+C10_p) + \
                               2*m_b_hat*(C7_eff-C7_eff_p)) * \
            ksi_par(q, FF)
        return A
    else:
        print("Invalid chirality argument")

        
def A_t(q, FF):
    A = (N(q)*m_B**2)/(m_Ks*np.sqrt(s_hat(q))) * \
        (1-s_hat(q))**2 * (C10-C10_p) * ksi_par(q, FF)
    return A


###Angular obervables
def J_1s(q, FF):
    J = ((2+beta_l(q)**2)/4.) * (np.absolute(A_ort(q, "L", FF))**2 + \
                                 np.absolute(A_par(q, "L", FF))**2 + \
                                 np.absolute(A_ort(q, "R" ,FF))**2 + \
                                 np.absolute(A_par(q, "R", FF))**2) + \
        ((4*(m_l**2))/q) * (A_ort(q, "L", FF) * np.conj(A_ort(q, "R", FF)) +  \
                            A_par(q, "L", FF) * np.conj(A_par(q, "R", FF))).real
    return J


def J_1c(q, FF):
    J = np.absolute(A_0(q, "L", FF))**2 + np.absolute(A_0(q, "R" ,FF))**2 + \
        (4*(m_l**2)/q) * (np.absolute(A_t(q, FF))**2 + \
                          2*(A_0(q, "L", FF)*np.conj(A_0(q, "R", FF))).real)
    return J


def J_2s(q, FF):
    J = ((beta_l(q)**2)/4) * (np.absolute(A_ort(q, "L", FF))**2 + \
                              np.absolute(A_par(q, "L", FF))**2 + \
                              np.absolute(A_ort(q, "R", FF))**2 + \
                              np.absolute(A_par(q, "R", FF))**2)
    return J


def J_2c(q, FF):
    J = -(beta_l(q)**2) * (np.absolute(A_0(q, "L", FF))**2 + \
                           np.absolute(A_0(q, "R", FF))**2)
    return J


def J_5(q, FF):
    J = np.sqrt(2)*beta_l(q)*((A_0(q, "L", FF)*np.conj(A_ort(q, "L", FF))).real - \
                              (A_0(q, "R", FF)*np.conj(A_ort(q, "R", FF))).real)
    return J


def J_1s_bar(q, FF):
    J = ((2+beta_l(q)**2)/4.) * (np.absolute(A_ort(q, "L", FF))**2 + \
                                 np.absolute(A_par(q, "L", FF))**2 + \
                                 np.absolute(A_ort(q, "R", FF))**2 + \
                                 np.absolute(A_par(q, "R", FF))**2) + \
        ((4* m_l**2)/q) * (A_ort(q, "R", FF)*np.conj(A_ort(q, "L", FF)) + \
                           A_par(q, "R", FF)*np.conj(A_par(q, "L", FF))).real
    return J


def J_1c_bar(q, FF):
    J = np.absolute(A_0(q, "L", FF))**2 + np.absolute(A_0(q, "R", FF))**2 + \
        (4*m_l**2/q)*(np.absolute(A_t(q, FF))**2 + \
                      2*(A_0(q, "R", FF)*np.conj(A_0(q,"L", FF))).real)
    return J


def J_2s_bar(q, FF):
    return(((beta_l(q)**2)/4)*(np.absolute(A_ort(q,"L", FF))**2+ np.absolute(A_par(q,"L", FF))**2 + np.absolute(A_ort(q,"R", FF))**2+ np.absolute(A_par(q,"R" ,FF))**2))

def J_2c_bar(q, FF):
    return(-(beta_l(q)**2)*(np.absolute(A_0(q, "L", FF))**2 + np.absolute(A_0(q, "R", FF))**2))

def J_5_bar(q, FF):
    return(np.sqrt(2)*beta_l(q)*((A_ort(q,"L", FF)*np.conj(A_0(q,"L", FF))).real-(A_ort(q,"R", FF)*np.conj(A_0(q,"R", FF))).real))


def DecayRate(q, FF):
    tau = 3*(2*J_1s(q, FF)+J_1c(q, FF))/4. - (2*J_2s(q, FF)+J_2c(q, FF))/4.
    return tau


def DecayRate_bar(q, FF):
    tau = 3*(2*J_1s_bar(q, FF) + J_1c_bar(q, FF))/4. - \
          (2*J_2s_bar(q, FF) + J_2c_bar(q, FF))/4.
    return tau


def S5(q, FF):
    return((J_5_bar(q, FF) + J_5(q, FF))/(DecayRate(q, FF) + DecayRate_bar(q, FF)))


def FL(q, FF):
    con = (np.absolute(A_0(q, "L", FF))**2 + \
           np.absolute(A_0(q, "R", FF))**2)/(np.absolute(A_0(q, "L", FF))**2 + \
                                             np.absolute(A_0(q, "R", FF))**2 + \
                                             np.absolute(A_par(q, "L", FF))**2 + \
                                             np.absolute(A_par(q, "R", FF))**2 + \
                                             np.absolute(A_ort(q, "L", FF))**2 + \
                                             np.absolute(A_ort(q, "R", FF))**2)
    return con


def P_5_p(q, FF):
    return(S5(q, FF)/(np.sqrt(FL(q, FF)*(1-FL(q, FF)))))


def P_5_p_2(q):
    return(J_5(q, FF)/(2 * np.sqrt(-J_2c(q, FF)*J_2s(q, FF))))

###############################################################################
# Run Experiment (Finding the integrated values)
bins = np.array([[0.1, 2.0], [2.0, 4.3], [4.3, 6]])
bins_lim = np.array([0.1])
for i in range(len(bins)):
    bins_lim = np.append(bins_lim, bins[i][1])

def c_4(q, FF):
    c4 = (1.-FL(q, FF))*(DecayRate(q, FF) + \
                              DecayRate_bar(q, FF))
    return c4
def c_0(q, FF):
    return( DecayRate(q, FF) + DecayRate_bar(q, FF))
def J_5_(q, FF):
    return(J_5(q, FF) + J_5_bar(q, FF))

results = []  #central values
for bin in range(len(bins)):
    min=bins[bin][0]
    max=bins[bin][1]
    J5_bin = integrate.quad(J_5_, min, max, args=(FormFac))
    c_0_bin = integrate.quad(c_0, min, max, args=(FormFac))
    c_4_bin = integrate.quad(c_4, min, max, args=(FormFac))
    P_5p_bin = J5_bin[0]/np.sqrt(c_4_bin[0] * (c_0_bin[0]-c_4_bin[0]))
    results.append( P_5p_bin)                              
print(results)

#max values
results_max=[]
for bin in range(len(bins)):
    min=bins[bin][0]
    max=bins[bin][1]
    J5_bin = integrate.quad(J_5_, min, max, args=(FormFacmax))
    c_0_bin = integrate.quad(c_0, min, max, args=(FormFacmax))
    c_4_bin = integrate.quad(c_4, min, max, args=(FormFacmax))
    P_5p_bin_max = J5_bin[0]/np.sqrt(c_4_bin[0] * (c_0_bin[0]-c_4_bin[0]))
    results_max.append(P_5p_bin_max)
print(results_max)
res_plt_max = np.array(results_max)
res_plt_max = np.append(res_plt_max, -1)

#min values
results_min=[]
for bin in range(len(bins)):
    min=bins[bin][0]
    max=bins[bin][1]
    J5_bin = integrate.quad(J_5_, min, max, args=(FormFacmin))
    c_0_bin = integrate.quad(c_0, min, max, args=(FormFacmin))
    c_4_bin = integrate.quad(c_4, min, max, args=(FormFacmin))
    print('min J5', J5_bin)
    P_5p_bin_min = J5_bin[0]/np.sqrt(c_4_bin[0] * (c_0_bin[0]-c_4_bin[0]))
    results_min.append(P_5p_bin_min)
print(results_min)
res_plt_min=np.array(results_min)
res_plt_min= np.append(res_plt_min, -1)

bins.tolist()
bins=[tuple(entry) for entry in bins]
res_ATLAS = np.array([0.67, -0.33, 0.26, -1])
###############################################################################
# BIN PLOT with ERROR BARS (coming only from FF)
rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)

ax=plt.gca()
ax.set_xlim([0, 6.1])
ax.set_ylim([-1, 1.])
for i in range(len(bins)):
    label= 'SM-th'
    if i>0:
       label=None
    ax.add_patch(patches.Rectangle((bins[i][0], res_plt_min[i]),
                                   bins[i][1]-bins[i][0],           # width
                                   res_plt_max[i]-res_plt_min[i],   # height
                                   ec='k', fc='c', lw=True,
                                   label=label, capstyle= 'butt'))
# Falvio
measur=['LHCb B->K*mumu 2015 P 0.1-0.98',
        'LHCb B->K*mumu 2015 P 1.1-2.5',
        'LHCb B->K*mumu 2015 P 2.5-4',
        'LHCb B->K*mumu 2015 P 4-6']
        #'ATLAS B->K*mumu 2017 P5p' ]
fpl.bin_plot_exp('<P5p>(B0->K*mumu)',
                 col_dict= {'ATLAS': 'y', 'LHCb': 'g'},
                 divide_binwidth=False,
                 include_measurements=measur)

plt.xlabel('$q^2 \hspace{2pt} (GeV^2)$')
plt.ylabel('$P5 \hspace{2pt} (q^2)$')
plt.legend()
plt.ylim(-1.2, 0.7)
plt.savefig('Fig1_NewP5.png', bbox_inches='tight')
