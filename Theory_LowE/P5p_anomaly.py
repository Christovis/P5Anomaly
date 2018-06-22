import numpy as np
from numpy import ma
import matplotlib.pyplot as plt
from collections import OrderedDict
import matplotlib.patches as patches
import scipy.integrate as integrate
import flavio
import flavio.plots as fpl
from matplotlib import rc

#######################################################

# Define Parameters------------------------

m_l= 0. #lepton mass (muon mass =0.10565 GeV)
m_b = 4.2 #+- 0.1 GeV from 1207.2753 pg.13
m_c = 1.3 #+- 0.2
mu_b = 4.
m_B = 5.27950 #GeV from 1207.2753 pg.13
m_Ks = 0.895  #GeV from 1207.2753 pg.13


C_F=4/3
alpha_s_h= 0.214
alpha_s_b= 0.214
alpha_em= 1/137 #at m_Z +- 0.0007
V_tbV_ts= 0.0385
G_f= 1.166378 * 10**(-5) #GeV**-2

# WC  arXiv:1606.00916v2 at scale= m_b= 4.2 Gev.

C1 =  -0.294 #257
C2 = 1.017  #109
C3 =  -0.005
C4 = -0.08
C5 = 0.000 
C6 = 0.001 
C7_eff = -0.295  #317 
C9 = 4.114      #100
C10 = -4.193     #308


# Adding NP Wilson Coefficients
NP_WC = { 'dC7'  : 0.1,
          'dC9'  : 0.1,
          'dC10' : 0.1}

SM_WC = { 'dC7'  : 0.,
          'dC9'  : 0.,
          'dC10' : 0.}


#Form Factor parameters-------------------

f_B = 0.200  # +-30 MeV
f_Ks =  0.220   # +-30 MeV
f_Ks_ort = 0.163
lambda_B_p = 3      # (+- 1 GeV**-1)



def E_Ks(q):
    return ((m_B/2) * (1- (q/(m_B**2))))


#Definition of the two form factors ksi.

#ksi parameters

ksi_ort_0 = 0.266  # +- 0.032 (Straub et. al.)
ksi_par_0 = 0.118 # +- 0.008 (same)

def ksi_ort(q):  #from arXiv hep-ph/0106067v2
    return( ksi_ort_0*(1/(1 - q/(m_B**2) )))


def ksi_par(q):
    return( ksi_par_0*(1/(1-q/(m_B**2)))**3 )


# formfactor corrections parameters


#O(Lambda/m_b) corrections

Lmb_corr_par= { 'V'  : [0.,     0.,    0.],  # a_F, b_F, c_F in KMPW scheme from arXiv 1407.8526v2
               'A0' : [0.002, 0.590, 1.473],
               'A1' : [-0.013, -0.056, 0.158],
               'A2' : [-0.018, -0.105, 0.192],
               'T1' : [-0.006, -0.012, -0.034],
               'T2' : [-0.005, 0.153, 0.544],
               'T3' : [-0.002, 0.308, 0.786]}

res = []
def Delta_lmb(q, par):
    for i in ['V', 'A1', 'A2', 'A0', 'T1', 'T2', 'T3']:
        val = par[i][0] + par[i][1]*(q/(m_B**2)) + par[i][2]*(q**2/(m_B**4))
        res.append(val)
    return(res)


# O(alpha_s) corrections

a_FF_par = { 'a_par' : [ 0.17, 0.05, -0.17, 0.05], #a_1_par and a_2_par for K_bar and K decays
             'a_ort' : [ 0.18, 0.03, -0.18, 0.03]}      #  from arXiv 9805422v2


def Phi_par(u, par, fun_type): #fun_type to distinguish between B and B_bar
    i = 0
    if fun_type == 'bar':
        i = 2
    return((6*u*(1-u)*(1 + 3*par['a_par'][0+i]*(2*u-1) +\
                       par['a_par'][1+i]*3/2 * (5*(2*u-1)**2-1) ))/(1-u) )
   
def Phi_ort(u, par, fun_type):
    i = 0
    if fun_type == 'bar':
        i = 2
    return((6*u*(1-u)*(1 + 3*par['a_ort'][0+i]*(2*u-1) +\
                       par['a_ort'][1+i]*3/2 * (5*(2*u-1)**2-1) ))/(1-u))


def factor_par(fun_type):
    return integrate.quad(Phi_par, 0, 1, args=(a_FF_par, fun_type))
def factor_ort(fun_type):
    return integrate.quad(Phi_ort, 0, 1, args=(a_FF_par, fun_type))


def deltaF_par(fun_type):
    return  8*(np.pi**2)*f_B*f_Ks/(3*m_B)*(factor_par(fun_type)[0]*lambda_B_p)


def deltaF_ort(fun_type):
    return 8*(np.pi**2)*f_B*f_Ks_ort/(3*m_B)*(factor_ort(fun_type)[0]*lambda_B_p)


def deltaT1(q, fun_type):
    return( m_B/(4*E_Ks(q)) * deltaF_ort(fun_type))

def deltaT2(q, fun_type):
    return( 1/2 * deltaF_ort(fun_type))

def deltaT3(q, fun_type):
    return( deltaT1(q, fun_type) + 2*m_Ks/(m_B)*(m_B/(2*E_Ks(q)))**2 * deltaF_par(fun_type))


def L(q):
    return(- 2*E_Ks(q)/(m_B-2*E_Ks(q)) * np.log(2*E_Ks(q)/m_B))



def Delta(q, fun_type):
    return(1 +  alpha_s_b *C_F/(4*np.pi) * (-2 + 2*L(q)) -\
           alpha_s_b*C_F*2*q/((E_Ks(q)**2)*(np.pi)) *\
           np.pi**2 * m_Ks*f_B * f_Ks * lambda_B_p/ \
           (3*m_B*E_Ks(q)*ksi_par(q)) * factor_par(fun_type)[0])

def Delta_V(q):
    return(0.)

def Delta_A1(q):
    return(0.)

def Delta_A2(q):
    return(0.)

def Delta_A0(q, fun_type):
    return((E_Ks(q)/m_Ks)*ksi_par(q)*(Delta(q, fun_type)**-1 - 1))

def Delta_T1(q, fun_type):
    return(C_F*alpha_s_b*ksi_ort(q)*(np.log(m_b**2/mu_b**2) - L(q)) +\
           C_F*alpha_s_b*deltaT1(q, fun_type)  )

def Delta_T2(q, fun_type):
    return(C_F*alpha_s_b*2*E_Ks(q)/(m_B)*ksi_ort(q)*(np.log(m_b**2/mu_b**2) - L(q)) +\
           C_F*alpha_s_h*deltaT2(q, fun_type)  )

def Delta_T3(q, fun_type):
    return(C_F*alpha_s_b*(ksi_ort(q)*(np.log(m_b**2/mu_b**2) - L(q)) -\
                        ksi_par(q)*(np.log(m_b**2/mu_b**2) + 2*L(q))) +\
           C_F*alpha_s_h*deltaT3(q, fun_type)  )


corr = { 'V' : [Delta_V,  Delta_lmb ],
         'A1': [Delta_A1, Delta_lmb ],
         'A2': [Delta_A2, Delta_lmb ],
         'A0': [Delta_A0, Delta_lmb ],
         'T1': [Delta_T1, Delta_lmb ],
         'T2': [Delta_T2, Delta_lmb ],
         'T3': [Delta_T3, Delta_lmb ]}

###############################################################################

#In the low q regime (q<< m_B^2) we have: NB we use q in place of q^2.


###Functions needed for C9_eff

def h(q, m):
    z = 4*m**2/q
    if m==0:
        con = -4/9*(-np.log(mu_b**2)-2/3-z) - \
              4/9*(2+z)*np.sqrt(np.absolute(z-1)) * \
              (np.log(2) - 1/2*np.log(4/q) - np.pi * 1j/2)
        return con
    else:
        con = -4/9*(np.log(m**2/mu_b**2)-2/3-z) - \
              4/9*(2+z)*np.sqrt(np.absolute(z-1))*np.arctan(1/np.sqrt(z-1))
        return con

def Y(q, m_b, m_c):
    con = h(q, m_c)*(4/3*C1 + C2 + 6*C3 + 60*C5) - \
          1/2*h(q, m_b)*(7*C3 + 4/3*C4 + 76*C5 + 64/3*C6) - \
          1/2*(h(q, 0)*(C3 + 4/3*C4 + 16*C5 + 64/3*C6)) + \
          4/3*C3 + 64/9*C5 + 64/27*C6
    return con


#Definition of the seven full form factors with NLO corrections

def V(q, corr, fun_type):
    return( (m_B + m_Ks)/m_B * ksi_ort(q) +\
                corr['V'][0](q) + corr['V'][1](q, Lmb_corr_par)[0])
  
def A1(q,  corr, fun_type):
    return((2* E_Ks(q))/(m_B + m_Ks)*ksi_ort(q) +\
           corr['A1'][0](q) + corr['A1'][1](q, Lmb_corr_par)[1])

def A2(q,  corr, fun_type):
    return(m_B/(m_B-m_Ks)*(ksi_ort(q) - ksi_par(q)) +\
           corr['A2'][0](q) + corr['A2'][1](q, Lmb_corr_par)[2])

def A0(q,  corr, fun_type):
    return((E_Ks(q)/m_Ks)*ksi_par(q) +\
           corr['A0'][0](q, fun_type) + corr['A0'][1](q, Lmb_corr_par)[3])

def T1(q,  corr, fun_type):
    return(ksi_ort(q) +\
           corr['T1'][0](q, fun_type) + corr['T1'][1](q, Lmb_corr_par)[4])

def T2(q,  corr, fun_type):
    return((2* E_Ks(q))/m_B*ksi_ort(q) +\
           corr['T2'][0](q, fun_type) + corr['T2'][1](q, Lmb_corr_par)[5])

def T3(q,  corr, fun_type):
    return(ksi_ort(q) - ksi_par(q) +\
           corr['T3'][0](q, fun_type) + corr['T3'][1](q, Lmb_corr_par)[6])



##Factors needed for the amplitudes

def lmb(q):
    return(m_B**4 + m_Ks**4 + q**2 - 2*((m_B*m_Ks)**2 + q*(m_Ks**2) + q*(m_B**2)))

def beta_l(q):
    return(np.sqrt(1 - (4* m_l**2)/q))

def N(q):
   con = V_tbV_ts*(np.sqrt(G_f**2 *alpha_em**2 *q *np.sqrt(lmb(q)) *beta_l(q) / \
                           (3* (2.**10)*(np.pi)**5 * m_B**3)))
   return con


##Amplitudes (arXiv 0807.2589v3)

def A_ort(q, chir, NP, corr, fun_type):
    if chir == 'L':
        res = np.sqrt(2*lmb(q))*N(q) * (\
              ((C9 + NP['dC9'] + Y(q, m_b, m_c))  - (C10 + NP['dC10']) )*\
                                        V(q,  corr, fun_type)/(m_B + m_Ks) + \
                ((2*m_b)/q) * (C7_eff + NP['dC7'])*T1(q,  corr, fun_type))
        return res
    elif chir == 'R':
        res = np.sqrt(2*lmb(q))*N(q) * (\
              ((C9 + NP['dC9'] + Y(q, m_b, m_c))  + (C10 + NP['dC10']) )*\
                                        V(q,  corr, fun_type)/(m_B + m_Ks) + \
                ((2*m_b)/q) * (C7_eff + NP['dC7'])*T1(q,  corr, fun_type))
        return res
    else:
        print("Invalid chirality argument")

def A_par(q, chir,  NP, corr, fun_type):
    if chir == 'L':
        res = -np.sqrt(2)*N(q)*(m_B**2 - m_Ks**2)*(\
            ( (C9 + NP['dC9'] + Y(q, m_b, m_c)) - (C10 + NP['dC10']))*\
                                                   A1(q,  corr, fun_type)/(m_B - m_Ks) +\
            ((2*m_b)/q) * (C7_eff + NP['dC7']) * T2(q,  corr, fun_type))
        return res
    elif chir == 'R':
       res = -np.sqrt(2)*N(q)*(m_B**2 - m_Ks**2)* (\
            ((C9 + NP['dC9'] + Y(q, m_b, m_c))  + (C10 + NP['dC10']))*\
                                                   A1(q,  corr, fun_type)/(m_B - m_Ks) +\
            ((2*m_b)/q) * (C7_eff + NP['dC7']) * T2(q,  corr, fun_type))
       return res
    else:
        print("Invalid chirality argument")


def A_0(q, chir, NP, corr, fun_type):
    if chir == 'L':
        res = -N(q)/(2.*m_Ks*np.sqrt(q)) * \
            ( ((C9 + NP['dC9'] + Y(q, m_b, m_c))  - (C10 + NP['dC10']))* ((m_B**2 - m_Ks**2 -q)*\
             (m_B + m_Ks)*A1(q,  corr, fun_type) - lmb(q)* A2(q,  corr, fun_type)/(m_B + m_Ks)) +\
             (2*m_b)*(C7_eff + NP['dC7']) * ((m_B**2 + 3*m_Ks**2-q)*T2(q,  corr, fun_type) -\
                                             (lmb(q)/(m_B**2-m_Ks**2))*T3(q,  corr, fun_type))) 
        return res
    elif chir == 'R':
        res = -N(q)/(2.*m_Ks*np.sqrt(q)) * \
              ( ((C9 + NP['dC9'] + Y(q, m_b, m_c)) + (C10 + NP['dC10']))* ((m_B**2 - m_Ks**2 -q)*\
                (m_B + m_Ks)*A1(q,  corr, fun_type) - lmb(q)* A2(q, corr, fun_type)/(m_B + m_Ks)) +\
                (2*m_b)*(C7_eff + NP['dC7']) * ((m_B**2 + 3*m_Ks**2-q)*T2(q,  corr, fun_type) -\
                                                (lmb(q)/(m_B**2-m_Ks**2))*T3(q, corr, fun_type))) 
        return res
    else:
        print("Invalid chirality argument")

        
def A_t(q,  NP, corr, fun_type):
    res =( (N(q)*np.sqrt(lmb(q)))/np.sqrt(q) * (2* C10+ NP['dC10'])*A0(q,  corr, fun_type))
    return(res)


###Angular obervables
def J_1s(q,  NP, corr, fun_type):
    if fun_type == 'bar':
        J = ((2+beta_l(q)**2)/4.) * (np.absolute(A_ort(q, "L",  NP, corr, fun_type))**2 + \
                                     np.absolute(A_par(q, "L",  NP, corr, fun_type))**2 + \
                                     np.absolute(A_ort(q, "R",  NP, corr, fun_type))**2 + \
                                     np.absolute(A_par(q, "R",  NP, corr, fun_type))**2) + \
                                     ((4* (m_l)**2)/q) * (A_ort(q, "R",  NP, corr, fun_type)*\
                                                    np.conj(A_ort(q, "L",  NP, corr, fun_type)) + \
                                                        A_par(q, "R",  NP, corr, fun_type)*\
                                                np.conj(A_par(q, "L",  NP, corr, fun_type))).real
        return J
    else:
        J = ((2+beta_l(q)**2)/4.) * (np.absolute(A_ort(q, "L",  NP, corr, fun_type))**2 + \
                                 np.absolute(A_par(q, "L",  NP, corr, fun_type))**2 + \
                                 np.absolute(A_ort(q, "R",  NP, corr, fun_type))**2 + \
                                 np.absolute(A_par(q, "R",  NP, corr, fun_type))**2) + \
        ((4*(m_l**2))/q) * (A_ort(q, "L",  NP, corr, fun_type) *\
                            np.conj(A_ort(q, "R",  NP, corr, fun_type)) +  \
                            A_par(q, "L",  NP, corr, fun_type) *\
                            np.conj(A_par(q, "R",  NP, corr, fun_type))).real
        return J


def J_1c(q,  NP, corr, fun_type):
    if fun_type == 'bar':
        J = np.absolute(A_0(q, "L",  NP, corr, fun_type))**2 +\
            np.absolute(A_0(q, "R",  NP, corr, fun_type))**2 + \
            (4* (m_l**2)/q)*(np.absolute(A_t(q,  NP, corr, fun_type))**2 + \
                          2*(A_0(q, "R",  NP, corr, fun_type)*\
                             np.conj(A_0(q,"L",  NP, corr, fun_type))).real)
        return J
    else:
        J = np.absolute(A_0(q, "L",  NP, corr, fun_type))**2 +\
            np.absolute(A_0(q, "R",  NP, corr, fun_type))**2 + \
            (4*(m_l**2)/q) * (np.absolute(A_t(q,  NP, corr, fun_type))**2 + \
                              2*(A_0(q, "L",  NP, corr, fun_type)* \
                                 np.conj(A_0(q, "R",  NP, corr, fun_type))).real)
        return J
  
        

def J_2s(q,  NP, corr, fun_type):
    if fun_type == 'bar':
        return(((beta_l(q)**2)/4)*(np.absolute(A_ort(q,"L",  NP, corr, fun_type))**2+\
                                   np.absolute(A_par(q,"L",  NP, corr, fun_type))**2 +\
                                   np.absolute(A_ort(q,"R",  NP, corr, fun_type))**2+\
                                   np.absolute(A_par(q,"R" , NP, corr, fun_type))**2))
    else:
        J = ((beta_l(q)**2)/4) * (np.absolute(A_ort(q, "L",  NP, corr, fun_type))**2 + \
                              np.absolute(A_par(q, "L",  NP, corr, fun_type))**2 + \
                              np.absolute(A_ort(q, "R",  NP, corr, fun_type))**2 + \
                              np.absolute(A_par(q, "R",  NP, corr, fun_type))**2)
        return J


def J_2c(q,  NP, corr, fun_type):
    if fun_type == 'bar':
        return(-(beta_l(q)**2)*(np.absolute(A_0(q, "L",  NP, corr, fun_type))**2\
                                + np.absolute(A_0(q, "R",  NP, corr, fun_type))**2))
    else: 
        J = -(beta_l(q)**2) * (np.absolute(A_0(q, "L",  NP, corr, fun_type))**2 + \
                           np.absolute(A_0(q, "R",  NP, corr, fun_type))**2)
        return J


def J_5(q,  NP, corr, fun_type):
    if fun_type == 'bar':
        return(np.sqrt(2)*beta_l(q)*((A_ort(q,"L",  NP, corr, fun_type)*\
                                      np.conj(A_0(q,"L",  NP, corr, fun_type))).real- \
                                     (A_ort(q,"R",  NP, corr, fun_type)*\
                                      np.conj(A_0(q,"R",  NP, corr, fun_type))).real))
    else:
        J = np.sqrt(2)*beta_l(q)*((A_0(q, "L",  NP, corr, fun_type)* \
                                   np.conj(A_ort(q, "L",  NP, corr, fun_type))).real - \
                              (A_0(q, "R",  NP, corr, fun_type)*\
                               np.conj(A_ort(q, "R",  NP, corr, fun_type))).real)
        return J


def DecayRate(q,  NP, corr, fun_type):
    gamma = 3*(2*J_1s(q,  NP, corr, fun_type)+J_1c(q,  NP, corr, fun_type))/ \
            4. - (2*J_2s(q,  NP, corr, fun_type)+J_2c(q,  NP, corr, fun_type))/4.
    return gamma


def S5(q,  NP, corr):
    return((J_5(q,  NP, corr, 'real') + J_5(q,  NP, corr, 'bar'))/ \
           (DecayRate(q,  NP, corr, 'real') + DecayRate(q,  NP, corr, 'bar')))


def S2_c(q, NP, corr):
    return( (J_2c(q, NP, corr, 'real') + J_2c(q, NP, corr, 'bar') )/ \
            (DecayRate(q,  NP, corr, 'real') + DecayRate(q,  NP, corr, 'bar')) )


def FL(q,  NP, corr):
    return (- S2_c(q, NP, corr))

def P_5_p(q,  NP, corr):
    return(S5(q,  NP, corr)/(np.sqrt(FL(q,  NP, corr)*(1-FL(q,  NP, corr)))))


###############################################################################
# Run Experiment (Finding the integrated values)


bins = np.array([[0.1, 0.98], [1.1, 2.5], [2.5, 4], [4, 6.]])
bins_lim = np.array([.1])
for i in range(len(bins)):
    bins_lim = np.append(bins_lim, bins[i][1])

   
def c_4(q,  NP, corr):
    c4 = (1.-FL(q, NP, corr))*(DecayRate(q, NP, corr, 'real') + \
                              DecayRate(q, NP, corr, 'bar'))
    return c4

def c_0(q,  NP, corr):
    return( DecayRate(q, NP, corr, 'real') + DecayRate(q, NP, corr, 'bar'))

def J_5_(q,  NP, corr):
    return(J_5(q,  NP, corr, 'real') + J_5(q,  NP, corr, 'bar'))

def P5p_binned():
    results_SM = []
    for bin in range(len(bins)):
        min_val=bins[bin][0]
        max_val=bins[bin][1]
        J5_bin_sm = integrate.quad(J_5_, min_val, max_val, args=( SM_WC, corr))
        c0_bin_sm = integrate.quad(c_0, min_val, max_val, args=( SM_WC, corr))
        c_4_bin_sm = integrate.quad(c_4, min_val, max_val, args=( SM_WC, corr))
        P_5p_bin_sm = J5_bin_sm[0]/np.sqrt( c_4_bin_sm[0] * (c0_bin_sm[0]-c_4_bin_sm[0]))
        results_SM.append(P_5p_bin_sm)
    return results_SM

'''
print('SM values= ', P5p_binned())

bins.tolist() #needed for Flavio th-prediction
bins=[tuple(entry) for entry in bins]
res = np.array(P5p_binned())
res = np.append(res, -1)
axes = plt.gca()
axes.set_xlim([0, 6.1])
axes.set_ylim([-1.6, 1])
plt.step( bins_lim, res,
              'c', where='post', label='SM')
    
fpl.bin_plot_th( '<P5p>(B0->K*mumu)', bins,
                 label='SM-th-Flavio', divide_binwidth=False,
                 N=50,threads=2)

plt.legend()
plt.show()
'''
