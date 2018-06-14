#from P5p_anomaly import *
import numpy as np

#In the low q regime (q<< M['m_B']^2) we have: NB we use q in place of q^2.

def E_Ks(q, m_B):
    return ((m_B/2) * (1- (q/(m_B**2))))

def s_hat(q, m_B):
    return(q/(m_B**2))
   
###Functions needed for C9_eff(q dC7)

def h(q, m, m_b):
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

def Y(q, WC, m_b, m_c):
    con = h(q, m_c, m_b)*(4/3*WC['C1'] + WC['C2'] + 6*WC['C3'] + 60*WC['C5']) - \
          1/2*h(q, m_b, m_b)*(7*WC['C3'] + 4/3*WC['C4'] + 76*WC['C5'] + 64/3*WC['C6']) - \
          1/2*(h(q, 0, m_b)*(WC['C3'] + 4/3*WC['C4'] + 16*WC['C5'] + 64/3*WC['C6'])) + \
          4/3*WC['C3'] + 64/9*WC['C5'] + 64/27*WC['C6']
    return con

# Soft From factors
def V(q, FF, m_B):  
    V = FF['F_0'][2] * np.exp(FF['c1'][2]*s_hat(q, m_B) + \
                              FF['c2'][2]*s_hat(q, m_B)**2 + \
                              FF['c3'][2]*s_hat(q, m_B)**3)
    return V


def A_1(q, FF, m_B): 
    A = FF['F_0'][0] * np.exp(FF['c1'][0]*s_hat(q, m_B) + \
                              FF['c2'][0]*s_hat(q, m_B)**2 + \
                              FF['c3'][0]*s_hat(q, m_B)**3)
    return A


def A_2(q, FF, m_B): 
    A = FF['F_0'][1] * np.exp(FF['c1'][1]*s_hat(q, m_B) + \
                              FF['c2'][1]*s_hat(q, m_B)**2 + \
                              FF['c3'][1]*s_hat(q, m_B)**3)
    return A


def ksi_ort(q, FF, M):
    ksi = M['m_B']*V(q, FF, M['m_B'])/(M['m_B'] + M['m_Ks'])
    return ksi


def ksi_par(q, FF, M):
    ksi = ((M['m_B'] + M['m_Ks'])/(2*E_Ks(q, M['m_B'])))*A_1(q, FF, M['m_B']) - \
          ((M['m_B'] - M['m_Ks'])/(M['m_B']))*A_2(q, FF, M['m_B'])
    return ksi

##Factors needed for the amplitudes

def lmb(q, M):
    lmb_ = M['m_B']**4 + M['m_Ks']**4 + q**2 - \
           2*((M['m_B']*M['m_Ks'])**2 + q*(M['m_Ks']**2) + q*(M['m_B']**2))
    return lmb_


def beta_l(q, m_l):
    return(np.sqrt(1 - (4*m_l**2)/q))


def N(q, M, P):
   con = np.sqrt(P['G_f']**2*P['alpha_em']**2*q*np.sqrt(lmb(q, M))*beta_l(q, M['m_l']) / \
                 (3*2.**10*np.pi**5*M['m_B']**3))
   con *= P['V_tbV_ts']
   return con


# Amplitudes 
def A_ort(q, chir, FF, WC, M, P):
    # https://arxiv.org/pdf/0807.2589.pdf
    # equ. 3.7
    # Y() causes deviations to results from paper
    WC['C9'] += Y(q, WC, M['m_b'], M['m_c'])
    if chir == 'L':
        A = np.sqrt(2)*N(q, M, P)*M['m_B']*(1 - s_hat(q, M['m_B'])) * \
              (WC['C9'] - WC['C10'] + ((2*M['m_b_hat'])/s_hat(q, M['m_B']))*WC['C7'])
        return A*ksi_ort(q, FF, M)
    elif chir == 'R':
        A = np.sqrt(2)*N(q, M, P)*M['m_B']*(1 - s_hat(q, M['m_B'])) * \
              (WC['C9'] + WC['C10'] + ((2*M['m_b_hat'])/s_hat(q, M['m_B']))*WC['C7'])
        return A*ksi_ort(q, FF, M)
    else:
        print("Invalid chirality argument")

        
def A_par(q, chir, FF, WC, M, P):
    # https://arxiv.org/pdf/0807.2589.pdf
    # equ. 3.8
    WC['C9'] += Y(q, WC, M['m_b'], M['m_c'])
    if chir == 'L':
        A = -np.sqrt(2)*N(q, M, P)*M['m_B']*(1 - s_hat(q, M['m_B'])) * \
            (WC['C9'] - WC['C10'] + (2*M['m_b_hat']/s_hat(q, M['m_B']))*WC['C7'])
        return A*ksi_ort(q, FF, M)
    elif chir == 'R':
        A = -np.sqrt(2)*N(q, M, P)*M['m_B']*(1 - s_hat(q, M['m_B'])) * \
            (WC['C9'] + WC['C10'] + (2*M['m_b_hat']/s_hat(q, M['m_B']))*WC['C7'])
        return A*ksi_ort(q, FF, M)
    else:
        print("Invalid chirality argument")


def A_0(q, chir, FF, WC, M, P):
    # https://arxiv.org/pdf/0807.2589.pdf
    # equ. 3.9
    WC['C9'] += Y(q, WC, M['m_b'], M['m_c'])
    if chir == 'L':
        A = (-N(q, M, P)*M['m_B']**2)/(2.*M['m_Ks']*np.sqrt(s_hat(q, M['m_B']))) * \
            (1-s_hat(q, M['m_B']))**2 * \
            (WC['C9'] - WC['C10'] + (2*M['m_b_hat'])*WC['C7'])
        return A*ksi_par(q, FF, M)
    elif chir == 'R':
        A = (-N(q, M, P)*M['m_B']**2)/(2.*M['m_Ks']*np.sqrt(s_hat(q, M['m_B']))) * \
            (1-s_hat(q, M['m_B']))**2 * \
            (WC['C9'] + WC['C10'] + 2*M['m_b_hat']*WC['C7'])
        return A*ksi_par(q, FF, M)
    else:
        print("Invalid chirality argument")

        
def A_t(q, FF, WC, M, P):
    A = (N(q, M, P)*M['m_B']**2)/(M['m_Ks']*np.sqrt(s_hat(q, M['m_B']))) * \
        (1-s_hat(q, M['m_B']))**2*WC['C10']
    return A*ksi_par(q, FF, M)


###Angular obervables
def J_1s(q, FF, WC, M, P):
    J = ((2+beta_l(q, M['m_l'])**2)/4.) * \
        (np.absolute(A_ort(q, "L", FF, WC, M, P))**2 + \
         np.absolute(A_par(q, "L", FF, WC, M, P))**2 + \
         np.absolute(A_ort(q, "R", FF, WC, M, P))**2 + \
         np.absolute(A_par(q, "R", FF, WC, M, P))**2) + \
        ((4*(M['m_l']**2))/q) * (A_ort(q, "L", FF, WC, M, P) * \
                                 np.conj(A_ort(q, "R", FF, WC, M, P)) +  \
                                 A_par(q, "L", FF, WC, M, P) * \
                                 np.conj(A_par(q, "R", FF, WC, M, P))).real
    return J


def J_1c(q, FF, WC, M, P):
    J = np.absolute(A_0(q, "L", FF, WC, M, P))**2 + np.absolute(A_0(q, "R", FF, WC, M, P))**2 + \
        (4*(M['m_l']**2)/q) * (np.absolute(A_t(q, FF, WC, M, P))**2 + \
                          2*(A_0(q, "L", FF, WC, M, P)*np.conj(A_0(q, "R", FF, WC, M, P))).real)
    return J


def J_2s(q, FF, WC, M, P):
    J = ((beta_l(q, M['m_l'])**2)/4) * (np.absolute(A_ort(q, "L", FF, WC, M, P))**2 + \
                                        np.absolute(A_par(q, "L", FF, WC, M, P))**2 + \
                                        np.absolute(A_ort(q, "R", FF, WC, M, P))**2 + \
                                        np.absolute(A_par(q, "R", FF, WC, M, P))**2)
    return J


def J_2c(q, FF, WC, M, P):
    J = -(beta_l(q, M['m_l'])**2) * (np.absolute(A_0(q, "L", FF, WC, M, P))**2 + \
                                     np.absolute(A_0(q, "R", FF, WC, M, P))**2)
    return J


def J_5(q, FF, WC, M, P):
    J = np.sqrt(2)*beta_l(q, M['m_l']) * \
        ((A_0(q, "L", FF, WC, M, P)*np.conj(A_ort(q, "L", FF, WC, M, P))).real - \
         (A_0(q, "R", FF, WC, M, P)*np.conj(A_ort(q, "R", FF, WC, M, P))).real)
    return J


def J_1s_bar(q, FF, WC, M, P):
    J = ((2+beta_l(q, M['m_l'])**2)/4.) * (np.absolute(A_ort(q, "L", FF, WC, M, P))**2 + \
                                           np.absolute(A_par(q, "L", FF, WC, M, P))**2 + \
                                           np.absolute(A_ort(q, "R", FF, WC, M, P))**2 + \
                                           np.absolute(A_par(q, "R", FF, WC, M, P))**2) + \
        ((4*M['m_l']**2)/q) * (A_ort(q, "R", FF, WC, M, P) * \
                           np.conj(A_ort(q, "L", FF, WC, M, P)) + \
                           A_par(q, "R", FF, WC, M, P) * \
                           np.conj(A_par(q, "L", FF, WC, M, P))).real
    return J


def J_1c_bar(q, FF, WC, M, P):
    J = np.absolute(A_0(q, "L", FF, WC, M, P))**2 + \
        np.absolute(A_0(q, "R", FF, WC, M, P))**2 + \
        (4*M['m_l']**2/q)*(np.absolute(A_t(q, FF, WC, M, P))**2 + \
                      2*(A_0(q, "R", FF, WC, M, P) * \
                         np.conj(A_0(q,"L", FF, WC, M, P))).real)
    return J


def J_2s_bar(q, FF, WC, M, P):
    J = ((beta_l(q, M['m_l'])**2)/4)*(np.absolute(A_ort(q,"L", FF, WC, M, P))**2 + \
                            np.absolute(A_par(q,"L", FF, WC, M, P))**2 + \
                            np.absolute(A_ort(q,"R", FF, WC, M, P))**2 + \
                            np.absolute(A_par(q,"R" ,FF, WC, M, P))**2)
    return J

def J_2c_bar(q, FF, WC, M, P):
    J = -(beta_l(q, M['m_l'])**2)*(np.absolute(A_0(q, "L", FF, WC, M, P))**2 + \
                         np.absolute(A_0(q, "R", FF, WC, M, P))**2)
    return J

def J_5_bar(q, FF, WC, M, P):
    J = np.sqrt(2)*beta_l(q, M['m_l'])*((A_ort(q,"L", FF, WC, M, P) * \
                               np.conj(A_0(q,"L", FF, WC, M, P))).real - \
                              (A_ort(q,"R", FF, WC, M, P) * \
                               np.conj(A_0(q,"R", FF, WC, M, P))).real)
    return J


def DecayRate(q, FF, WC, M, P):
    gamma = 3*(2*J_1s(q, FF, WC, M, P) + J_1c(q, FF, WC, M, P))/4. - \
            (2*J_2s(q, FF, WC, M, P) + J_2c(q, FF, WC, M, P))/4.
    return gamma


def DecayRate_bar(q, FF, WC, M, P):
    gamma = 3*(2*J_1s_bar(q, FF, WC, M, P) + J_1c_bar(q, FF, WC, M, P))/4. - \
          (2*J_2s_bar(q, FF, WC, M, P) + J_2c_bar(q, FF, WC, M, P))/4.
    return gamma


def S5(q, FF, WC, M, P):
    s_5 = (J_5_bar(q, FF, WC, M, P) + J_5(q, FF, WC, M, P)) / \
          (DecayRate(q, FF, WC, M, P) + DecayRate_bar(q, FF, WC, M, P))
    return s_5


def FL(q, FF, WC, M, P):
    con = (np.absolute(A_0(q, "L", FF, WC, M, P))**2 + 
           np.absolute(A_0(q, "R", FF, WC, M, P))**2) / \
          (np.absolute(A_0(q, "L", FF, WC, M, P))**2 + \
           np.absolute(A_0(q, "R", FF, WC, M, P))**2 + \
           np.absolute(A_par(q, "L", FF, WC, M, P))**2 + \
           np.absolute(A_par(q, "R", FF, WC, M, P))**2 + \
           np.absolute(A_ort(q, "L", FF, WC, M, P))**2 + \
           np.absolute(A_ort(q, "R", FF, WC, M, P))**2)
    return con


def P_5_p(q, FF, WC, M, P):
    return(S5(q, FF, WC, M, P)/(np.sqrt(FL(q, FF, WC, M, P)*(1-FL(q, FF, WC, M, P)))))
