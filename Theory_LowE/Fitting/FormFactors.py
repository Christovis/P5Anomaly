import numpy as np
from Corrections import CO
global ksi_ort_0, ksi_par_0, Lmb_CO_par
ksi_ort_0 = 0.266  # +- 0.032 (Straub et. al.)
ksi_par_0 = 0.118  # +- 0.008 (same)
Lmb_CO_par = {'V'  : [0.,     0.,    0.],
             'A0' : [0.002, 0.590, 1.473],
             'A1' : [-0.013, -0.056, 0.158],
             'A2' : [-0.018, -0.105, 0.192],
             'T1' : [-0.006, -0.012, -0.034],
             'T2' : [-0.005, 0.153, 0.544],
             'T3' : [-0.002, 0.308, 0.786]}


# In the low q regime (q<< M['m_B']^2) we have: NB we use q in place of q^2.
def E_Ks(q, m_B):
    E = (m_B/2) * (1-(q/(m_B**2)))
    return E


# Form Factor ksi's
def ksi_ort(q, m_B):
    #from arXiv hep-ph/0106067v2
    ksi = ksi_ort_0*(1/(1 - q/(m_B**2)))
    return ksi


def ksi_par(q, m_B):
    #from arXiv hep-ph/0106067v2
    ksi = ksi_par_0*(1/(1 - q/(m_B**2)))**3
    return ksi


def V(q, M, cmplx):
    V = (M['m_B'] + M['m_Ks'])/M['m_B'] * ksi_ort(q, M['m_B']) + \
        CO['V'][0](q) + CO['V'][1](q, Lmb_CO_par, M['m_B'])[0]
    return V  


def A1(q, M, cmplx):
    assert ('m_B' in M); assert ('m_Ks' in M)
    assert ('V' in CO); assert ('A0' in CO)
    a = (2*E_Ks(q, M['m_B']))/(M['m_B'] + M['m_Ks'])*ksi_ort(q, M['m_B']) + \
        CO['A1'][0](q) + CO['A1'][1](q, Lmb_CO_par, M['m_B'])[1]
    return a


def A2(q, M, cmplx):
    a = M['m_B']/(M['m_B']-M['m_Ks']) * \
        (ksi_ort(q, M['m_B']) -  ksi_par(q, M['m_B'])) + \
        CO['A2'][0](q) + CO['A2'][1](q, Lmb_CO_par, M['m_B'])[2]
    return a


def A0(q, FF, M, Ex, cmplx):
    a = (E_Ks(q, M['m_B'])/M['m_Ks'])*ksi_par(q, M['m_B']) + \
        CO['A0'][0](q, FF, M, Ex, cmplx) + \
        CO['A0'][1](q, Lmb_CO_par, M['m_B'])[3]
    return a


def T1(q, FF, M, Ex, cmplx):
    t = ksi_ort(q, M['m_B']) + \
        CO['T1'][0](q, FF, M, Ex, cmplx) + \
        CO['T1'][1](q, Lmb_CO_par, M['m_B'])[4]
    return t


def T2(q, FF, M, Ex, cmplx):
    t = (2*E_Ks(q, M['m_B']))/M['m_B']*ksi_ort(q, M['m_B']) + \
        CO['T2'][0](q, FF, M, Ex, cmplx) + \
        CO['T2'][1](q, Lmb_CO_par, M['m_B'])[5]
    return t


def T3(q, FF, M, Ex, cmplx):
    t = ksi_ort(q, M['m_B']) - ksi_par(q, M['m_B']) + \
        CO['T3'][0](q, FF, M, Ex, cmplx) + \
        CO['T3'][1](q, Lmb_CO_par, M['m_B'])[6]
    return t
