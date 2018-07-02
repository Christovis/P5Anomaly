import numpy as np
from numpy import ma
import scipy.integrate as integrate
from scipy.optimize import curve_fit
from pypet import Environment, cartesian_product
import Funcs as p5f


def load_constants(filename):
    data = open(filename, 'r')
    Settings = data.readlines()
    # Mass
    M = {'m_l' : 0.,
         'm_b' : 4.2,
         'm_c' : 1.3,
         'mu_b' : 4.8,
         'm_B' : 5.27950,
         'm_Ks' : 0.895}
    Ex= {'C_F' : 1.333333,
         'alpha_s_h' : 0.214,
         'alpha_s_b' : 0.214,
         'alpha_em' : 0.00729927007,
         'V_tbV_ts' : 0.0385,
         'G_f' : 0.00001166378}
    WC = {'C1' : -0.257,
          'C2' : 1.009,
          'C3' : -0.005,
          'C4' : -0.078,
          'C5' : 0.,
          'C6' : 0.001,
          'C7_eff' : -0.317,
          'dC7' : 0.1,
          'C9' : 4.1,
          'dC9' : .1,
          'C10' : -4.308,
          'dC10' : .1}
    # Form Factor
    FF = {'f_B' : 0.18,
          'f_Ks' : 0.225,
          'f_Ks_ort' : 0.185,
          'lambda_B_p' : 3}
    return WC, FF, M, Ex


def P5p_SM(X, HP, WC, FF, M, Ex):
    """
    Input:
        X: engery [GeV]
        HP: hyper-paramets
        WC: Wilson coeff.
        FF: form factor
        M: masses
        CO: corrections
        Ex: extra params.
    """
    Dicts = {'HP' : HP,
             'WC' : WC,
             'FF' : FF,
             'M' : M,
             'Ex' : Ex}

    J5_bin = integrate.quad(p5f.J_5_, X[0], X[1], args=(Dicts))
    c0_bin = integrate.quad(p5f.c_0, X[0], X[1], args=(Dicts))
    c4_bin = integrate.quad(p5f.c_4, X[0], X[1], args=(Dicts))
    P5p = p5f.P5p(J5_bin, c0_bin, c4_bin)
    return P5p


def P5p_NP():
    # central values
    res = np.zeros(len(bins))
    res_max = np.zeros(len(bins))
    res_min = np.zeros(len(bins))
    for b in range(len(bins)):
        min = bins[b][0]
        max = bins[b][1]
        J5_bin = integrate.quad(p5f.J_5_, min, max, args=(FormFac, NP_WC, Mass, Param))
        c0_bin = integrate.quad(p5f.c_0, min, max, args=(FormFac, NP_WC, Mass, Param))
        c4_bin = integrate.quad(p5f.c_4, min, max, args=(FormFac, NP_WC, Mass, Param))
        P5p_bin = p5f.P5p(J5_bin, c0_bin, c4_bin)
        res[b] = P5p_bin
    return res_sm


#def load_constants(filename):
#    data = open(filename, 'r')
#    Settings = data.readlines()
#    # Mass
#    M = {'m_l' : float(Settings[0].split()[1]),
#         'm_b' : float(Settings[1].split()[1]),
#         'm_c' : float(Settings[2].split()[1]),
#         'm_B' : float(Settings[3].split()[1]),
#         'm_Ks' : float(Settings[4].split()[1])}
#    Ex= {'alpha_em' : float(Settings[5].split()[1]),
#         'V_tbV_ts' : float(Settings[6].split()[1]),
#         'G_f' : float(Settings[7].split()[1])}
#    WC = {'C1' : float(Settings[8].split()[1]),
#          'C2' : float(Settings[9].split()[1]),
#          'C3' : float(Settings[10].split()[1]),
#          'C4' : float(Settings[11].split()[1]),
#          'C5' : float(Settings[12].split()[1]),
#          'C6' : float(Settings[13].split()[1]),
#          'C7_eff' : float(Settings[14].split()[1]),
#          'C7_eff_p' : float(Settings[15].split()[1]),
#          'dC7' : float(Settings[16].split()[1]),
#          'C9' : float(Settings[17].split()[1]),
#          'C9_eff_p' : float(Settings[18].split()[1]),
#          'dC9' : float(Settings[19].split()[1]),
#          'C10' : float(Settings[20].split()[1]),
#          'C10_p' : float(Settings[21].split()[1]),
#          'dC10' : float(Settings[22].split()[1])}
#    # Form Factor
#    FF = {'F_0' : [float(Settings[23].split()[1]),
#                   float(Settings[23].split()[2]),
#                   float(Settings[23].split()[3])],
#          'c1' : [float(Settings[24].split()[1]),
#                  float(Settings[24].split()[2]),
#                  float(Settings[24].split()[3])],
#          'c2' : [float(Settings[25].split()[1]),
#                  float(Settings[25].split()[2]),
#                  float(Settings[25].split()[3])],
#          'c3' : [float(Settings[26].split()[1]),
#                  float(Settings[26].split()[2]),
#                  float(Settings[26].split()[3])]}
#    return M, Ex, WC, FF
