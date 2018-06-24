import numpy as np
import scipy.integrate as integrate
import FormFactors as ff

global a_FF_par, Lmb_CO_par
a_FF_par = {'a_par' : [ 0.03, 0.08, -0.03, 0.08],
            'a_ort' : [ 0.03, 0.08, -0.03, 0.08]}
#O(Lambda/m_b) corrections
# a_F, b_F, c_F in KMPW scheme from arXiv 1407.8526v2
Lmb_CO_par = {'V'  : [0.,     0.,    0.],
              'A0' : [0.002, 0.590, 1.473],
              'A1' : [-0.013, -0.056, 0.158],
              'A2' : [-0.018, -0.105, 0.192],
              'T1' : [-0.006, -0.012, -0.034],
              'T2' : [-0.005, 0.153, 0.544],
              'T3' : [-0.002, 0.308, 0.786]}


def Phi_par(u, par, cmplx):
    i = 0
    if cmplx == 'bar':
        i = 2
    phi = (6*u*(1-u)*(1 + 3*par['a_par'][0+i]*(2*u-1) + \
           par['a_par'][1+i]*3/2 * (5*(2*u-1)**2-1)))/(1-u)
    return phi


def Phi_ort(u, par, cmplx):
    i = 0
    if cmplx == 'bar':
        i = 2
    phi = (6*u*(1-u)*(1 + 3*par['a_ort'][0+i]*(2*u - 1) + \
           par['a_ort'][1+i]*3/2*(5*(2*u - 1)**2 - 1)))/(1 - u)
    return phi


def factor_par(cmplx):
    return integrate.quad(Phi_par, 0, 1, args=(a_FF_par, cmplx))


def factor_ort(cmplx):
    return integrate.quad(Phi_ort, 0, 1, args=(a_FF_par, cmplx))


def deltaF_par(f_B, f_Ks, m_B, lambda_B_p, cmplx):
    return  8*(np.pi**2)*f_B*f_Ks/(3*m_B)*(factor_par(cmplx)[0]*lambda_B_p)


def deltaF_ort(f_B, f_Ks_ort, m_B, lambda_B_p, cmplx):
    return 8*(np.pi**2)*f_B*f_Ks_ort/(3*m_B)*(factor_ort(cmplx)[0]*lambda_B_p)


def deltaT1(q, M, FF, Ex, cmplx):
    d = M['m_B']/(4*ff.E_Ks(q, M['m_B'])) * \
        deltaF_ort(FF['f_B'], FF['f_Ks_ort'], M['m_B'], FF['lambda_B_p'], cmplx)
    return d


def deltaT2(q, FF, M, cmplx):
    d = 1/2*deltaF_ort(FF['f_B'], FF['f_Ks_ort'], M['m_B'], FF['lambda_B_p'], cmplx)
    return d


def deltaT3(q, FF, M, Ex, cmplx):
    d = deltaT1(q, M, FF, Ex, cmplx) + \
        2*M['m_Ks']/(M['m_B'])*(M['m_B']/(2*ff.E_Ks(q, M['m_B'])))**2 * \
        deltaF_par(FF['f_B'], FF['f_Ks'], M['m_B'], FF['lambda_B_p'], cmplx)
    return d


def L(q, m_B):
    l = -2*ff.E_Ks(q, m_B)/(m_B - 2*ff.E_Ks(q, m_B)) * \
        np.log(2*ff.E_Ks(q, m_B)/m_B)
    return l


def Delta(q, FF, M, Ex, cmplx):
    d = 1 + Ex['alpha_s_b']*Ex['C_F']/(4*np.pi) * (-2 + 2*L(q, M['m_B'])) - \
        Ex['alpha_s_b']*Ex['C_F']*2*q/((ff.E_Ks(q, M['m_B'])**2)*(np.pi)) * \
        np.pi**2 * M['m_Ks']*FF['f_B']*FF['f_Ks']*FF['lambda_B_p']/ \
        (3*M['m_B']*ff.E_Ks(q, M['m_B'])*ff.ksi_par(q, M['m_B'])) * \
        factor_par(cmplx)[0]
    return d
    
def Delta_V(q):
    return(0.)


def Delta_A1(q):
    return(0.)


def Delta_A2(q):
    return(0.)


def Delta_A0(q, FF, M, Ex, cmplx):
    d = (ff.E_Ks(q, M['m_B'])/M['m_Ks'])*ff.ksi_par(q, M['m_B']) * \
        (Delta(q, FF, M, Ex, cmplx)**-1 - 1)
    return d


def Delta_T1(q, FF, M, Ex, cmplx):
    d = Ex['C_F']*Ex['alpha_s_b']*ff.ksi_ort(q, M['m_B']) * \
        (np.log(M['m_b']**2/M['mu_b']**2) - L(q, M['m_B'])) + \
        Ex['C_F']*Ex['alpha_s_b']*deltaT1(q, M, FF, Ex, cmplx)
    return d


def Delta_T2(q, FF, M, Ex, cmplx):
    d = Ex['C_F']*Ex['alpha_s_b']*2*ff.E_Ks(q, M['m_B'])/(M['m_B']) * \
        ff.ksi_ort(q, M['m_B'])*(np.log(M['m_b']**2/M['mu_b']**2) - L(q, M['m_B'])) + \
        Ex['C_F']*Ex['alpha_s_h']*deltaT2(q, FF, M, cmplx)
    return d


def Delta_T3(q, FF, M, Ex, cmplx):
    d = Ex['C_F']*Ex['alpha_s_b']*(ff.ksi_ort(q, M['m_B']) * \
        (np.log(M['m_b']**2/M['mu_b']**2) - L(q, M['m_B'])) - \
        ff.ksi_par(q, M['m_B'])*(np.log(M['m_b']**2/M['mu_b']**2) + \
        2*L(q, M['m_B']))) + Ex['C_F']*Ex['alpha_s_h']*deltaT3(q, FF, M, Ex, cmplx)
    return d


def Delta_lmb(q, par, m_B):
    res = []
    for i in ['V', 'A1', 'A2', 'A0', 'T1', 'T2', 'T3']:
        val = par[i][0] + par[i][1]*(q/(m_B**2)) + par[i][2]*(q**2/(m_B**4))
        res.append(val)
    return(res)


CO = {'V' : [Delta_V,  Delta_lmb],
      'A1': [Delta_A1, Delta_lmb],
      'A2': [Delta_A2, Delta_lmb],
      'A0': [Delta_A0, Delta_lmb],
      'T1': [Delta_T1, Delta_lmb],
      'T2': [Delta_T2, Delta_lmb],
      'T3': [Delta_T3, Delta_lmb]}
