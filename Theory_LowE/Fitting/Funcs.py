#from P5p_anomaly import *
import numpy as np
import FormFactors as ff
import Amplitudes as amp


def s_hat(q, m_B):
    #print('q', q)
    shat = q/m_B**2
    #print('s_hat', shat)
    return shat


###Angular obervables
def J_1s(q, Dicts, cmplx):
    # https://arxiv.org/pdf/1005.0571.pdf
    # equ. 2.4a
    M = Dicts['M']
    m_l = M['m_l']
    if cmplx == 'bar':
        J = (2+amp.beta_l(q, m_l)**2)/4. * \
            (np.absolute(amp.A_ort(q, "L", Dicts, cmplx))**2 + \
             np.absolute(amp.A_par(q, "L", Dicts, cmplx))**2 + \
             np.absolute(amp.A_ort(q, "R", Dicts, cmplx))**2 + \
             np.absolute(amp.A_par(q, "R", Dicts, cmplx))**2) + \
             ((4* (m_l)**2)/q) * (amp.A_ort(q, "R", Dicts, cmplx) * \
                                  np.conj(amp.A_ort(q, "L", Dicts, cmplx)) + \
                                  amp.A_par(q, "R", Dicts, cmplx) * \
                                  np.conj(amp.A_par(q, "L", Dicts, cmplx))).real
        return J
    else:
        J = (2+amp.beta_l(q, m_l)**2)/4. * \
            (np.absolute(amp.A_ort(q, "L", Dicts, cmplx))**2 + \
             np.absolute(amp.A_par(q, "L", Dicts, cmplx))**2 + \
             np.absolute(amp.A_ort(q, "R", Dicts, cmplx))**2 + \
             np.absolute(amp.A_par(q, "R", Dicts, cmplx))**2) + \
             ((4*(m_l**2))/q) * (amp.A_ort(q, "L", Dicts, cmplx) * \
                                 np.conj(amp.A_ort(q, "R", Dicts, cmplx)) + \
                                 amp.A_par(q, "L", Dicts, cmplx) * \
                                 np.conj(amp.A_par(q, "R", Dicts, cmplx))).real
        return J

def J_1c(q, Dicts, cmplx):
    # https://arxiv.org/pdf/1005.0571.pdf
    # equ. 2.4b
    M = Dicts['M']
    m_l = M['m_l']
    if cmplx == 'bar':
        J = np.absolute(amp.A_0(q, "L", Dicts, cmplx))**2 +\
            np.absolute(amp.A_0(q, "R", Dicts, cmplx))**2 + \
            (4* (m_l**2)/q)*(np.absolute(amp.A_t(q, Dicts, cmplx))**2 + \
                          2*(amp.A_0(q, "R", Dicts, cmplx)*\
                             np.conj(amp.A_0(q, "L", Dicts, cmplx))).real)
        return J
    else:
        J = np.absolute(amp.A_0(q, "L", Dicts, cmplx))**2 +\
            np.absolute(amp.A_0(q, "R", Dicts, cmplx))**2 + \
            (4*(m_l**2)/q) * (np.absolute(amp.A_t(q, Dicts, cmplx))**2 + \
                              2*(amp.A_0(q, "L", Dicts, cmplx)* \
                                 np.conj(amp.A_0(q, "R", Dicts, cmplx))).real)
        return J


def J_2s(q, Dicts, cmplx):
    # https://arxiv.org/pdf/1005.0571.pdf
    # equ. 2.4c
    M = Dicts['M']
    if cmplx == 'bar':
        J = ((amp.beta_l(q, M['m_l'])**2)/4) * \
            (np.absolute(amp.A_ort(q, "L", Dicts, cmplx))**2 + \
             np.absolute(amp.A_par(q, "L", Dicts, cmplx))**2 + \
             np.absolute(amp.A_ort(q, "R", Dicts, cmplx))**2 + \
             np.absolute(amp.A_par(q, "R", Dicts, cmplx))**2)
        return J
    else:
        J = ((amp.beta_l(q, M['m_l'])**2)/4) * \
            (np.absolute(amp.A_ort(q, "L", Dicts, cmplx))**2 + \
             np.absolute(amp.A_par(q, "L", Dicts, cmplx))**2 + \
             np.absolute(amp.A_ort(q, "R", Dicts, cmplx))**2 + \
             np.absolute(amp.A_par(q, "R", Dicts, cmplx))**2)
        return J


def J_2c(q, Dicts, cmplx):
    # https://arxiv.org/pdf/1005.0571.pdf
    # equ. 2.4d
    M = Dicts['M']
    if cmplx == 'bar':
        J = -(amp.beta_l(q, M['m_l'])**2) * \
            (np.absolute(amp.A_0(q, "L", Dicts, cmplx))**2 + \
             np.absolute(amp.A_0(q, "R", Dicts, cmplx))**2)
        return J
    else: 
        J = -(amp.beta_l(q, M['m_l'])**2) * \
            (np.absolute(amp.A_0(q, "L", Dicts, cmplx))**2 + \
             np.absolute(amp.A_0(q, "R", Dicts, cmplx))**2)
        return J


def J_5(q, Dicts, cmplx):
    # https://arxiv.org/pdf/1005.0571.pdf
    # equ. 2.4g
    M = Dicts['M']
    if cmplx == 'bar':
        J = np.sqrt(2)*amp.beta_l(q, M['m_l']) * \
            ((amp.A_ort(q, "L", Dicts, cmplx) * \
              np.conj(amp.A_0(q, "L", Dicts, cmplx))).real - \
             (amp.A_ort(q, "R", Dicts, cmplx) * \
              np.conj(amp.A_0(q, "R", Dicts, cmplx))).real)
        return J
    else:
        J = np.sqrt(2)*amp.beta_l(q, M['m_l']) * \
            ((amp.A_0(q, "L", Dicts, cmplx) * \
              np.conj(amp.A_ort(q, "L", Dicts, cmplx))).real - \
             (amp.A_0(q, "R", Dicts, cmplx)*\
              np.conj(amp.A_ort(q, "R", Dicts, cmplx))).real)
        return J


def DecayRate(q, Dicts, cmplx):
    gamma = 3*(2*J_1s(q, Dicts, cmplx) + J_1c(q, Dicts, cmplx))/4. - \
            (2*J_2s(q, Dicts, cmplx) + J_2c(q, Dicts, cmplx))/4.
    return gamma


def S5(q, FF, WC, M, Ex):
    # https://arxiv.org/pdf/1207.2753.pdf
    # footnote 2, page 7
    s_5 = (J_5(q, Dicts, 'real') + J_5(q, Dicts, 'bar'))/ \
          (DecayRate(q, Dicts, 'real') + DecayRate(q, Dicts, 'bar'))
    return s_5


def S2_c(q, Dicts):
    s = (J_2c(q, Dicts, 'real') + J_2c(q, Dicts, 'bar'))/ \
        (DecayRate(q, Dicts, 'real') + DecayRate(q, Dicts, 'bar'))
    return s


def FL(q, Dicts):
    fl = -S2_c(q, Dicts)
    return fl


def c_0(q, Dicts):
    c0 = DecayRate(q, Dicts, 'real') + DecayRate(q, Dicts, 'bar')
    return c0

def c_4(q, Dicts):
    c4 = (1. - FL(q, Dicts)) * \
         (DecayRate(q, Dicts, 'real') + DecayRate(q, Dicts, 'bar'))
    return c4


def J_5_(q, Dicts):
    """
    Input:
        q: x
        dicts: HP, WC, FF, M, CO, Ex
    """
    j5 = J_5(q, Dicts, 'real') + J_5(q, Dicts, 'bar')
    return j5


def P5p(J5_bin, c0_bin, c4_bin):
    P5_p = J5_bin[0]/np.sqrt(c4_bin[0]*(c0_bin[0] - c4_bin[0]))
    return P5_p

