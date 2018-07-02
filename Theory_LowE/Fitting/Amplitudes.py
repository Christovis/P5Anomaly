##Amplitudes (arXiv 0807.2589v3)
import numpy as np
import FormFactors as ff
##Factors needed for the amplitudes


# Functions needed for C9_eff(q dC7)
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


def lmb(q, m_Ks, m_B):
    return(m_B**4 + m_Ks**4 + q**2 - 2*((m_B*m_Ks)**2 + q*(m_Ks**2) + q*(m_B**2)))


def beta_l(q, m_l):
    #print('Amplitude beta_l', q)
    return(np.sqrt(1 - (4* m_l**2)/q))


def N(q, M, Ex):
    con = (np.sqrt(Ex['G_f']**2*Ex['alpha_em']**2*q * \
                   np.sqrt(lmb(q, M['m_Ks'], M['m_B']))*beta_l(q, M['m_l']) / \
                   (3* (2.**10)*(np.pi)**5 * M['m_B']**3)))*Ex['V_tbV_ts']
    return con


def A_ort(q, chir, Dicts, cmplx):
    # https://arxiv.org/pdf/0807.2589.pdf
    # equ. 3.7
    # Y() causes deviations to results from paper
    HP=Dicts['HP']; FF=Dicts['FF']; WC=Dicts['WC']
    M=Dicts['M']; Ex=Dicts['Ex']
    if chir == 'L':
        res = np.sqrt(2*lmb(q, M['m_Ks'], M['m_B']))*N(q, M, Ex) * \
              ((HP['C9'] - HP['C10'] + Y(q, WC, M['m_b'], M['m_c'])) * \
               ff.V(q, M, cmplx)/(M['m_B'] + M['m_Ks']) + \
               ((2*M['m_b'])/q)*HP['C7']*ff.T1(q, FF, M, Ex, cmplx))
        return res
    elif chir == 'R':
        res = np.sqrt(2*lmb(q, M['m_Ks'], M['m_B']))*N(q, M, Ex) * \
              ((HP['C9'] + HP['C10'] + Y(q, WC, M['m_b'], M['m_c'])) * \
               ff.V(q, M, cmplx)/(M['m_B'] + M['m_Ks']) + \
               ((2*M['m_b'])/q)*HP['C7']*ff.T1(q, FF, M, Ex, cmplx))
        return res
    else:
        print("Invalid chirality argument")


def A_par(q, chir, Dicts, cmplx):
    # https://arxiv.org/pdf/0807.2589.pdf
    # equ. 3.8
    HP=Dicts['HP']; FF=Dicts['FF']; WC=Dicts['WC']
    M=Dicts['M']; Ex=Dicts['Ex']
    if chir == 'L':
        res = -np.sqrt(2)*N(q, M, Ex)*(M['m_B']**2 - M['m_Ks']**2) * \
              ((HP['C9'] - HP['C10'] + Y(q, WC, M['m_b'], M['m_c'])) * \
               ff.A1(q, M, cmplx)/(M['m_B'] - M['m_Ks']) + \
              (2*M['m_b']/q)*HP['C7']*ff.T2(q, FF, M, Ex, cmplx))
        return res
    elif chir == 'R':
        res = -np.sqrt(2)*N(q, M, Ex)*(M['m_B']**2 - M['m_Ks']**2)* (\
              (HP['C9'] + HP['C10'] + Y(q, WC, M['m_b'], M['m_c'])) * \
               ff.A1(q, M, cmplx)/(M['m_B'] - M['m_Ks']) +\
              ((2*M['m_b'])/q) * HP['C7'] * ff.T2(q, FF, M, Ex, cmplx))
        return res
    else:
        print("Invalid chirality argument")


def A_0(q, chir, Dicts, cmplx):
    # https://arxiv.org/pdf/0807.2589.pdf
    # equ. 3.9
    HP=Dicts['HP']; FF=Dicts['FF']; WC=Dicts['WC']
    M=Dicts['M']; Ex=Dicts['Ex']
    if chir == 'L':
        res = -N(q, M, Ex)/(2.*M['m_Ks']*np.sqrt(q)) * \
              ((HP['C9'] - HP['C10'] + Y(q, WC, M['m_b'], M['m_c'])) * \
               ((M['m_B']**2 - M['m_Ks']**2 - q) * \
                (M['m_B'] + M['m_Ks'])*ff.A1(q, M, cmplx) - \
                lmb(q, M['m_Ks'], M['m_B']) * \
                ff.A2(q, M, cmplx)/(M['m_B'] + M['m_Ks'])) + \
               (2*M['m_b'])*HP['C7'] * \
               ((M['m_B']**2 + 3*M['m_Ks']**2-q)*ff.T2(q, FF, M, Ex, cmplx) - \
                (lmb(q, M['m_Ks'], M['m_B'])/(M['m_B']**2-M['m_Ks']**2)) * \
                ff.T3(q, FF, M, Ex, cmplx))) 
        return res
    elif chir == 'R':
        res = -N(q, M, Ex)/(2.*M['m_Ks']*np.sqrt(q)) * \
              ((HP['C9'] + HP['C10'] + Y(q, WC, M['m_b'], M['m_c'])) * \
               ((M['m_B']**2 - M['m_Ks']**2 - q) * \
                (M['m_B'] + M['m_Ks'])*ff.A1(q, M, cmplx) - \
                lmb(q, M['m_Ks'], M['m_B']) * \
                ff.A2(q, M, cmplx)/(M['m_B']+ M['m_Ks'])) + \
               (2*M['m_b'])*HP['C7'] * \
               ((M['m_B']**2 + 3*M['m_Ks']**2-q)*ff.T2(q, FF, M, Ex, cmplx) - \
                (lmb(q, M['m_Ks'], M['m_B'])/(M['m_B']**2-M['m_Ks']**2)) * \
                ff.T3(q, FF, M, Ex, cmplx))) 
        return res
    else:
        print("Invalid chirality argument")

        
def A_t(q, Dicts, cmplx):
    HP=Dicts['HP']; FF=Dicts['FF']; WC=Dicts['WC']
    M=Dicts['M']; Ex=Dicts['Ex']
    a = N(q, M, Ex)*np.sqrt(lmb(q, M['m_Ks'], M['m_B']))/np.sqrt(q) * \
        2*HP['C10']*ff.A0(q, FF, M, Ex, cmplx)
    return a
