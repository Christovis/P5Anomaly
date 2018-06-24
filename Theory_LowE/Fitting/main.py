from time import time
import numpy as np
from sklearn import datasets
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report
from scipy.stats import randint as sp_randint
from Regressor import P5p_Regressor as P5R

###############################################################################
# Specify parameters and distributions to sample from

param_explore = {'m_b' : [4.6, 4.8 , 5.],
                 'm_c' : [1.27, 1.3, 1.32],
                 'C7' : [-0.378, -0.365, -0.351],
                 'C9' : [4.324, 4.334, 4.344],
                 'C10' : [-4.503, -4.513, -4.523]}
#param_explore = {'m_b' : sp_randint(48,50)/10,
#                 'm_c' : sp_randint(127, 132)*0.01,
#                 'C7' : sp_randint(378, 351)*(-0.001),
#                 'C9' : sp_randint(4324, 4344)*0.001,
#                 'C10' : sp_randint(4503, 4523)*(-0.0001)}

###############################################################################
# Data bins
bins = np.array([[0.1, 2.0], [2.0, 4.3], [4.3, 6]])
bins_lim = np.array([0.1])
for i in range(len(bins)):
    bins_lim = np.append(bins_lim, bins[i][1])

bins.tolist() #needed for Flavio th-prediction
bins=[tuple(entry) for entry in bins]

###############################################################################
# LHC Data
LHC_x = [0.25, 1.9, 3.2, 4.8]     #[GeV]
LHC_y = [0.38, -0.3, 0.0, -0.25]  #[P5p]

###############################################################################

# run randomized search
rs = RandomizedSearchCV(P5R(),
                        param_distributions=param_explore)
#print('Randomized Search Parameters: \n', rs.param_distributions)
#print('\n')

start = time()
rs.fit(bins, LHC_y[:3])  # fit bins to LHCb P5p
print("RandomizedSearchCV took %.2f seconds"
      " parameter settings." % (time.time() - start))
print("Best parameters set found on development set:")
print(random_search.best_params_)
