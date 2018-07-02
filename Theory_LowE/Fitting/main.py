from time import time
import numpy as np
from sklearn import datasets
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import classification_report
from scipy.stats import randint as sp_randint
from Rand_regr import P5p_Regressor as P5R
import Experiments as p5e
import matplotlib.pyplot as plt

###############################################################################
# Specify parameters and distributions to sample from

hp = {'C7' : [-0.378, -0.365, -0.351],
      'C9' : [4.324, 4.334, 4.344],
      'C10' : [-4.503, -4.513, -4.523]}
#param_explore = {'C7' : sp_randint(378, 351)*(-0.001),
#                 'C9' : sp_randint(4324, 4344)*0.001,
#                 'C10' : sp_randint(4503, 4523)*(-0.0001)}

###############################################################################
# Data bins
bins = np.array([[0.1, 0.98], [1.1, 2.5], [2.5, 4], [4, 6.]])
#bins = np.array([[.0, .0], [.0, .0], [0.1, 0.98], [1.1, 2.5], [2.5, 4], [4, 6.]])
bins_lim = np.array([.1])
for i in range(len(bins)):
    bins_lim = np.append(bins_lim, bins[i][1])

bins.tolist() #needed for Flavio th-prediction
bins=[tuple(entry) for entry in bins]

###############################################################################
# LHC Data
#LHC_x = [0.25, 1.9, 3.2, 4.8]     #[GeV]
LHC_y = [0.387, 0.289, -0.066, -0.3]  #[P5p]
#LHC_y = [0, 0, 0.387, 0.289, -0.066, -0.3]  #[P5p]

###############################################################################
# run randomized search
WC, FF, M, Ex = p5e.load_constants('Settings.txt')

rand = True
if rand:
    rs = RandomizedSearchCV(estimator=P5R(),
                            #fit_params={'WC':WC, 'FF':FF, 'M':M, 'Ex':Ex},
                            param_distributions=hp)
                            #scoring="accuracy",
                            #n_iter=10)
else:
    rs = GridSearchCV(estimator=P5R(),
                            #fit_params={'WC':WC, 'FF':FF, 'M':M, 'Ex':Ex},
                            param_grid=hp,
                            scoring="accuracy")
start = time()
print('Estimator status:')
print('     ', rs.get_params())
rs.fit(bins, LHC_y)  # fit bins to LHCb P5p
print("RandomizedSearchCV took %.2f seconds" % (time() - start))
print("Best parameters set found on development set:")
print("        ", rs.best_params_)
print("Best score:")
print("        ", rs.best_score_)
print("They give the following P5p:")
print("        ", fit_y)


###############################################################################
# Create Figures

