# http://danielhnyk.cz/creating-your-own-estimator-scikit-learn/
# http://scikit-learn.org/dev/developers/contributing.html#rolling-your-own-estimator
# https://stackoverflow.com/questions/29393739/how-to-use-gridsearchcv-with-custom-estimator-in-sklearn
import numpy as np
#from inspect import getargvalues, currentframe
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import euclidean_distances
import Experiments as p5e


class P5p_Regressor(BaseEstimator, RegressorMixin):  
    """
    Regression Classifier for Wilson coeff. tuning to predict P5_prime observable

    Note:
        Check if costum classifier works with
            from sklearn.utils.estimator_checks import check_estimator
            check_estimator(P5pClassifier)
    Input:
        BaseEstimator: defines set_param, get_param
        RegressorMixin: defines score
    """

    def __init__(self, C7='C7', C9='C9', C10='C10'):
    #def __init__(self, C7=-0.365, C9=4.334, C10=-4.513):
        """
        Called when initializing the classifier.
        Defines Hyper-Parameters to be explored.
        Input:
            m_b: mass of [GeV]
            m_c: mass of [GeV]
            C7, C9, C10: Wilson coefficiens
        """
        #args, _, _, values = getargvalues(currentframe())
        #values.pop("self")
        #for arg, val in values.items():
        #    setattr(self, arg, val)
        self.C7 = C7
        self.C9 = C9
        self.C10 = C10


    def fit(self, X, y=None):  #, **fit_params):
        """
        Fit Regressor
        Note: assert is not a good choice here and you should rather
        use try/except blog with exceptions. This is just for short syntax.

        # Here the function to be fitting is defined
        # read fixed values (e.g. C1, C2, ..., C6)
        
        Input:
            X, y: energy, P5p-observable

        Output:
            self
        """
        assert (type(self.C7) == float)
        assert (type(self.C9) == float)
        assert (type(self.C10) == float)
        # Check that X and y have correct shape
        X, y = check_X_y(X, y, dtype=np.float)
        self.X_ = X
        self.y_ = y

        # Hyper-Parameters
        HP = {'C7' : self.C7,
              'C9' : self.C9,
              'C10' : self.C10}

        # Load Wilson-Coefficients, Form-Factors, Mass, Extra-params
        WC, FF, M, Ex = p5e.load_constants('Settings.txt')
        # Iterate over energy ranges
        self.p5p_ = np.zeros(len(X))
        for i in range(len(X)):
            self.p5p_[i] = p5e.P5p_SM(X[i], HP, WC, FF, M, Ex)
        return self


    def predict(self, X):
        # Check if fit had been called
        check_is_fitted(self, ['p5p_'])
        # Input validation
        X = check_array(X)
        #closest = np.argmin(euclidean_distances(X, self.X_), axis=1)
        return self.p5p_[closest]


    #def score(self, X, y=None):
    #    # counts number of values bigger than mean
    #    print('in score')
    #    return(sum(self.predict(X))) 
