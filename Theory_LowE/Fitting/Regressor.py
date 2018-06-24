# http://danielhnyk.cz/creating-your-own-estimator-scikit-learn/
# http://scikit-learn.org/dev/developers/contributing.html#rolling-your-own-estimator
# https://stackoverflow.com/questions/29393739/how-to-use-gridsearchcv-with-custom-estimator-in-sklearn
import numpy as np
from inspect import getargvalues, currentframe
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.base import BaseEstimator, RegressorMixin
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

    def __init__(self, m_b=4.8, m_c=1.3, C7=-0.365, C9=4.334, C10=-4.513):
        """
        Called when initializing the classifier.
        Defines Hyper-Parameters to be explored.
        Input:
            m_b: mass of [GeV]
            m_c: mass of [GeV]
            C7, C9, C10: Wilson coefficiens
        """
        args, _, _, values = getargvalues(currentframe())
        values.pop("self")

        HP = {}
        for arg, val in values.items():
            setattr(self, arg, val)
            HP[arg] = val

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
        assert (type(self.m_c) == float); assert (type(self.C9) == float)
       
        # Hyper-Parameters
        HP = {'C7' : self.C7,
              'C9' : self.C9,
              'C10' : self.C10}

        # Load Wilson-Coefficients, Form-Factors, Mass, Extra-params
        WC, FF, M, Ex = p5e.load_constants('Settings.txt')
        # Iterate over energy ranges
        print('len of X', len(X))
        for value in X:
            self.p5p_res = p5e.P5p_SM(value, HP, WC, FF, M, Ex)

        return self

    def _meaning(self, x):
        print('in _meaning')
        # This function makes only sense in ClassifierMixi but not in 
        # RegressionMixi
        # returns True/False according to fitted classifier
        # notice underscore on the beginning
        return( True if x >= self.p5p_res else False )

    def predict(self, X, y=None):
        print('in predict')
        #try:
        #    getattr(self, "treshold_")
        #except AttributeError:
        #    raise RuntimeError("You must train classifer before predicting data!")
        return 
        #return([self._meaning(x) for x in X])

    #def score(self, X, y=None):
    #    # counts number of values bigger than mean
    #    print('in score')
    #    return(sum(self.predict(X))) 
