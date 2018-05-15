from keras import losses
from keras import backend as K

import settings


################################################################################
# Loss functions
################################################################################

def loss_function_carl(y_true, y_pred):
    return losses.binary_crossentropy(y_true[:, 0], y_pred[:, 0])


def loss_function_ratio_regression(y_true, y_pred):
    r_loss = losses.mean_squared_error(K.exp(K.clip(y_true[:, 1], -10., 10.)),
                                       K.exp(K.clip(y_pred[:, 1], -10., 10.)))
    inverse_r_loss = losses.mean_squared_error(K.exp(-K.clip(y_true[:, 1], -10., 10.)),
                                               K.exp(-K.clip(y_pred[:, 1], -10., 10.)))
    return y_true[:, 0] * r_loss + (1. - y_true[:, 0]) * inverse_r_loss


def loss_function_score(y_true, y_pred):
    score_loss = losses.mean_squared_error(y_true[:, 2:settings.n_params + 2],
                                           y_pred[:, 2:settings.n_params + 2])
    return (1. - y_true[:, 0]) * score_loss


def loss_function_combined(y_true, y_pred, alpha=0.1):
    carl_loss = loss_function_carl(y_true, y_pred)
    score_loss = loss_function_score(y_true, y_pred)
    return carl_loss + alpha*score_loss


def loss_function_combinedregression(y_true, y_pred, alpha=0.005):
    ratio_regr_loss =  loss_function_ratio_regression(y_true, y_pred)
    score_loss = loss_function_score(y_true, y_pred)
    return ratio_regr_loss + alpha*score_loss


def loss_function_score_regression(y_true, y_pred):
    return losses.mean_squared_error(y_true[:, :], y_pred[:, :])

