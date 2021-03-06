"""
ML: Machine Learning
NN: Neural Network
SM: Standard Model
"""

from keras.models import Model
from keras.layers import Input, Dense, Lambda, Concatenate, Multiply, Reshape, ActivityRegularization, Activation, Add
from keras import optimizers
import keras.backend as K

import settings
from Estimator.E_ml_utils import build_hidden_layers
from Estimator.loss_func import loss_function_carl, loss_function_combined, loss_function_combinedregression, loss_function_ratio_regression, loss_function_score
from Estimator.metrics import full_cross_entropy, full_mse_log_r, full_mse_score
from Estimator.metrics import full_mae_log_r, full_mae_score
from Estimator.metrics import trimmed_cross_entropy, trimmed_mse_log_r, trimmed_mse_score
from Estimator.E_morphing import generate_wi_layer, generate_wtilde_layer

metrics = [full_cross_entropy, trimmed_cross_entropy,
           full_mse_log_r, trimmed_mse_log_r,
           full_mae_log_r,
           full_mse_score, trimmed_mse_score,
           full_mae_score]


################################################################################
# Regression models
################################################################################

def make_regressor(n_hidden_layers=3, hidden_layer_size=100, activation='tanh',
                   dropout_prob=0.0, learning_rate=1.e-3, lr_decay=0.):
    # Input tensor for NN-layers
    input_layer = Input(shape=(settings.n_thetas_features,))

    # Network
    hidden_layer = Dense(hidden_layer_size, activation=activation)(input_layer)
    if n_hidden_layers > 1:
        hidden_layer_ = build_hidden_layers(n_hidden_layers - 1,
                                            hidden_layer_size=hidden_layer_size,
                                            activation=activation,
                                            dropout_prob=dropout_prob)
        hidden_layer = hidden_layer_(hidden_layer)
    log_r_hat_layer = Dense(1, activation='linear')(hidden_layer)

    # Translate to s ------------------------------------------------- ?????
    r_hat_layer = Lambda(lambda x: K.exp(x))(log_r_hat_layer)
    s_hat_layer = Lambda(lambda x: 1. / (1. + x))(r_hat_layer)

    # Score
    gradient_layer = Lambda(lambda x: K.gradients(x[0], x[1])[0],
                            output_shape=(settings.n_thetas_features,))([log_r_hat_layer,                                                                          input_layer])
    score_layer = Lambda(lambda x: x[:, -settings.n_params:],
                         output_shape=(settings.n_params,))(gradient_layer)

    # Combine outputs
    output_layer = Concatenate()([s_hat_layer, log_r_hat_layer, score_layer])
    model = Model(inputs=[input_layer], outputs=[output_layer])

    # Compile model/learnable function
    model.compile(loss=loss_function_ratio_regression,
                  metrics=metrics,
                  optimizer=optimizers.Adam(lr=learning_rate, decay=lr_decay, clipnorm=10.))
    return model


def make_regressor_morphingaware(n_hidden_layers=2,
                                 hidden_layer_size=100,
                                 activation='tanh',
                                 dropout_prob=0.0,
                                 factor_out_sm=True,
                                 epsilon=1.e-4,
                                 learning_rate=1.e-3,
                                 lr_decay=0.):
    # Inputs
    input_layer = Input(shape=(settings.n_thetas_features,))
    x_layer = Lambda(lambda x: x[:, :settings.n_features],
                     output_shape=(settings.n_features,))(input_layer)
    theta_layer = Lambda(lambda x: x[:, -settings.n_params:],
                         output_shape=(settings.n_params,))(input_layer)

    # Morphing weights
    wtilde_layer = generate_wtilde_layer(theta_layer)
    wi_layer = generate_wi_layer(wtilde_layer)

    # Log ratio estimator for SM
    if factor_out_sm:
        hidden_layer = Dense(hidden_layer_size, activation=activation)(input_layer)
        if n_hidden_layers > 1:
            hidden_layer_ = build_hidden_layers(n_hidden_layers - 1,
                                                hidden_layer_size=hidden_layer_size,
                                                activation=activation,
                                                dropout_prob=dropout_prob)
            hidden_layer = hidden_layer_(hidden_layer)
        log_r0_hat_layer = Dense(1, activation='linear')(hidden_layer)
        r0_hat_layer = Lambda(lambda x: K.exp(x))(log_r0_hat_layer)

    # Log ratio estimators for each component
    ri_hat_layers = []
    for i in range(settings.n_morphing_samples):
        hidden_layer = Dense(hidden_layer_size, activation=activation)(x_layer)
        if n_hidden_layers > 1:
            hidden_layer_ = build_hidden_layers(n_hidden_layers - 1,
                                                hidden_layer_size=hidden_layer_size,
                                                activation=activation,
                                                dropout_prob=dropout_prob)
            hidden_layer = hidden_layer_(hidden_layer)

        if factor_out_sm:
            delta_ri_hat_layer = Dense(1, activation='linear')(hidden_layer)
            ri_hat_layer = Add()([delta_ri_hat_layer, r0_hat_layer])

        else:
            log_ri_hat_layer = Dense(1, activation='linear')(hidden_layer)
            ri_hat_layer = Lambda(lambda x: K.exp(x))(log_ri_hat_layer)

        ri_hat_layers.append(Reshape((1,))(ri_hat_layer))
    ri_hat_layer = Concatenate()(ri_hat_layers)

    # Combine, clip, transform to \hat{s}
    wi_ri_hat_layer = Multiply()([wi_layer, ri_hat_layer])
    r_hat_layer = Reshape((1,))(Lambda(lambda x: K.sum(x, axis=1))(wi_ri_hat_layer))
    positive_r_hat_layer = Activation('relu')(r_hat_layer)

    # Translate to s
    s_hat_layer = Lambda(lambda x: 1. / (1. + x))(positive_r_hat_layer)

    # Score
    log_r_hat_layer = Lambda(lambda x: K.log(x + epsilon))(positive_r_hat_layer)
    gradient_layer = Lambda(lambda x: K.gradients(x[0], x[1])[0], output_shape=(settings.n_thetas_features,))(
        [log_r_hat_layer, input_layer])
    score_layer = Lambda(lambda x: x[:, -settings.n_params:], output_shape=(settings.n_params,))(gradient_layer)

    # Combine outputs
    output_layer = Concatenate()([s_hat_layer, log_r_hat_layer, score_layer,
                                  wi_layer, ri_hat_layer])
    model = Model(inputs=[input_layer], outputs=[output_layer])

    # Compile model
    model.compile(loss=loss_function_ratio_regression,
                  metrics=metrics,
                  optimizer=optimizers.Adam(lr=learning_rate, decay=lr_decay, clipnorm=10.))

    return model


################################################################################
# Regression + score models
################################################################################

def make_combined_regressor(n_hidden_layers=3,
                            hidden_layer_size=100,
                            activation='tanh',
                            dropout_prob=0.0,
                            alpha=0.005,
                            learning_rate=1.e-3,
                            lr_decay=0.):
    # Inputs
    input_layer = Input(shape=(settings.n_thetas_features,))

    # Network
    hidden_layer = Dense(hidden_layer_size, activation=activation)(input_layer)
    if n_hidden_layers > 1:
        hidden_layer_ = build_hidden_layers(n_hidden_layers - 1,
                                            hidden_layer_size=hidden_layer_size,
                                            activation=activation,
                                            dropout_prob=dropout_prob)
        hidden_layer = hidden_layer_(hidden_layer)
    log_r_hat_layer = Dense(1, activation='linear')(hidden_layer)

    # Translate to s
    r_hat_layer = Lambda(lambda x: K.exp(x))(log_r_hat_layer)
    s_hat_layer = Lambda(lambda x: 1. / (1. + x))(r_hat_layer)

    # Score
    gradient_layer = Lambda(lambda x:
            K.gradients(x[0], x[1])[0],
            output_shape=(settings.n_thetas_features,))([log_r_hat_layer, input_layer])
    score_layer = Lambda(lambda x: 
            x[:, -settings.n_params:],
            output_shape=(settings.n_params,))(gradient_layer)

    # Combine outputs
    output_layer = Concatenate()([s_hat_layer, log_r_hat_layer, score_layer])
    model = Model(inputs=[input_layer], outputs=[output_layer])

    # Compile model
    model.compile(loss=lambda x, y: loss_function_combinedregression(x, y, alpha=alpha),
                  metrics=metrics,
                  optimizer=optimizers.Adam(lr=learning_rate,
                                            decay=lr_decay,
                                            clipnorm=10.))
    return model


def make_combined_regressor_morphingaware(n_hidden_layers=2,
                                          hidden_layer_size=100,
                                          activation='tanh',
                                          dropout_prob=0.0,
                                          factor_out_sm=True,
                                          alpha=0.005,
                                          epsilon=1.e-4,
                                          learning_rate=1.e-3,
                                          lr_decay=0.):
    # Inputs
    input_layer = Input(shape=(settings.n_thetas_features,))
    x_layer = Lambda(lambda x: x[:, :settings.n_features], output_shape=(settings.n_features,))(input_layer)
    theta_layer = Lambda(lambda x: x[:, -settings.n_params:], output_shape=(settings.n_params,))(input_layer)

    # Morphing weights
    wtilde_layer = generate_wtilde_layer(theta_layer)
    wi_layer = generate_wi_layer(wtilde_layer)

    # Log ratio estimator for SM
    if factor_out_sm:
        hidden_layer = Dense(hidden_layer_size, activation=activation)(input_layer)
        if n_hidden_layers > 1:
            hidden_layer_ = build_hidden_layers(n_hidden_layers - 1,
                                                hidden_layer_size=hidden_layer_size,
                                                activation=activation,
                                                dropout_prob=dropout_prob)
            hidden_layer = hidden_layer_(hidden_layer)
        log_r0_hat_layer = Dense(1, activation='linear')(hidden_layer)
        r0_hat_layer = Lambda(lambda x: K.exp(x))(log_r0_hat_layer)

    # Log ratio estimators for each component
    ri_hat_layers = []
    for i in range(settings.n_morphing_samples):
        hidden_layer = Dense(hidden_layer_size, activation=activation)(x_layer)
        if n_hidden_layers > 1:
            hidden_layer_ = build_hidden_layers(n_hidden_layers - 1,
                                                hidden_layer_size=hidden_layer_size,
                                                activation=activation,
                                                dropout_prob=dropout_prob)
            hidden_layer = hidden_layer_(hidden_layer)

        if factor_out_sm:
            delta_ri_hat_layer = Dense(1, activation='linear')(hidden_layer)
            ri_hat_layer = Add()([delta_ri_hat_layer, r0_hat_layer])

        else:
            log_ri_hat_layer = Dense(1, activation='linear')(hidden_layer)
            ri_hat_layer = Lambda(lambda x: K.exp(x))(log_ri_hat_layer)

        ri_hat_layers.append(Reshape((1,))(ri_hat_layer))
    ri_hat_layer = Concatenate()(ri_hat_layers)

    # Combine, clip, transform to \hat{s}
    wi_ri_hat_layer = Multiply()([wi_layer, ri_hat_layer])
    r_hat_layer = Reshape((1,))(Lambda(lambda x: K.sum(x, axis=1))(wi_ri_hat_layer))
    positive_r_hat_layer = Activation('relu')(r_hat_layer)

    # Translate to s
    s_hat_layer = Lambda(lambda x: 1. / (1. + x))(positive_r_hat_layer)

    # Score
    log_r_hat_layer = Lambda(lambda x: K.log(x + epsilon))(positive_r_hat_layer)
    gradient_layer = Lambda(lambda x: K.gradients(x[0], x[1])[0], output_shape=(settings.n_thetas_features,))(
        [log_r_hat_layer, input_layer])
    score_layer = Lambda(lambda x: x[:, -settings.n_params:], output_shape=(settings.n_params,))(gradient_layer)

    # Combine outputs
    # output_layer = Concatenate()([log_r_hat_layer, score_layer])
    output_layer = Concatenate()([s_hat_layer, log_r_hat_layer, score_layer, wi_layer, ri_hat_layer])
    model = Model(inputs=[input_layer], outputs=[output_layer])

    # Compile model
    model.compile(loss=lambda x, y: loss_function_combinedregression(x, y, alpha=alpha),
                  metrics=metrics,
                  optimizer=optimizers.Adam(lr=learning_rate, decay=lr_decay, clipnorm=10.))

    return model


################################################################################
# carl models
################################################################################

def make_classifier_carl(n_hidden_layers=3,
                         hidden_layer_size=100,
                         activation='tanh',
                         dropout_prob=0.0,
                         learn_log_r=False,
                         learning_rate=1.e-3,
                         lr_decay=0.):
    # Inputs
    input_layer = Input(shape=(settings.n_thetas_features,))

    # Network
    hidden_layer = Dense(hidden_layer_size, activation=activation)(input_layer)
    if n_hidden_layers > 1:
        hidden_layer_ = build_hidden_layers(n_hidden_layers - 1,
                                            hidden_layer_size=hidden_layer_size,
                                            activation=activation,
                                            dropout_prob=dropout_prob)
        hidden_layer = hidden_layer_(hidden_layer)

    if learn_log_r:
        log_r_hat_layer = Dense(1, activation='linear')(hidden_layer)
        r_hat_layer = Lambda(lambda x: K.exp(x))(log_r_hat_layer)
        s_hat_layer = Lambda(lambda x: 1. / (1. + x))(r_hat_layer)

    else:
        s_hat_layer = Dense(1, activation='sigmoid')(hidden_layer)
        r_hat_layer = Lambda(lambda x: (1. - x) / x)(s_hat_layer)
        log_r_hat_layer = Lambda(lambda x: K.log(x))(r_hat_layer)

    # Score
    gradient_layer = Lambda(lambda x: K.gradients(x[0], x[1])[0],
                            output_shape=(settings.n_thetas_features,))([log_r_hat_layer, input_layer])
    score_layer = Lambda(lambda x: x[:, -settings.n_params:], output_shape=(settings.n_params,))(gradient_layer)

    # Combine outputs
    output_layer = Concatenate()([s_hat_layer, log_r_hat_layer, score_layer])
    model = Model(inputs=[input_layer], outputs=[output_layer])

    # Compile model
    model.compile(loss=loss_function_carl,
                  metrics=metrics,
                  optimizer=optimizers.Adam(lr=learning_rate, decay=lr_decay, clipnorm=1.))

    return model


def make_classifier_carl_morphingaware(n_hidden_layers=2,
                                       hidden_layer_size=100,
                                       activation='tanh',
                                       dropout_prob=0.0,
                                       learn_log_r=False,
                                       epsilon=1.e-4,
                                       learning_rate=1.e-3,
                                       lr_decay=0.):
    # Inputs
    input_layer = Input(shape=(settings.n_thetas_features,))
    x_layer = Lambda(lambda x: x[:, :settings.n_features], output_shape=(settings.n_features,))(input_layer)
    theta_layer = Lambda(lambda x: x[:, -settings.n_params:], output_shape=(settings.n_params,))(input_layer)

    # Morphing weights
    wtilde_layer = generate_wtilde_layer(theta_layer)
    wi_layer = generate_wi_layer(wtilde_layer)

    # Ratio estimators for each component
    ri_hat_layers = []
    for i in range(settings.n_morphing_samples):
        hidden_layer = Dense(hidden_layer_size, activation=activation)(x_layer)
        if n_hidden_layers > 1:
            hidden_layer_ = build_hidden_layers(n_hidden_layers - 1,
                                                hidden_layer_size=hidden_layer_size,
                                                activation=activation,
                                                dropout_prob=dropout_prob)
            hidden_layer = hidden_layer_(hidden_layer)

        if learn_log_r:
            log_ri_hat_layer = Dense(1, activation='linear')(hidden_layer)
            ri_hat_layers.append(Lambda(lambda x: K.exp(x))(log_ri_hat_layer))
        else:
            si_hat_layer = Dense(1, activation='sigmoid')(hidden_layer)
            ri_hat_layers.append(Reshape((1,))(Lambda(lambda x: (1. - x) / (x + epsilon))(si_hat_layer)))

    ri_hat_layer = Concatenate()(ri_hat_layers)

    # Combine, clip, transform to \hat{s}
    wi_ri_hat_layer = Multiply()([wi_layer, ri_hat_layer])
    r_hat_layer = Reshape((1,))(Lambda(lambda x: K.sum(x, axis=1))(wi_ri_hat_layer))
    positive_r_hat_layer = Activation('relu')(r_hat_layer)
    s_hat_layer = Lambda(lambda x: 1. / (1. + x))(positive_r_hat_layer)

    # Score
    log_r_hat_layer = Lambda(lambda x: K.log(x + epsilon))(positive_r_hat_layer)
    gradient_layer = Lambda(lambda x: K.gradients(x[0], x[1])[0], output_shape=(settings.n_thetas_features,))(
        [log_r_hat_layer, input_layer])
    score_layer = Lambda(lambda x: x[:, -settings.n_params:], output_shape=(settings.n_params,))(gradient_layer)

    # Combine outputs
    # output_layer = Concatenate()([s_hat_layer, score_layer])
    output_layer = Concatenate()([s_hat_layer, log_r_hat_layer, score_layer, wi_layer, ri_hat_layer])
    model = Model(inputs=[input_layer], outputs=[output_layer])

    # Compile model
    model.compile(loss=loss_function_carl,
                  metrics=metrics,
                  optimizer=optimizers.Adam(lr=learning_rate, decay=lr_decay, clipnorm=1.))

    return model


################################################################################
# Score only models
################################################################################

def make_classifier_score(n_hidden_layers=3,
                          hidden_layer_size=100,
                          activation='tanh',
                          dropout_prob=0.0,
                          learn_log_r=False,
                          l2_regularization=0.001,
                          learning_rate=1.e-3,
                          lr_decay=0.):
    # Inputs
    input_layer = Input(shape=(settings.n_thetas_features,))

    # Network
    hidden_layer = Dense(hidden_layer_size, activation=activation)(input_layer)
    if n_hidden_layers > 1:
        hidden_layer_ = build_hidden_layers(n_hidden_layers - 1,
                                            hidden_layer_size=hidden_layer_size,
                                            activation=activation,
                                            dropout_prob=dropout_prob)
        hidden_layer = hidden_layer_(hidden_layer)

    if learn_log_r:
        log_r_hat_layer = Dense(1, activation='linear')(hidden_layer)
        r_hat_layer = Lambda(lambda x: K.exp(x))(log_r_hat_layer)
        s_hat_layer = Lambda(lambda x: 1. / (1. + x))(r_hat_layer)

    else:
        s_hat_layer = Dense(1, activation='sigmoid')(hidden_layer)
        r_hat_layer = Lambda(lambda x: (1. - x) / x)(s_hat_layer)
        log_r_hat_layer = Lambda(lambda x: K.log(x))(r_hat_layer)

    # Score
    regularizer_layer = ActivityRegularization(l1=0., l2=l2_regularization)(log_r_hat_layer)
    gradient_layer = Lambda(lambda x: K.gradients(x[0], x[1])[0],
                            output_shape=(settings.n_thetas_features,))([regularizer_layer, input_layer])
    score_layer = Lambda(lambda x: x[:, -settings.n_params:], output_shape=(settings.n_params,))(gradient_layer)

    # Combine outputs
    output_layer = Concatenate()([s_hat_layer, log_r_hat_layer, score_layer])
    model = Model(inputs=[input_layer], outputs=[output_layer])

    # Compile model
    model.compile(loss=loss_function_score,
                  metrics=metrics,
                  optimizer=optimizers.Adam(lr=learning_rate, decay=lr_decay, clipnorm=1.))

    return model


def make_classifier_score_morphingaware(n_hidden_layers=2,
                                        hidden_layer_size=100,
                                        activation='tanh',
                                        dropout_prob=0.0,
                                        l2_regularization=0.001,
                                        learn_log_r=False,
                                        epsilon=1.e-4,
                                        learning_rate=1.e-3,
                                        lr_decay=0.):
    # Inputs
    input_layer = Input(shape=(settings.n_thetas_features,))
    x_layer = Lambda(lambda x: x[:, :settings.n_features], output_shape=(settings.n_features,))(input_layer)
    theta_layer = Lambda(lambda x: x[:, -settings.n_params:], output_shape=(settings.n_params,))(input_layer)

    # Morphing weights
    wtilde_layer = generate_wtilde_layer(theta_layer)
    wi_layer = generate_wi_layer(wtilde_layer)

    # Ratio estimators for each component
    ri_hat_layers = []
    for i in range(settings.n_morphing_samples):
        hidden_layer = Dense(hidden_layer_size, activation=activation)(x_layer)
        if n_hidden_layers > 1:
            hidden_layer_ = build_hidden_layers(n_hidden_layers - 1,
                                                hidden_layer_size=hidden_layer_size,
                                                activation=activation,
                                                dropout_prob=dropout_prob)
            hidden_layer = hidden_layer_(hidden_layer)

        if learn_log_r:
            log_ri_hat_layer = Dense(1, activation='linear')(hidden_layer)
            ri_hat_layers.append(Lambda(lambda x: K.exp(x))(log_ri_hat_layer))
        else:
            si_hat_layer = Dense(1, activation='sigmoid')(hidden_layer)
            ri_hat_layers.append(Reshape((1,))(Lambda(lambda x: (1. - x) / (x + epsilon))(si_hat_layer)))

    ri_hat_layer = Concatenate()(ri_hat_layers)

    # Combine, clip, transform to \hat{s}
    wi_ri_hat_layer = Multiply()([wi_layer, ri_hat_layer])
    r_hat_layer = Reshape((1,))(Lambda(lambda x: K.sum(x, axis=1))(wi_ri_hat_layer))
    positive_r_hat_layer = Activation('relu')(r_hat_layer)
    s_hat_layer = Lambda(lambda x: 1. / (1. + x))(positive_r_hat_layer)

    # Score
    log_r_hat_layer = Lambda(lambda x: K.log(x + epsilon))(positive_r_hat_layer)
    regularizer_layer = ActivityRegularization(l1=0., l2=l2_regularization)(log_r_hat_layer)
    gradient_layer = Lambda(lambda x: K.gradients(x[0], x[1])[0], output_shape=(settings.n_thetas_features,))(
        [regularizer_layer, input_layer])
    score_layer = Lambda(lambda x: x[:, -settings.n_params:], output_shape=(settings.n_params,))(gradient_layer)

    # Combine outputs
    # output_layer = Concatenate()([s_hat_layer, score_layer])
    output_layer = Concatenate()([s_hat_layer, log_r_hat_layer, score_layer, wi_layer, ri_hat_layer])
    model = Model(inputs=[input_layer], outputs=[output_layer])

    # Compile model
    model.compile(loss=loss_function_score,
                  metrics=metrics,
                  optimizer=optimizers.Adam(lr=learning_rate, decay=lr_decay, clipnorm=1.))

    return model


################################################################################
# carl + score models
################################################################################

def make_classifier_combined(n_hidden_layers=3,
                             hidden_layer_size=100,
                             activation='tanh',
                             dropout_prob=0.0,
                             learn_log_r=False,
                             alpha=0.1,
                             learning_rate=1.e-3,
                             lr_decay=0.):
    # Inputs
    input_layer = Input(shape=(settings.n_thetas_features,))

    # Network
    hidden_layer = Dense(hidden_layer_size, activation=activation)(input_layer)
    if n_hidden_layers > 1:
        hidden_layer_ = build_hidden_layers(n_hidden_layers - 1,
                                            hidden_layer_size=hidden_layer_size,
                                            activation=activation,
                                            dropout_prob=dropout_prob)
        hidden_layer = hidden_layer_(hidden_layer)

    if learn_log_r:
        log_r_hat_layer = Dense(1, activation='linear')(hidden_layer)
        r_hat_layer = Lambda(lambda x: K.exp(x))(log_r_hat_layer)
        s_hat_layer = Lambda(lambda x: 1. / (1. + x))(r_hat_layer)

    else:
        s_hat_layer = Dense(1, activation='sigmoid')(hidden_layer)
        r_hat_layer = Lambda(lambda x: (1. - x) / x)(s_hat_layer)
        log_r_hat_layer = Lambda(lambda x: K.log(x))(r_hat_layer)

    # Score
    gradient_layer = Lambda(lambda x: K.gradients(x[0], x[1])[0],
                            output_shape=(settings.n_thetas_features,))([log_r_hat_layer, input_layer])
    score_layer = Lambda(lambda x: x[:, -settings.n_params:], output_shape=(settings.n_params,))(gradient_layer)

    # Combine outputs
    output_layer = Concatenate()([s_hat_layer, log_r_hat_layer, score_layer])
    model = Model(inputs=[input_layer], outputs=[output_layer])

    # Compile model
    model.compile(loss=lambda x, y: loss_function_combined(x, y, alpha=alpha),
                  metrics=metrics,
                  optimizer=optimizers.Adam(lr=learning_rate, decay=lr_decay, clipnorm=1.))

    return model


def make_classifier_combined_morphingaware(n_hidden_layers=2,
                                           hidden_layer_size=100,
                                           activation='tanh',
                                           dropout_prob=0.0,
                                           alpha=0.1,
                                           learn_log_r=False,
                                           epsilon=1.e-4,
                                           learning_rate=1.e-3,
                                           lr_decay=0.):
    # Inputs
    input_layer = Input(shape=(settings.n_thetas_features,))
    x_layer = Lambda(lambda x: x[:, :settings.n_features], output_shape=(settings.n_features,))(input_layer)
    theta_layer = Lambda(lambda x: x[:, -settings.n_params:], output_shape=(settings.n_params,))(input_layer)

    # Morphing weights
    wtilde_layer = generate_wtilde_layer(theta_layer)
    wi_layer = generate_wi_layer(wtilde_layer)

    # Ratio estimators for each component
    ri_hat_layers = []
    for i in range(settings.n_morphing_samples):
        hidden_layer = Dense(hidden_layer_size, activation=activation)(x_layer)
        if n_hidden_layers > 1:
            hidden_layer_ = build_hidden_layers(n_hidden_layers - 1,
                                                hidden_layer_size=hidden_layer_size,
                                                activation=activation,
                                                dropout_prob=dropout_prob)
            hidden_layer = hidden_layer_(hidden_layer)

        if learn_log_r:
            log_ri_hat_layer = Dense(1, activation='linear')(hidden_layer)
            ri_hat_layers.append(Lambda(lambda x: K.exp(x))(log_ri_hat_layer))
        else:
            si_hat_layer = Dense(1, activation='sigmoid')(hidden_layer)
            ri_hat_layers.append(Reshape((1,))(Lambda(lambda x: (1. - x) / (x + epsilon))(si_hat_layer)))

    ri_hat_layer = Concatenate()(ri_hat_layers)

    # Combine, clip, transform to \hat{s}
    wi_ri_hat_layer = Multiply()([wi_layer, ri_hat_layer])
    r_hat_layer = Reshape((1,))(Lambda(lambda x: K.sum(x, axis=1))(wi_ri_hat_layer))
    positive_r_hat_layer = Activation('relu')(r_hat_layer)
    s_hat_layer = Lambda(lambda x: 1. / (1. + x))(positive_r_hat_layer)

    # Score
    log_r_hat_layer = Lambda(lambda x: K.log(x + epsilon))(positive_r_hat_layer)
    gradient_layer = Lambda(lambda x: K.gradients(x[0], x[1])[0], output_shape=(settings.n_thetas_features,))(
        [log_r_hat_layer, input_layer])
    score_layer = Lambda(lambda x: x[:, -settings.n_params:], output_shape=(settings.n_params,))(gradient_layer)

    # Combine outputs
    # output_layer = Concatenate()([s_hat_layer, score_layer])
    output_layer = Concatenate()([s_hat_layer, log_r_hat_layer, score_layer, wi_layer, ri_hat_layer])
    model = Model(inputs=[input_layer], outputs=[output_layer])

    # Compile model
    model.compile(loss=lambda x, y: loss_function_combined(x, y, alpha=alpha),
                  metrics=metrics,
                  optimizer=optimizers.Adam(lr=learning_rate, decay=lr_decay, clipnorm=1.))

    return model
