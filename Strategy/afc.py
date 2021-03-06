from __future__ import absolute_import, division, print_function

import logging
import numpy as np

from scipy.interpolate import LinearNDInterpolator
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KernelDensity

import settings
from various.utils import format_number, calculate_mean_squared_error


def afc_inference(statistics='x',
                  indices_X=None,
                  epsilon=None,
                  kernel='gaussian',
                  use_smearing=False,
                  denominator=0,
                  kde_relative_tolerance=1.e-2,
                  kde_absolute_tolerance=1.e-4,
                  do_neyman=False,
                  options=''):
    """
    Approximates the likelihood through Approximate Frequentist Inference,
    a frequentist twist on ABC and effectively the same as kernel density
    estimation in the summary statistics space.

    Input:
        use_smearing:
        statistics: Defines which summary statistics is used to decide upon
                    acceptance or rejection of events. Currently the only
                    option is 'x', which bases the acceptance decision on an
                    epsilon ball in (re-scaled) feature space.
        indices_X: If statistics is 'x', this defines which of the features
                   are used in the distance calculation. If None, a default
                   selection of five variables is used.
        epsilon: This float > 0 defines the size of the epsilon-ball, i.e.
                 the bandwidth of the KDE. A smaller value makes the inference
                 more precise, but requires more data, especially when using
                 high-dimensional statistics. If no value is given, the
                 algorithm uses 0.1^(1/n_dim), where n_dim is the number of 
                 dimensions of the statistics space, for instance the length
                 of indices_X.
        kernel: The kernel. 'tophat' is equivalent to classic rejection ABC.
                Another option is 'gaussian'.
        do_neyman:
        kde_absolute_tolerance:
        kde_relative_tolerance:
        options: Further options in a list of strings or string.
    """

    logging.info('Starting AFC inference')

    ################################################################################
    # Settings
    ################################################################################

    if statistics is not 'x':
        raise NotImplementedError

    if indices_X is None:
        indices_X = [1, 38, 39, 40, 41]
        # pT(j1), m(Z2), m(jj), delta_eta(jj), delta_phi(jj)
        # indices_X = [1, 41]  # pT(j1), delta_phi(jj)

    statistics_dimensionality = len(indices_X)

    filename_addition = '_' + str(statistics_dimensionality) + 'd'

    if epsilon is None:
        epsilon = 0.05 ** (1. / statistics_dimensionality)
    else:
        filename_addition = filename_addition + '_epsilon_' + format_number(epsilon, 2)

    input_X_prefix = ''
    if use_smearing:
        input_X_prefix = 'smeared_'
        filename_addition += '_smeared'

    theta1 = settings.theta1_default
    input_filename_addition = ''
    if denominator > 0:
        input_filename_addition = '_denom' + str(denominator)
        filename_addition += '_denom' + str(denominator)
        theta1 = settings.theta1_alternatives[denominator - 1]

    results_dir = settings.base_dir + '/results/afc'
    neyman_dir = settings.neyman_dir + '/afc'

    logging.info('Settings:')
    if statistics == 'x':
        logging.info('  Statistics:              x %s', indices_X)
    else:
        logging.info('  Statistics:              %s', statistics)
    logging.info('  Epsilon (bandwidth):     %s', epsilon)
    logging.info('  Kernel:                  %s', kernel)
    logging.info('  Denominator theta:       denominator %s = theta %s = %s',
                 denominator, theta1, settings.thetas[theta1])

    ################################################################################
    # Data
    ################################################################################

    X_test = np.load(
        settings.unweighted_events_dir + '/' + input_X_prefix + 'X_test' + input_filename_addition + '.npy')
    r_test = np.load(settings.unweighted_events_dir + '/r_test' + input_filename_addition + '.npy')
    #X_neyman_observed = np.load(settings.unweighted_events_dir + '/neyman/' + input_X_prefix + 'X_neyman_observed.npy')
    n_events_test = X_test.shape[0]

    ################################################################################
    # AFC
    ################################################################################

    expected_llr = []
    mse_log_r = []
    trimmed_mse_log_r = []

    # Loop over the hypothesis thetas
    for i, t in enumerate(settings.pbp_training_thetas):

        logging.info('Starting theta %s/%s: number %s (%s)',
                     i + 1, len(settings.pbp_training_thetas), t, settings.thetas[t])

        # Load data
        X_train = np.load(
            settings.unweighted_events_dir + '/point_by_point/' + input_X_prefix + 'X_train_point_by_point_' + str(
                t) + input_filename_addition + '.npy')
        y_train = np.load(
            settings.unweighted_events_dir + '/point_by_point/y_train_point_by_point_' + str(t) + input_filename_addition + '.npy')

        # Scale data
        scaler = StandardScaler()
        scaler.fit(np.array(X_train, dtype=np.float64))
        X_train_transformed = scaler.transform(X_train)
        X_test_transformed = scaler.transform(X_test)

        # Construct summary statistics
        if statistics == 'x':
            summary_statistics_train = X_train_transformed[:, indices_X]
            summary_statistics_test = X_test_transformed[:, indices_X]
        else:
            raise NotImplementedError

        ################################################################################
        # "Training"
        ################################################################################

        # Set up KDEs for numerator and denominator
        kde_num = KernelDensity(bandwidth=epsilon, kernel=kernel, rtol=kde_relative_tolerance,
                                atol=kde_absolute_tolerance)
        kde_den = KernelDensity(bandwidth=epsilon, kernel=kernel, rtol=kde_relative_tolerance,
                                atol=kde_absolute_tolerance)

        # Fit KDEs for numerator and denominator
        kde_num.fit(summary_statistics_train[y_train == 0])
        kde_den.fit(summary_statistics_train[y_train == 1])

        ################################################################################
        # Evaluation
        ################################################################################

        # Evaluation
        log_p_hat_num_test = kde_num.score_samples(summary_statistics_test)
        log_p_hat_den_test = kde_den.score_samples(summary_statistics_test)

        # Sanitize output
        log_p_hat_num_test[np.invert(np.isfinite(log_p_hat_num_test))] = -1000.
        log_p_hat_den_test[np.invert(np.isfinite(log_p_hat_den_test))] = -1000.

        log_r_hat_test = log_p_hat_num_test - log_p_hat_den_test

        # Extract numbers of interest
        expected_llr.append(- 2. * settings.n_expected_events / n_events_test * np.sum(log_r_hat_test))
        mse_log_r.append(calculate_mean_squared_error(np.log(r_test[t]), log_r_hat_test, 0.))
        trimmed_mse_log_r.append(calculate_mean_squared_error(np.log(r_test[t]), log_r_hat_test, 'auto'))

        # For some benchmark thetas, save r for each phase-space point
        if t == settings.theta_benchmark_nottrained:
            np.save(results_dir + '/r_nottrained_afc' + filename_addition + '.npy', np.exp(log_r_hat_test))
        elif t == settings.theta_benchmark_trained:
            np.save(results_dir + '/r_trained_afc' + filename_addition + '.npy', np.exp(log_r_hat_test))

        # if do_neyman:
        #     # logging.debug('Neyman observed')
        #
        #     # Neyman construction: prepare observed data and construct summary statistics
        #     X_neyman_observed_transformed = scaler.transform(
        #         X_neyman_observed.reshape((-1, X_neyman_observed.shape[2])))
        #
        #     if statistics == 'x':
        #         summary_statistics_neyman_observed = X_neyman_observed_transformed[:, indices_X]
        #     else:
        #         raise NotImplementedError
        #
        #     # Neyman construction: evaluate observed sample
        #     log_p_hat_num_neyman_observed = kde_num.score_samples(summary_statistics_neyman_observed)
        #     log_p_hat_den_neyman_observed = kde_den.score_samples(summary_statistics_neyman_observed)
        #
        #     # Sanitize output
        #     log_p_hat_num_neyman_observed[np.invert(np.isfinite(log_p_hat_num_neyman_observed))] = -1000.
        #     log_p_hat_den_neyman_observed[np.invert(np.isfinite(log_p_hat_den_neyman_observed))] = -1000.
        #
        #     log_r_hat_neyman_observed = log_p_hat_num_neyman_observed - log_p_hat_den_neyman_observed
        #     log_r_hat_neyman_observed = log_r_hat_neyman_observed.reshape((-1, settings.n_expected_events))
        #
        #     # logging.debug('Neyman observed log p (num): shape %s, %s nans, content \n %s',
        #     #               log_p_hat_num_neyman_observed.shape, np.sum(np.isnan(log_p_hat_num_neyman_observed)),
        #     #               log_p_hat_num_neyman_observed)
        #     # logging.debug('Neyman observed log p (den): shape %s, %s nans, content \n %s',
        #     #               log_p_hat_den_neyman_observed.shape, np.sum(np.isnan(log_p_hat_den_neyman_observed)),
        #     #               log_p_hat_den_neyman_observed)
        #     # logging.debug('Neyman observed log r: shape %s, %s nans, content \n %s',
        #     #               log_r_hat_neyman_observed.shape, np.sum(np.isnan(log_r_hat_neyman_observed)),
        #     #               log_r_hat_neyman_observed)
        #
        #     llr_neyman_observed = -2. * np.sum(log_r_hat_neyman_observed, axis=1)
        #     np.save(neyman_dir + '/neyman_llr_observed_afc' + '_' + str(t) + filename_addition + '.npy',
        #             llr_neyman_observed)
        #
        #     # Neyman construction: loop over distribution samples generated from different thetas
        #     llr_neyman_distributions = []
        #     for tt in range(settings.n_thetas):
        #
        #         # Only evaluate certain combinations of thetas to save computation time
        #         if not decide_toy_evaluation(tt, t):
        #             placeholder = np.empty(settings.n_neyman_null_experiments)
        #             placeholder[:] = np.nan
        #             llr_neyman_distributions.append(placeholder)
        #             continue
        #
        #         # Neyman construction: load distribution sample and construct summary statistics
        #         X_neyman_distribution = np.load(
        #             settings.unweighted_events_dir + '/' + input_X_prefix + 'X_neyman_distribution_' + str(tt) + '.npy')
        #         X_neyman_distribution_transformed = scaler.transform(
        #             X_neyman_distribution.reshape((-1, X_neyman_distribution.shape[2])))
        #
        #         if statistics == 'x':
        #             summary_statistics_neyman_distribution = X_neyman_distribution_transformed[:, indices_X]
        #         else:
        #             raise NotImplementedError
        #
        #         # Neyman construction: evaluate observed sample
        #         log_p_hat_num_neyman_distribution = kde_num.score_samples(summary_statistics_neyman_distribution)
        #         log_p_hat_den_neyman_distribution = kde_den.score_samples(summary_statistics_neyman_distribution)
        #
        #         # Sanitize output
        #         log_p_hat_num_neyman_distribution[np.invert(np.isfinite(log_p_hat_num_neyman_distribution))] = -1000.
        #         log_p_hat_den_neyman_distribution[np.invert(np.isfinite(log_p_hat_den_neyman_distribution))] = -1000.
        #
        #         log_r_hat_neyman_distribution = log_p_hat_num_neyman_distribution - log_p_hat_den_neyman_distribution
        #         log_r_hat_neyman_distribution = log_r_hat_neyman_distribution.reshape((-1, settings.n_expected_events))
        #
        #         # logging.debug('Neyman distribution %s log p (num): shape %s, %s nans, content \n %s',
        #         #               tt, log_p_hat_num_neyman_distribution.shape,
        #         #               np.sum(np.isnan(log_p_hat_num_neyman_distribution)), log_p_hat_num_neyman_distribution)
        #         # logging.debug('Neyman distribution %s log p (den): shape %s, %s nans, content \n %s',
        #         #               tt, log_p_hat_den_neyman_distribution.shape,
        #         #               np.sum(np.isnan(log_p_hat_den_neyman_distribution)), log_p_hat_den_neyman_distribution)
        #         # logging.debug('Neyman distribution %s log r: shape %s, %s nans, content \n %s',
        #         #               tt, log_r_hat_neyman_distribution.shape, np.sum(np.isnan(log_r_hat_neyman_distribution)),
        #         #               log_r_hat_neyman_distribution)
        #
        #         llr_neyman_distribution = -2. * np.sum(log_r_hat_neyman_distribution, axis=1)
        #         np.save(neyman_dir + '/neyman_llr_distribution_afc' + '_' + str(t) + filename_addition + '.npy',
        #                 llr_neyman_distribution)
        #
        #     llr_neyman_distributions = np.asarray(llr_neyman_distributions)
        #     np.save(neyman_dir + '/neyman_llr_distribution_afc' + '_' + str(t) + filename_addition + '.npy',
        #             llr_neyman_distributions)

    # Interpolate and save evaluation results
    expected_llr = np.asarray(expected_llr)
    mse_log_r = np.asarray(mse_log_r)
    trimmed_mse_log_r = np.asarray(trimmed_mse_log_r)

    logging.info('Interpolation')

    interpolator = LinearNDInterpolator(settings.thetas[settings.pbp_training_thetas], expected_llr)
    expected_llr_all = interpolator(settings.thetas)
    # gp = GaussianProcessRegressor(normalize_y=True,
    #                              kernel=C(1.0) * Matern(1.0, nu=0.5), n_restarts_optimizer=10)
    # gp.fit(thetas[settings.pbp_training_thetas], expected_llr)
    # expected_llr_all = gp.predict(thetas)
    np.save(results_dir + '/llr_afc' + filename_addition + '.npy', expected_llr_all)

    interpolator = LinearNDInterpolator(settings.thetas[settings.pbp_training_thetas], mse_log_r)
    mse_log_r_all = interpolator(settings.thetas)
    np.save(results_dir + '/mse_logr_afc' + filename_addition + '.npy',
            mse_log_r_all)

    interpolator = LinearNDInterpolator(settings.thetas[settings.pbp_training_thetas], trimmed_mse_log_r)
    trimmed_mse_log_r_all = interpolator(settings.thetas)
    np.save(results_dir + '/trimmed_mse_logr_afc' + filename_addition + '.npy',
            trimmed_mse_log_r_all)
