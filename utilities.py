import os
import pandas
import pandas as pd

import numpy as np

import sklearn

import matplotlib.pyplot as plt

import scipy as sp

from copy import deepcopy

from scipy.stats import ks_2samp

from collections import defaultdict

from sklearn import pipeline

from sklearn.externals import joblib

from sklearn.model_selection import (ParameterGrid, GridSearchCV,
                                     StratifiedKFold, train_test_split)

from sklearn.metrics import (roc_auc_score, log_loss, f1_score, 
                             average_precision_score, precision_score, 
                             brier_score_loss, recall_score, accuracy_score,
                             precision_recall_fscore_support)


# Finds model with most number of scores
def model_with_most_scores(db):
    """
    Finds model with most number of scores

    Parameters
    ----------
    db : dict

    Returns
    -------
    int: model number
    """

    # Find max count
    max_count = max(len(v) for v in db.values())

    # returns first element in case of ties
    return [k for k, v in db.items() if len(v) == max_count][0]


## Generate a list of models base on a set of hyper-parameters
def model_grid_setup(estimator, param_grid):
    """
    Generates a list of models with all possible combination of
    hyper-paramters set

    Parameters
    ----------
    estimator : estimator object
        This is assumed to implement the scikit-learn estimator interface
        (klearn model, gridseachcv, or pipeline object)
    param_grid : dict
        Dictionary with hyper-parameters names (string) as keys and lists
        of parameter settings to try as values. This enables searching over
        any sequence of parameter settings.

    Returns
    -------
    list: list of models with their respective hyper-parameters set
    """

    models = []

    # generate list of models from param_grid
    for params in list(ParameterGrid(param_grid)):
        model = estimator.set_params(**params)

        models.append(deepcopy(model))

    return models


# generates random indices for each class in equal quantities
def random_indices(df, undersampling=False):
    """
    Produces random indices for undersampling majority class
    of pandas dataframe

    Parameters
    ----------
    df : Series, shape = [n_samples]
        pandas series object

    Returns
    -------
    Index: pandas index object
        undersampled indices
    """

    print('BEWARE UNDERSAMPLING IS TURNED ON')
    # initialize indices of random undersampling
    undersampled_indices = pd.Index([])
 
    # sample size of minorirty class
    sample_size = df.groupby(df).count().min()

    df = df.to_frame()

    col = df.columns.values[0]

    for i in set(df[col]):
        undersampled_indices = undersampled_indices.union(df[df[col]==i].sample(n=sample_size, replace=True).index)

    return undersampled_indices


# Evaluate model
def model_score(estimator, X, y, scoring='accuracy', average='weighted'):
    """
    Nested k-fold crossvalidation

    Parameters
    ----------
    estimator : estimator object
    X : {array-like, sparse matrix}, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features
    y : array-like, shape (n_samples,)
        Target vector relative to X.
    scoring : string, default 'accuracy'
    average : string, default 'weighted'
        options are 'binary', 'micro', 'macro', 'weighted',
        and 'samples'

    Returns
    -------
    score: float
        Score of
    """

    if scoring=='accuracy':
        return accuracy_score(y, estimator.predict(X))

    elif scoring=='brier_score':
        return 1 - brier_score_loss(y, estimator.predict_proba(X)[:, 1])

    elif scoring=='log_loss':
        return log_loss(y, estimator.predict_proba(X))

    elif scoring=='f1_score':
        return f1_score(y, estimator.predict(X), average=average)

    elif scoring=='precision':
        return precision_score(y, estimator.predict(X), average=average)

    elif scoring=='recall':
        return recall_score(y, estimator.predict(X), average=average)

    elif scoring=='average_precision': # does not support multiclass
        return average_precision_score(y, estimator.predict(X), average=average)

    elif scoring=='roc_auc': # does not support multiclass
        return roc_auc_score(y, estimator.predict(X), average=average)

    else:
        return None


## Select appropriate score
def score_selector(score, scoring):
    """
    Selects the appropriate score

    Parameters
    ----------
    score : array, shape = [n_samples]
            true class, intergers in [0, n_classes - 1)
    scoring : shape = [n_samples, n_classes]

    Returns
    -------
    int : model with the best score
    """
    
    num_model = None
    
    methods = ['accuracy', 'brier_score', 'f1_score', 
               'precision', 'recall', 'average_precision', 
               'roc_auc']
    
    if scoring=='log_loss':
        return min(score, key=score.get)
    elif scoring in methods:
        return max(score, key=score.get)
    else:
        return None


## Defined overfitting plot
def plot_overfitting(model, X_train, X_test, y_train, y_test,
                     bins=50, pos_class=1, directory='models'):
    """
    Multi class version of Logarithmic Loss metric
    Parameters
    ----------
    y_true : array, shape = [n_samples]
            true class, intergers in [0, n_classes - 1)
    y_pred : array, shape = [n_samples, n_classes]
    Returns
    -------
    loss : float
    """

    print(type(model))

    # check to see if model is a pipeline object or not
    if isinstance(model, sklearn.pipeline.Pipeline):
        data_type = type(model._final_estimator)
        print(type(model._final_estimator))
    elif isinstance(model, sklearn.model_selection._search.GridSearchCV):
        if isinstance(model.best_estimator_, sklearn.pipeline.Pipeline):
            print('ecc')
            data_type = type(model.best_estimator_._final_estimator)
        else:
            print('ccc')
            data_type = type(model.best_estimator_)
    else:
        data_type = type(model)

    name = ''.join(filter(str.isalnum, str(data_type).split(".")[-1]))

    print(name)

    # check to see if model file exist
    if not os.path.isfile(directory+'/'+str(name)+'.pkl'):
        model.fit(X_train, y_train)
        joblib.dump(model, directory+'/'+str(name)+'.pkl')
    else:
        print('Using model file stored in', directory)
        model = joblib.load(directory+'/'+str(name)+'.pkl')

    # use subplot to extract axis to add ks and p-value to plot
    fig, ax = plt.subplots(figsize=(8, 6))

    # Customize the major grid
    ax.grid(which='major', linestyle='-', linewidth='0.2', color='gray')
    ax.set_facecolor('white')

    if not hasattr(model, 'predict_proba'): # use decision function
        d = model.decision_function(sp.sparse.vstack([X_train, X_test]))
        bin_edges_low_high = np.linspace(min(d), max(d), bins + 1)
    else: # use prediction function
        bin_edges_low_high = np.linspace(0., 1., bins + 1)

    label_name = ""
    y_scores = []
    for X, y in [(X_train, y_train), (X_test, y_test)]:

        if hasattr(model, 'predict_proba'):
            label_name = 'Prediction Probability'
            y_scores.append(model.predict_proba(X[y > 0])[:, pos_class])
            y_scores.append(model.predict_proba(X[y < 1])[:, pos_class])
        else:
            label_name = 'Decision Function'
            y_scores.append(model.decision_function(X[y > 0]))
            y_scores.append(model.decision_function(X[y < 1]))

    width = np.diff(bin_edges_low_high)

    # Signal training histogram
    hist_sig_train, bin_edges = np.histogram(y_scores[0], bins=bin_edges_low_high)

    hist_sig_train = hist_sig_train / np.sum(hist_sig_train, dtype=np.float32)

    plt.bar(bin_edges[:-1], hist_sig_train, width=width, color='r', alpha=0.5,
            label='signal (train)')

    # Background training histogram
    hist_bkg_train, bin_edges = np.histogram(y_scores[1], bins=bin_edges_low_high)

    hist_bkg_train = hist_bkg_train / np.sum(hist_bkg_train, dtype=np.float32)

    plt.bar(bin_edges[:-1], hist_bkg_train, width=width,
            color='steelblue', alpha=0.5, label='background (train)')

    # Signal test histogram
    hist_sig_test, bin_edges = np.histogram(y_scores[2], bins=bin_edges_low_high)

    hist_sig_test = hist_sig_test / np.sum(hist_sig_test, dtype=np.float32)
    scale = len(y_scores[2]) / np.sum(hist_sig_test, dtype=np.float32)
    err = np.sqrt(hist_sig_test * scale) / scale

    plt.errorbar(bin_edges[:-1], hist_sig_test, yerr=err, fmt='o', c='r', label='signal (test)')

    # Background test histogram
    hist_bkg_test, bin_edges = np.histogram(y_scores[3], bins=bin_edges_low_high)

    hist_bkg_test = hist_bkg_test / np.sum(hist_bkg_test, dtype=np.float32)
    scale = len(y_scores[3]) / np.sum(hist_bkg_test, dtype=np.float32)
    err = np.sqrt(hist_bkg_test * scale) / scale

    plt.errorbar(bin_edges[:-1], hist_bkg_test, yerr=err, fmt='o', c='steelblue',
                 label='background (test)')

    # Estimate ks-test and p-values as an indicator of overtraining of fit model
    #s_ks, s_pv = ks_2samp(hist_sig_test, hist_sig_train)
    #b_ks, b_pv = ks_2samp(hist_bkg_test, hist_bkg_train)

    ax.set_title(name, fontsize=14)
    
    plt.xlabel(label_name)
    plt.ylabel('Arbitrary units')

    leg = plt.legend(loc='best', frameon=False, fancybox=False, fontsize=12)
    leg.get_frame().set_edgecolor('w')

    frame = leg.get_frame()
    frame.set_facecolor('White')

    return display(plt.show())


# Define summary report
def summary_report(model, y_test, y_train, X_test, X_train):
    """
    Summary report listing accuracy, precision, and log loss

    Parameters
    ----------
    model : array, shape = [n_samples]
            true class, intergers in [0, n_classes - 1)
    X : array,  shape = [n_samples, n_classes]
    y : array,  shape = [n_samples, n_classes]
    outer_cv:   shape = [n_samples, n_classes]
    inner_cv:   shape = [n_samples, n_classes]
    scoring:    shape = [n_samples, n_classes]

    Returns
    -------
    model: classifier re-fitted to full dataset
    """

    print(type(model))

    y_train_predict = model.predict(X_train)
    y_train_predict_proba = model.predict_proba(X_train)

    y_test_predict = model.predict(X_test)
    y_test_predict_proba = model.predict_proba(X_test)

    accuracy = accuracy_score(y_train, y_train_predict)
    print("Accuracy score on train data:", accuracy)

    accuracy = accuracy_score(y_test, y_test_predict)
    print("Accuracy score on test data:", accuracy)

    print('')

    precision = precision_score(y_train, y_train_predict, average=None)
    print("Precision score on train data:", precision)

    precision = precision_score(y_test, y_test_predict, average=None)
    print("Precision score on test data:", precision)

    print('')

    recall = recall_score(y_train, y_train_predict, average=None)
    print("Recall score on train data:", recall)

    recall = recall_score(y_test, y_test_predict, average=None)
    print("Recall score on test data:", recall)

    print('')

    logloss = log_loss(y_train, y_train_predict_proba)
    print("Log loss on train data:", logloss)

    logloss = log_loss(y_test, y_test_predict_proba)
    print("Log loss on test data:", logloss)


## Standard nested k-fold cross validation
def grid_search(model, X, y, outer_cv, inner_cv,
                param_grid, scoring="accuracy",
                sampling=None, n_jobs=1):
    """
    Nested k-fold crossvalidation.
    Parameters
    ----------
    Classifier : array, shape = [n_samples]
            true class, intergers in [0, n_classes - 1)
    X : array,  shape = [n_samples, n_classes]
    y : array,  shape = [n_samples, n_classes]
    outer_cv:   shape = [n_samples, n_classes]
    inner_cv:   shape = [n_samples, n_classes]
    param_grid: shape = [n_samples, n_classes]
    scoring:    shape = [n_samples, n_classes]
    Returns
    -------
    Grid classifier: classifier re-fitted to full dataset
    grid_search: GridSearchCV object
        A post-fit (re-fitted to full dataset) GridSearchCV object where the estimator is a Pipeline.
    """

    outer_scores = []

    pre_rec_f1_sup = (np.array([0, 0,0], dtype='float64'), np.array([0, 0,0], dtype='float64'),
                      np.array([0, 0,0], dtype='float64'), np.array([0, 0,0], dtype='int64'))

    # Set up grid search configuration
    grid =  GridSearchCV(estimator=model, param_grid=param_grid,
                         cv=inner_cv, scoring=scoring, n_jobs=n_jobs)

    # Set aside a hold-out test dataset for model evaluation
    n = 0
    for k, (training_samples, test_samples) in enumerate(outer_cv.split(X, y)):

        # x training and test datasets
        if isinstance(X, pandas.core.frame.DataFrame):
            x_train = X.iloc[training_samples]
            x_test = X.iloc[test_samples]
        else:  # in case of spare matrices
            x_train = X[training_samples]
            x_test = X[test_samples]

        # y training and test datasets
        if isinstance(y, pandas.core.frame.Series):
            y_train = y.iloc[training_samples]
            y_test = y.iloc[test_samples]
        else: # in case of numpy arrays
            y_train = y[training_samples]
            y_test = y[test_samples]

        # Build classifier on best parameters using outer training set
        # Fit model to entire training dataset (i.e tuning & validation dataset)
        print('fold-', k+1, 'model fitting ...')

        # Train on the training set
        grid.fit(x_train, y_train)

        # Hyper-parameters of the best model
        print(grid.best_estimator_.get_params())

        # Evaluate
        score = grid.score(x_test, y_test)

        outer_scores.append(abs(score))
        print('\tModel validation score', score)

        # Add the precision, recall, f1 score, and support of each outer fold
        # i.e. running total
        all_scores = precision_recall_fscore_support(y_test, grid.predict(x_test))

        pre_rec_f1_sup = [x + y for x, y in zip(pre_rec_f1_sup, all_scores)]

        print(all_scores)

        n += 1

    # Print final model evaluation (i.e. mean cross-validation scores)
    print('\nFinal model evaluation (mean cross-val scores):', np.array(outer_scores).mean())

    # Print cross validated precision, recall, f1 score, and support
    print('\n Final precision, recall, f1 score, and support values:\n',[s/n for s in pre_rec_f1_sup])

    return grid


## Standard nested k-fold cross validation
def nested_grid_search_cv(model, X, y, outer_cv, inner_cv, param_grid,
                          scoring="accuracy", average='weighted',
                          undersampling=False, debug=False):
    """
    Nested k-fold crossvalidation

    Parameters
    ----------
    model : array, shape = [n_samples]
            true class, intergers in [0, n_classes - 1)
    X : array,  shape = [n_samples, n_classes]
    y : array,  shape = [n_samples, n_classes]
    outer_cv :   shape = [n_samples, n_classes]
    inner_cv :   shape = [n_samples, n_classes]
    scoring :    shape = [n_samples, n_classes]
    undersampling : boolean, default False
        determines whether to perform undersampling
    debug : boolean
        turns on print statements used for debugging purposes

    Returns
    -------
    model: classifier re-fitted to full dataset
    """

    outer_scores = []

    outer_model_dict = defaultdict(list)
    outer_scores = dict()

    list_of_models = model_grid_setup(model, param_grid)

    # Outer fold
    for k, (k_training_samples, k_test_samples) in enumerate(outer_cv.split(X, y)):

        inner_model_dict = defaultdict(list)
        inner_scores = dict()

        # Build classifier on best parameters using outer training set
        # Fit model to entire training dataset (i.e tuning & validation dataset)
        print('outer fold-', k+1, ' ...', sep='')

        # x training and test data sets
        if isinstance(X, pandas.core.frame.DataFrame):
            x_train = X.iloc[k_training_samples]
            x_test = X.iloc[k_test_samples]
        else:  # in case of spare matrices
            x_train = X[k_training_samples]
            x_test = X[k_test_samples]

        # y training and test data sets
        if isinstance(y, pandas.core.frame.Series):
            y_train = y.iloc[k_training_samples]
            y_test = y.iloc[k_test_samples]
        else: # in case of numpy arrays
            y_train = y[k_training_samples]
            y_test = y[k_test_samples]

        # Inner fold
        for m, (m_training_samples, m_test_samples) in enumerate(inner_cv.split(x_train, y_train)):
            indices = []

            print('\n\tinner fold-', m+1, ' model fitting ...', sep='')

            # y training and test data sets
            if isinstance(y, pandas.core.frame.Series):
                # find indices for undersampling
                indices = None if not undersampling else random_indices(y_train.iloc[m_training_samples])

                y_train_val = y_train.iloc[m_training_samples] if not undersampling else y_train.loc[indices]
                y_test_val = y_train.iloc[m_test_samples]

            else: # in case of numpy arrays
                y_train_val = y_train[m_training_samples]
                y_test_val = y_train[m_test_samples]

            # x training and test data sets
            if isinstance(X, pandas.core.frame.DataFrame):
                x_train_val = x_train.iloc[m_training_samples] if not undersampling else x_train.loc[indices]
                x_test_val = x_train.iloc[m_test_samples]

            else:  # in case of spare matrices
                x_train_val = x_train[m_training_samples]
                x_test_val = x_train[m_test_samples]

            # Train on the validation training dataset
            for i, ml in enumerate(list_of_models):
                ml.fit(x_train_val, y_train_val)

                # Evaluate performance measurement for inner fold
                score = model_score(ml, x_test_val, y_test_val, scoring=scoring)
                if debug:
                    print('\t\tModel', i, 'validation score:', score)

                # Store into inner model dict
                inner_model_dict[i].append(score)

        # Find mean model scores
        inner_model_dict = dict(inner_model_dict.items())

        if debug:
            print('\n\tMean valadition scores ...')

        for n in inner_model_dict:
            inner_scores[n] = np.mean(inner_model_dict[n])

            if debug:
                print('\tModel', n, 'mean validation score:', inner_scores[n])

        # Find model with best score
        model_num = score_selector(inner_scores, scoring=scoring)

        # Print final hyper-parameter evaluation (i.e. mean cross-validation scores)
        score = inner_scores[model_num]
        print('\n\tModel',  model_num, 'has the best hyper-parameter evaluation (mean score):', score, '\n')

        # Find indices for undersampling
        indices = None if not undersampling else random_indices(y_train)

        y_train_outer = y_train if not undersampling else y_train.loc[indices]
        x_train_outer = x_train if not undersampling else x_train.loc[indices]

        # Fit best model
        best_model = list_of_models[model_num].fit(x_train_outer, y_train_outer)

        # Evaluate performance measurement for outer fold
        score = model_score(best_model, x_test, y_test, scoring=scoring, average=average)

        # Store into outer scores
        outer_model_dict[model_num].append(score)

    outer_model_dict = dict(outer_model_dict.items())

    model_num = model_with_most_scores(outer_model_dict)

    # Print final model evaluation (i.e. mean cross-validation scores)
    score = np.array(outer_model_dict[model_num]).mean()

    print('Model', model_num, 'final evaluation (mean cross-val scores):', score, '\n')

    # Fit model to entire dataset (i.e tuning & validation dataset)
    indices = None if not undersampling else random_indices(y)

    y_final = y if not undersampling else y.loc[indices]
    X_final = X if not undersampling else X.loc[indices]

    final_model = list_of_models[model_num].fit(X_final, y_final)

    print('Best Hyper-parameter values:\n')
    print(final_model.get_params())
    print('')

    return final_model


