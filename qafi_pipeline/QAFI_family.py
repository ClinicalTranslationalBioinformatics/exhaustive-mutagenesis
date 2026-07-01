import os
import sys
import math
import random
from collections import Counter
from itertools import repeat
import warnings

import numpy as np
import pandas as pd
import scipy as sc
from scipy.stats import spearmanr, pearsonr, norm

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression, Lasso, LassoCV
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import RepeatedKFold, GridSearchCV
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning

from imblearn.under_sampling import RandomUnderSampler
from yellowbrick.regressor import AlphaSelection

from os import listdir
from os.path import isfile, join

from Bio.SeqUtils import seq1
from Bio.PDB import *

from xgboost import XGBRegressor

import warnings
import numpy as np
import warnings
warnings.filterwarnings("ignore")



############################################################################################################################################

############################################################################################################################################

def normalize_features(train_x, features_cols):
    """
    Normalize the training set using MinMaxScaler.

    Parameters:
    train_x (pd.DataFrame): The training data.
    features_cols (list): List of feature column names.

    Returns:
    pd.DataFrame: Normalized training data.
    """
    # normalize training set
    scaler_train = MinMaxScaler()
    train_x_transformed = scaler_train.fit_transform(train_x)
    train_x_transformed = pd.DataFrame(train_x_transformed, columns=features_cols)

    return train_x_transformed




def feature_plot_prep_lasso(DB, collect, features_list):
    """
    Prepares a DataFrame for feature plotting from the Lasso model results.

    Parameters:
    DB (pd.DataFrame): DataFrame containing the dataset.
    collect (dict): Dictionary with best alpha and coefficients for each protein.
    features_list (list): List of feature names.

    Returns:
    pd.DataFrame: DataFrame with proteins, alphas, features, and their corresponding values.
    """
    mykeys, myvals, myalphas =[],[],[]

    for k in collect.keys():
        mykeys.extend(repeat(k,len(features_list)))

    for val in collect.values():
        myvals.extend(val[1])
        myalphas.extend(repeat(val[0], len(features_list)))

    collect_df = pd.DataFrame({
        'proteins': mykeys,
        'alphas': myalphas,
        'features': features_list * len(DB.protein.unique()),
        'values': myvals})

    return collect_df


############################################################################################################################################
# Protein-Specific Predictor
############################################################################################################################################




############################################################################################################################################

def train_test_prepare(train_db, test_db, features, target, undersample):

    # check train/test are splitted correctly
    if len(train_db) < len(test_db):
        sys.exit('ERROR. Training set is smaller than testing.')
    if len(test_db) > 1 and len(test_db.pos.unique()) != 1:
        sys.exit('ERROR. Test set has more than one position.')

    if undersample:
        # get the threshold
        threshold_prot, _ = GMM_interaction(train_db)

        # undersample the training set
        train_db_undersampled, counting = Undersample_Assay(train_db, threshold_prot)

        # train sets
        train_x, train_y = train_db_undersampled[features], train_db_undersampled[target]

    else:
        threshold_prot, _ = GMM_interaction(train_db)
        # train sets
        train_x, train_y = train_db[features], train_db[target]
        counting = ['-', '-', '-', '-']

    # test set
    test_x = test_db[features]

    # feature scaling
    train_x_scaled, test_x_scaled = normalize_features_train_test(train_x, test_x, features)

    # ==== Clean up NaN / inf values ====
    train_x_scaled = np.nan_to_num(train_x_scaled, nan=0.0, posinf=0.0, neginf=0.0)
    test_x_scaled  = np.nan_to_num(test_x_scaled,  nan=0.0, posinf=0.0, neginf=0.0)

    return train_x_scaled, train_y, test_x_scaled, threshold_prot, counting
############################################################################################################################################
def normalize_features_train_test(train_x, test_x, features_cols):
    """
    Normalizes the training and testing datasets using MinMaxScaler.
    Returns:
        - pd.DataFrame: Normalized training dataset.
        - pd.DataFrame: Normalized testing dataset.
    """
    # Normalize training set
    scaler_train = MinMaxScaler()
    train_x_transformed = scaler_train.fit_transform(train_x)
    train_x_transformed = pd.DataFrame(train_x_transformed, columns=features_cols)

    # Normalize testing set using the scaler fitted on the training set
    test_x_transformed = scaler_train.transform(test_x)
    test_x_transformed = pd.DataFrame(test_x_transformed, columns=features_cols)

    return train_x_transformed, test_x_transformed

############################################################################################################################################

def GMM_interaction(db_prot, plot=False):
    """
    Build Gaussian Mixture Models with two components for continuous data.

    Returns
    -------
    threshold : float
        The intersection point that separates two peaks.
    plot_files : list
        If plot=True, returns values required for plotting.
    """
    prot = db_prot.protein.unique()[0]
    X = db_prot['score_log_normalized'].to_numpy().reshape(-1, 1)

    # === Sanity check ===
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)  # handle NaN/inf
    if np.all(X == X[0]):  # all values identical
        return float(X[0]), []

    # Range for plotting
    x = np.linspace(np.min(X) - 0.1, np.max(X) + 0.1, 1000)

    # === GMM fit ===
    gmm = GaussianMixture(
        n_components=2,
        random_state=23,
        init_params="random",  # avoid KMeans overflow
        n_init=5
    ).fit(X)

    m1, m2 = gmm.means_
    w1, w2 = gmm.weights_
    c1, c2 = gmm.covariances_
    std1 = np.sqrt(c1[0][0])
    std2 = np.sqrt(c2[0][0])

    y1 = sc.stats.norm.pdf(x, m1[0], std1)
    y2 = sc.stats.norm.pdf(x, m2[0], std2)

    # Find intersections
    idxs = np.argwhere(np.diff(np.sign(y1 * w1 - y2 * w2))).flatten()

    # === Threshold selection ===
    if len(idxs) == 0:
        # fallback: midpoint between means
        threshold = round(float((m1[0] + m2[0]) / 2), 2)
    else:
        special_proteins = ['HXK4', 'NUD15', 'OTC', 'TRPC',
                            'MAPK1', 'SRC', 'TPK1', 'UBE2I']
        if len(idxs) > 1 and prot in special_proteins:
            threshold = round(float(x[idxs[1]]), 2)
        else:
            threshold = round(float(x[idxs[0]]), 2)

    plot_files = [gmm, X, x, y1, y2, w1, w2, idxs] if plot else []

    return threshold, plot_files
############################################################################################################################################


def Undersample_Assay(db_org, threshold):
    """
    Performs undersampling on a dataset based on a given threshold.

    Parameters
    ----------
    db_org : pd.DataFrame
        Original dataset.
    threshold : float
        Threshold for categorizing the score_log_normalized values.

    Returns
    -------
    tuple
        - pd.DataFrame: Undersampled dataset.
        - list: Group distributions before and after undersampling.
    """
    db = db_org.copy()

    # Initialize group column as object dtype (can hold strings safely)
    db['group'] = pd.Series([np.nan] * len(db), dtype="object")

    # Assign groups based on threshold
    db.loc[db.score_log_normalized < threshold, 'group'] = 'Below thr.'
    db.loc[db.score_log_normalized >= threshold, 'group'] = 'Above thr.'

    x, y = db[db_org.columns], db['group']

    # Perform undersampling
    undersample = RandomUnderSampler(sampling_strategy='majority', random_state=1234)
    x_under, y_under = undersample.fit_resample(x, y)
    db_undersampled = x_under.copy()

    # Check if undersampling was successful
    if len(db_undersampled) >= len(db):
        sys.exit('❌ undersampling failed.')

    # Group distributions before and after undersampling
    counts_before = Counter(y)
    counts_after = Counter(y_under)

    group1_before, group2_before = list(counts_before.items())
    group1_after, group2_after = list(counts_after.items())

    return db_undersampled, [group1_before, group2_before, group1_after, group2_after]


############################################################################################################################################


def calculate_stats(db_protein_predicted, target, predictor_name):
    """
    Calculates Pearson and Spearman correlation coefficients between the target and predictor.

    Parameters:
    db_protein_predicted (pd.DataFrame): DataFrame containing the predicted values.
    target (str): Name of the target variable.
    predictor_name (str): Name of the predictor variable.

    Returns:
    tuple: Tuple containing:
        - float: Pearson correlation coefficient rounded to 2 decimal places.
        - float: Spearman correlation coefficient rounded to 2 decimal places.
    """
    db_protein_predicted = db_protein_predicted.reset_index(drop=True)

    r, p = pearsonr(db_protein_predicted[target],db_protein_predicted[predictor_name])
    rho, p = spearmanr(db_protein_predicted[target],db_protein_predicted[predictor_name])
    print(f'{db_protein_predicted.protein.unique()[0]}\npearson: {round(r,2)}, spearman: {round(rho,2)}\n__________________________\n\n')

    # Return rounded correlation coefficients
    return round(r,2), round(rho,2)

############################################################################################################################################
# Cross Prediction
############################################################################################################################################

def train_test_prepare_cross(train_db, test_db, features, target, undersample):

    """
    Prepares training and testing datasets, with an option to undersample the training set.
    """

    if undersample == True:
        # get the threshold
        threshold_prot, _ = GMM_interaction(train_db)

        # undersample the training set
        train_db_undersampled, counting = Undersample_Assay(train_db, threshold_prot)
        # train sets
        train_x, train_y = train_db_undersampled[features], train_db_undersampled[target]

    elif undersample == False:
        # train sets
        train_x, train_y = train_db[features], train_db[target]
        counting = ['-','-','-','-']
    threshold_prot, _ = GMM_interaction(train_db)
    # test set
    test_x = test_db[features]
    # feature scaling
    train_x_scaled, test_x_scaled = normalize_features_train_test(train_x, test_x, features)

    return train_x_scaled, train_y, test_x_scaled, threshold_prot, counting


############################################################################################################################################

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import pandas as pd

def Cross_Predictions(DB, target, method="mlr"):
    """
    Perform cross-predictions using different regression models (MLR, RFR, or XGB).

    Parameters:
    - DB (DataFrame): The dataset containing protein data.
    - target (str): The target column name.
    - method (str): Regression model to use: 'mlr', 'rfr', or 'xgb'.

    Returns:
    - cross_predictions (DataFrame): DataFrame containing cross-prediction statistics.
    """

    # Fixed parameters
    prot_list = DB.protein.unique()
    features = [
        'Blosum62', 'PSSM', "Shannon's entropy", "Shannon's entropy of seq. neighbours",
        'pLDDT', 'pLDDT bin', 'colasi', 'fraction cons. 3D neighbor', 'fanc', 'fbnc',
        'M.J. potential', 'access.dependent vol.', 'neco', 'laar'
    ]
    predictor_name = f"qafi_{method}"
    undersample = False

    # Print information about the run
    print("=======================================")
    print("Selected features:", features)
    print("Target:", target)
    print("Machine learning method:", method.upper())
    print("=======================================")

    cp_list = []

    # Loop over training proteins
    for prot_train in prot_list:
        db_protein_train = DB[DB.protein == prot_train].reset_index(drop=True).copy()

        # Test on all other proteins
        prot_test_list = [p for p in prot_list if p != prot_train]
        for prot_test in prot_test_list:
            db_protein_test = DB[DB.protein == prot_test].reset_index(drop=True).copy()

            # Prepare train/test sets
            train_x_scaled, train_y, test_x_scaled, _, _ = train_test_prepare_cross(
                db_protein_train, db_protein_test, features, target, undersample
            )

            # Choose the model
            if method == "mlr":
                model = LinearRegression(fit_intercept=True)
            elif method == "rfr":
                model = RandomForestRegressor(
                    max_depth=75,
                    min_samples_leaf=4,
                    min_samples_split=10,
                    n_estimators=100,
                    random_state=12234
                )
            elif method == "xgb":
                model = XGBRegressor(
                    n_estimators=300,
                    learning_rate=0.05,
                    max_depth=6,
                    min_child_weight=3,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    gamma=0,
                    reg_alpha=0.1,
                    reg_lambda=1.0,
                    objective="reg:squarederror",
                    random_state=12234,
                    verbosity=0
                )
            else:
                raise ValueError("method must be 'mlr', 'rfr', or 'xgb'")

            # Train & predict
            model.fit(train_x_scaled, train_y)
            db_protein_test[predictor_name] = model.predict(test_x_scaled)

            # Calculate Pearson and Spearman correlations
            r, rho = calculate_stats(db_protein_test, target, predictor_name)

            # Save results
            cp_list.append([predictor_name, prot_train, prot_test, r, rho])

    # Return results as a DataFrame
    cross_predictions = pd.DataFrame(
        cp_list, columns=["model", "trained_protein", "tested_protein", "pearson", "spearman"]
    )

    return cross_predictions


############################################################################################################################################


def tested_protein_csvs(DB, proteins_list, target,predictor_name, path_save_trained, path_save_tested):
    """
    Collect the real prediction values for each trained_protein/predicted_protein combination and save to CSV files.

    Parameters:
    - DB (DataFrame): The dataset containing protein data.
    - proteins_list (list): List of proteins to be used for testing.
    - target (str): The target column name.
    - predictor_name (str): Name of the predictor column.
    - path_save_trained (str): Path where trained model predictions are saved.
    - path_save_tested (str): Path where the collected predictions will be saved.

    Returns:
    - None: The function saves the output CSV files to the specified path.
    """

    for test_protein in proteins_list:
        # Select the test protein data
        tested = DB[DB.protein==test_protein][['protein','variant',target]]
        trained_prots = [t for t in proteins_list if t != test_protein]

        print(f'test protein: {test_protein} // collect preds. from:{len(trained_prots),trained_prots}\n')

        for train in trained_prots:
            # Read the trained protein predictions
            pp = pd.read_csv(f'{path_save_trained}train_{train}_predict_rest.csv')

            # Filter predictions for the current test protein
            train_preds = pp[pp.tested_protein==test_protein]
            # Merge predictions with the test data
            tested = tested.merge(train_preds[['variant',target,predictor_name]], on=['variant', target], how='left')
            tested.rename(columns={predictor_name:predictor_name+ '_trainedby_'+train}, inplace=True)

        # Rename the target column for clarity
        tested.rename(columns={target:target+'_'+test_protein},inplace=True)

        tested.to_csv(path_save_tested + test_protein + '.csv', index=0)
############################################################################################################################################

def function_cross_preds_median(path_save_tested, stats_all, predictor_name, target, proteins_to_be_tested,howmany):

    """
    Perform cross-predictions using the median of the best predictors for each tested protein.

    Parameters:
    - path_save_tested (str): Path where the tested predictions are saved.
    - stats_all (DataFrame): DataFrame containing statistics of all cross-predictions.
    - predictor_name (str): Name of the predictor column.
    - target (str): The target column name.
    - proteins_to_be_tested (list): List of proteins to be tested.
    - howmany (int): Number of top predictors to use for the median prediction.

    Returns:
    - stat_table (DataFrame): DataFrame containing statistics of the median model predictions.
    """

    print(f'{predictor_name}....\n\n')

    stat_table = pd.DataFrame(columns=['tested_protein', 'MedianModel','pearson','spearman','median_of_which_proteins'])

    for tested_protein in proteins_to_be_tested:
        # ROUND 1: SELECTING THE BEST PROTEINS FOR THE MEDIAN
        print(f'___________________________________________________________________________\n:::::::ROUND 1::::::: \n we are going to find who are the best {howmany} predictors for predicting {tested_protein}\n___________________________________________________________________________\n\n')
        proteins_rest = [p for p in proteins_to_be_tested if p != tested_protein]
        p_col,median_col = [],[]
        for p in proteins_rest:
            trained = p
            tested = [rest for rest in proteins_rest if rest != p]
            print(f'TRAIN: {trained}\nTESTED:{tested}')
            collect_pearsons = []
            for t in tested:
                pearson_t = stats_all.loc[(stats_all.trained_protein == p) & (stats_all.tested_protein==t)]['pearson'].values[0]
                #print(f'\t{t}: {pearson_t}')
                collect_pearsons.append(pearson_t)
            median_pearsons = np.median(collect_pearsons)
            print(f'\nMedian of pearsons: {median_pearsons}')
            print('_____________\n')
            p_col.append(p)
            median_col.append(median_pearsons)
        db = pd.DataFrame({'proteins':p_col, 'median_pearsons':median_col})
        db = db.sort_values(by='median_pearsons', ascending=False)
        top_proteins = db.head(howmany).proteins.values
        print(f'best predictors for {tested_protein} are: {top_proteins}\n')
        print(f'___________________________________________________________________________\n:::::::ROUND 2::::::: \n we are going to predict {tested_protein} using median of {top_proteins}\n___________________________________________________________________________\n\n')

        stat_table_perprot = cross_preds_median_one(tested_protein, path_save_tested, top_proteins, predictor_name, target)
        stat_table_perprot['median_of_which_proteins'] = str(top_proteins)


        stat_table = pd.concat([stat_table, stat_table_perprot])

        print('- - - - - - - - - - - - - - - - - - - - - - - \n- - - - - - - - - - - - - - - - - - - - - - -\n\n')
    return stat_table.reset_index(drop=True)

############################################################################################################################################

def cross_preds_median_one(tested_protein, path_save_tested, median_prots_list, predictor_name, target):

    """
    tested_protein = this is the protein you would like to predict
    median_prots_list = list of proteins you want to get predictions from, and to take their predictions' median
    """

    print(f'Tested protein:\t{tested_protein}\n\nProteins selected for median:\n\n  {sorted(median_prots_list)}\n\n')
    cp_list = []

    tested = pd.read_csv(path_save_tested + tested_protein + '.csv')

    pred = predictor_name.split("_")[0]

    newname = f'QAFI({pred}_median_'+str(len(median_prots_list))+')'

    names = [predictor_name + '_trainedby_'+ prot for prot in median_prots_list]
    tested[newname] = tested[names].median(axis=1)
    tested[newname] = round(tested[newname],3 )
    tested['total_medians'+str(len(median_prots_list))] = str(median_prots_list)

    median_pred_name = newname
    target_name = target + '_' + str(tested_protein)

    r, p = pearsonr(tested[target_name],tested[median_pred_name])
    rho, p = spearmanr(tested[target_name],tested[median_pred_name])

    r, rho = round(r,2), round(rho,2)
    print(f'output column name: // {newname} // \n\nmedian of {len(median_prots_list)} predictions: r = {r}, rho = {rho}')

    tested.to_csv(f'{path_save_tested}{tested_protein}_median{len(median_prots_list)}.csv', index=0)

    cp_list.append([tested_protein, median_pred_name, r, rho])
    stat_table = pd.DataFrame(cp_list, columns=['tested_protein', 'MedianModel','pearson','spearman'])

    print('_________________________________________________________________\n')

    return stat_table

############################################################################################################################################

def count_protein_occurrences(input_array):
    """
    Creates a table with the count of each protein name in a numpy array of protein lists
    and returns the top N most frequent proteins.
    """

    all_proteins = []
    for proteins_str in input_array:
        proteins_list = proteins_str.strip("[]").replace("'", "").split()
        all_proteins.extend(proteins_list)

    protein_counts = pd.Series(all_proteins).value_counts().reset_index()
    protein_counts.columns = ['protein', 'count']

    return protein_counts

############################################################################################################################################
# PSP-family
############################################################################################################################################


def pspbase(df, target, features, predictor_name, method="mlr", undersample=False, zero_low_pLDDT=False):
    """
    Leave-One-Protein-Out + Leave-One-Position-Out (LOPO) cross-validation.
    """

    # Mapping for professional names
    method_names = {
        "mlr": "Multiple Linear Regression (MLR)",
        "xgb": "Extreme Gradient Boosting (XGBoost)",
        "rfr": "Random Forest Regressor (RFR)"
    }

    if method not in method_names:
        sys.exit(f"Unknown method: {method}. Use 'mlr', 'xgb', or 'rfr'.")

    # Apply zeroing if required
    features_to_zero = [
        'colasi', 'fraction cons. 3D neighbor', 'fanc', 'fbnc',
        'colasi_pdff', 'fraction cons. 3D neighbor_pdff', 'fanc_pdff',
        'e_native_pdff', 'NCE-NR', 'fbnc_pdff', 'M.J. potential',
        'access.dependent vol.', 'neco', 'neco2', 'neco3', 'neco12'
    ]
    if zero_low_pLDDT:
        df.loc[df['pLDDT bin'] == 0, features_to_zero] = 0

    # ==== Print settings ====
    print("=" * 60)
    print(" PSP - Model Configuration")
    print("=" * 60)
    print(f" ▶ Target Variable     : {target}")
    print(f" ▶ Method              : {method_names[method]}")
    print(f" ▶ Predictor Name      : {predictor_name}")
    print(f" ▶ Undersample         : {undersample}")
    print(f" ▶ Zero low pLDDT feat.: {zero_low_pLDDT}")
    print("-" * 60)
    print(" ▶ Features Used:")
    for f in features:
        print(f"    - {f}")
    print("=" * 60 + "\n")

    all_predictions = []

    # ===== Loop over proteins =====
    for idx, uni in enumerate(df['uniprot'].unique(), 1):
        protein_name = df.loc[df['uniprot'] == uni, 'protein'].iloc[0]
        print(f"[{idx}/{df['uniprot'].nunique()}] Processing protein: {protein_name} (UniProt: {uni})")
    
        db_protein = df[df['uniprot'] == uni].reset_index(drop=True)
        db_protein_predicted = pd.DataFrame()

        # ===== Loop over positions =====
        for position in db_protein.pos.unique():
            train_db = db_protein[db_protein.pos != position].reset_index(drop=True).copy()
            test_db  = db_protein[db_protein.pos == position].reset_index(drop=True).copy()

            train_x_scaled, train_y, test_x_scaled, _, _ = train_test_prepare(
                train_db, test_db, features, target, undersample
            )

            # ===== Choose model =====
            if method == "mlr":
                model = LinearRegression(fit_intercept=True)

            elif method == "xgb":
                model = XGBRegressor(
                    n_estimators=300,
                    learning_rate=0.05,
                    max_depth=6,
                    min_child_weight=3,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    gamma=0,
                    reg_alpha=0.1,
                    reg_lambda=1.0,
                    objective="reg:squarederror",
                    random_state=12234,
                    verbosity=0
                )

            elif method == "rfr":
                model = RandomForestRegressor(
                    max_depth=75,
                    min_samples_leaf=4,
                    min_samples_split=10,
                    n_estimators=100,
                    random_state=12234
                )

            # Train & predict
            model.fit(train_x_scaled, train_y)
            test_db[predictor_name] = [round(a, 3) for a in model.predict(test_x_scaled)]

            db_protein_predicted = pd.concat([db_protein_predicted, test_db])

        db_protein_predicted = db_protein_predicted.reset_index(drop=True)

        # Keep only required columns
        db_protein_predicted = db_protein_predicted[["uniprot", "protein", "variant", predictor_name]]

        all_predictions.append(db_protein_predicted)

    # Concatenate all proteins
    DB_PREDICTED = pd.concat(all_predictions, ignore_index=True)
    return DB_PREDICTED


####################################           pspsplit       ####################################

from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
import pandas as pd

def pspsplit(df, target_variant, features, position_features,
             predictor_name="pspsplit1", undersample=False, zero_low_pLDDT=False):
    """
    Unified LOPO modeling (per protein) with multiple predictor options.

    predictor_name options
    ----------------------
    - "pspsplit1"        : Two-step (MLR + scaler for median → XGB main with median_feature)
    - "pspsplit2"        : Three-step (MLR median → XGB residual → Fusion = sum)
    - "pspsplit2_obs"    : Three-step (MLR median → XGB residual using true median → Fusion = sum)
    - "pspsplit2_fusion" : Three-step (MLR median → XGB residual → Fusion with LR / BR / XGB)

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset, must contain 'protein', 'uniprot', 'pos',
        the target variable, and all features.
    target_variant : str
        Variant-level target variable (e.g., "score_log_normalized").
    features : list
        List of variant-level features.
    position_features : list
        List of position-level features.
    predictor_name : str, default="pspsplit1"
        Which predictor to run ("pspsplit1", "pspsplit2", "pspsplit2_obs", "pspsplit2_fusion").
    undersample : bool, default=False
        Whether to apply undersampling (placeholder, not used here).
    zero_low_pLDDT : bool, default=False
        If True, set selected features to zero where 'pLDDT bin' == 0.

    Returns
    -------
    pd.DataFrame
        DataFrame containing predictions with columns depending on predictor_name.
    """

    # === Print configuration ===
    print("=" * 60)
    print(f"▶ Running {predictor_name}")
    print("-" * 60)
    print(f"Target variable : {target_variant}")
    print(f"Undersample     : {undersample}")
    print(f"Zero low pLDDT  : {zero_low_pLDDT}")
    print("Position features:")
    for f in position_features: print(f"   - {f}")
    print("Variant features:")
    for f in features: print(f"   - {f}")
    print("-" * 60)

    if predictor_name == "pspsplit1":
        print(" Step 1: MLR + StandardScaler → predict median per position")
        print(" Step 2: XGBoost → main model with median_feature")
    elif predictor_name == "pspsplit2":
        print(" Step 1: MLR → predict median per position")
        print(" Step 2: XGBoost → residual = observed - median_pred")
        print(" Step 3: Fusion → final = median_pred + residual_pred")
    elif predictor_name == "pspsplit2_obs":
        print(" Step 1: MLR → predict median per position")
        print(" Step 2: XGBoost → residual = observed - true median")
        print(" Step 3: Fusion → final = median_pred + residual_pred")
    elif predictor_name == "pspsplit2_fusion":
        print(" Step 1: MLR → predict median per position")
        print(" Step 2: XGBoost → residual = observed - median_pred")
        print(" Step 3: Fusion → LR, BayesianRidge, XGB")
    else:
        raise ValueError(f"Unknown predictor_name: {predictor_name}")
    print("=" * 60 + "\n")

    # === Apply zeroing if requested ===
    features_to_zero = [
        'colasi', 'fraction cons. 3D neighbor', 'fanc', 'fbnc',
        'colasi_pdff', 'fraction cons. 3D neighbor_pdff', 'fanc_pdff',
        'e_native_pdff', 'NCE-NR', 'fbnc_pdff', 'M.J. potential',
        'access.dependent vol.', 'neco', 'neco2', 'neco3', 'neco12'
    ]
    if zero_low_pLDDT:
        df.loc[df['pLDDT bin'] == 0, features_to_zero] = 0

    # === Core function per protein ===
    def run_model(db_protein):
        db_pred = pd.DataFrame()
        target_position = f"median_{target_variant}"

        for pos in db_protein.pos.unique():
            # LOPO split
            train_db = db_protein[db_protein.pos != pos].reset_index(drop=True)
            test_db = db_protein[db_protein.pos == pos].reset_index(drop=True)

            # Step 1: Position-level median modeling
            train_pos_level = train_db.groupby('pos', as_index=False).agg(
                {**{f: 'first' for f in position_features}, target_variant: 'median'}
            ).rename(columns={target_variant: target_position})

            X_pos, y_pos = train_pos_level[position_features].values, train_pos_level[target_position].values

            if predictor_name == "pspsplit1":
                # Use StandardScaler for MLR
                scaler_pos = StandardScaler()
                model_pos = LinearRegression().fit(scaler_pos.fit_transform(X_pos), y_pos)
                test_pred_val = model_pos.predict(scaler_pos.transform(test_db[position_features].iloc[[0]].values))[0]
                train_db['median_pos_predicted'] = model_pos.predict(scaler_pos.transform(train_db[position_features].values))
            else:
                # Direct MLR without scaling
                model_pos = LinearRegression().fit(X_pos, y_pos)
                test_pred_val = model_pos.predict(test_db[position_features].iloc[[0]].values)[0]
                train_db['median_pos_predicted'] = model_pos.predict(train_db[position_features].values)

            test_db['median_pos_predicted'] = test_pred_val
            train_db = train_db.merge(train_pos_level[['pos', target_position]], on='pos', how='left')

            # Step 2 & Step 3 depending on predictor_name
            if predictor_name == "pspsplit1":
                # Main model with median_feature
                train_db['median_feature'] = train_db['median_pos_predicted']
                test_db['median_feature'] = test_db['median_pos_predicted']
                model_final = XGBRegressor(n_estimators=300, learning_rate=0.03, max_depth=3,
                                           min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
                                           reg_alpha=0.5, reg_lambda=1.0, objective='reg:squarederror',
                                           random_state=12234, verbosity=0)
                model_final.fit(train_db[features + ['median_feature']], train_db[target_variant])
                test_db[predictor_name] = model_final.predict(test_db[features + ['median_feature']])

            elif predictor_name == "pspsplit2":
                # Residual modeling
                train_db['residual'] = train_db[target_variant] - train_db['median_pos_predicted']
                model_resid = XGBRegressor(n_estimators=300, learning_rate=0.03, max_depth=3,
                                           min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
                                           reg_alpha=0.5, reg_lambda=1.0, objective='reg:squarederror',
                                           random_state=12234, verbosity=0)
                model_resid.fit(train_db[features + ['median_pos_predicted']], train_db['residual'])
                resid_pred = model_resid.predict(test_db[features + ['median_pos_predicted']])
                test_db['residual_predicted'] = resid_pred
                test_db[predictor_name] = test_db['median_pos_predicted'] + resid_pred

            elif predictor_name == "pspsplit2_obs":
                # Residual = observed - true median
                train_db['residual'] = train_db[target_variant] - train_db.groupby('pos')[target_variant].transform('median')
                model_resid = XGBRegressor(n_estimators=300, learning_rate=0.03, max_depth=3,
                                           min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
                                           reg_alpha=0.5, reg_lambda=1.0, objective='reg:squarederror',
                                           random_state=12234, verbosity=0)
                model_resid.fit(train_db[features + ['median_pos_predicted']], train_db['residual'])
                resid_pred = model_resid.predict(test_db[features + ['median_pos_predicted']])
                test_db['residual_predicted'] = resid_pred
                test_db[predictor_name] = test_db['median_pos_predicted'] + resid_pred

            elif predictor_name == "pspsplit2_fusion":
                # Residual + Fusion
                train_db['residual'] = train_db[target_variant] - train_db['median_pos_predicted']
                model_resid = XGBRegressor(n_estimators=300, learning_rate=0.03, max_depth=3,
                                           min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
                                           reg_alpha=0.5, reg_lambda=1.0, objective='reg:squarederror',
                                           random_state=12234, verbosity=0)
                model_resid.fit(train_db[features + ['median_pos_predicted']], train_db['residual'])
                resid_pred = model_resid.predict(test_db[features + ['median_pos_predicted']])
                test_db['residual_predicted'] = resid_pred

                # Fusion training
                X_fuse_train = train_db[['median_pos_predicted']].copy()
                X_fuse_train['residual_predicted'] = model_resid.predict(train_db[features + ['median_pos_predicted']])
                y_fuse_train = train_db[target_variant].values
                X_fuse_test = pd.DataFrame([[test_pred_val, resid_pred[0]]],
                                           columns=['median_pos_predicted', 'residual_predicted'])

                # LR fusion
                model_lr = LinearRegression().fit(X_fuse_train, y_fuse_train)
                test_db[predictor_name + "_lr"] = model_lr.predict(X_fuse_test)[0]

                # Bayesian Ridge fusion
                model_br = BayesianRidge().fit(X_fuse_train, y_fuse_train)
                test_db[predictor_name + "_br"] = model_br.predict(X_fuse_test)[0]

                # XGB fusion
                model_xgb_fuse = XGBRegressor(n_estimators=30, max_depth=2, learning_rate=0.1, verbosity=0)
                model_xgb_fuse.fit(X_fuse_train.values, y_fuse_train)
                test_db[predictor_name + "_xgb"] = model_xgb_fuse.predict(X_fuse_test.values)[0]

            db_pred = pd.concat([db_pred, test_db])

        return db_pred.reset_index(drop=True)

    # Run for all proteins
    all_results = []
    for idx, uni in enumerate(df['uniprot'].unique(), 1):
        protein_name = df.loc[df['uniprot'] == uni, 'protein'].iloc[0]
        print(f"[{idx}/{df['uniprot'].nunique()}] Processing protein: {protein_name} (UniProt: {uni})")

        db_uni = df[df.uniprot == uni].reset_index(drop=True)
        result_uni = run_model(db_uni)
        all_results.append(result_uni)

    DB_PREDICTED = pd.concat(all_results, ignore_index=True)

    # Keep only relevant columns
    if predictor_name == "pspsplit1":
        return DB_PREDICTED[['uniprot', 'protein', 'variant', 'median_pos_predicted', predictor_name]]
    elif predictor_name == "pspsplit2":
        return DB_PREDICTED[['uniprot', 'protein', 'variant', 'median_pos_predicted', 'residual_predicted', predictor_name]]
    elif predictor_name == "pspsplit2_obs":
        return DB_PREDICTED[['uniprot', 'protein', 'variant', 'median_pos_predicted', 'residual_predicted', predictor_name]]
    elif predictor_name == "pspsplit2_fusion":
        return DB_PREDICTED[['uniprot', 'protein', 'variant', 'median_pos_predicted', 'residual_predicted',
                             predictor_name + "_lr", predictor_name + "_br", predictor_name + "_xgb"]]

        
############################################################################################################################################
# QAFI-family
############################################################################################################################################        


import os
import sys
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import os
import sys
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

def qafibase_Cross_Predictions(DB, path_base, predictor_name,
                               features, target, undersample,
                               method="mlr", zero_low_pLDDT=False):
    """
    Perform cross-predictions using different regression models across proteins.

    Parameters:
    - DB (DataFrame): The dataset containing protein data. Must contain a "protein" column.
    - path_base (str): Base path for outputs; a subfolder named predictor_name will be created.
    - predictor_name (str): Name of the predictor column.
    - features (list): Feature column names.
    - target (str): Target column name.
    - undersample (bool): Whether to undersample training set (handled in train_test_prepare_cross).
    - method (str): One of {"mlr", "rfr", "xgb"}.
    - zero_low_pLDDT (bool): If True, zero selected features where pLDDT bin == 0.

    Returns:
    - cross_predictions (DataFrame): Cross-prediction statistics.
    """

    # --- Build directory tree ---
    base_dir = os.path.join(path_base, predictor_name)
    path_save_trained = os.path.join(base_dir, "train_one_predict_rest")
    path_log = os.path.join(base_dir, "log")
    stats_path = os.path.join(base_dir, "stats_PSP_cross_preds.csv")

    os.makedirs(path_save_trained, exist_ok=True)
    os.makedirs(path_log, exist_ok=True)

    # --- Optional zeroing for low pLDDT regions ---
    if zero_low_pLDDT:
        features_to_zero = [
            'colasi', 'fraction cons. 3D neighbor', 'fanc', 'fbnc',
            'colasi_pdff', 'fraction cons. 3D neighbor_pdff', 'fanc_pdff',
            'e_native_pdff', 'NCE-NR', 'fbnc_pdff', 'M.J. potential',
            'access.dependent vol.', 'neco', 'neco2', 'neco3', 'neco12'
        ]
        DB.loc[DB['pLDDT bin'] == 0, features_to_zero] = 0

    cp_list = []
    debug = pd.DataFrame(columns=[
        'protein_train','GMM_threshold','protein_test',
        'train_before1','train_before2','train_after1','train_after2',
        'train_size','test_size'
    ])

    # --- Protein list from DB ---
    prot_list = DB["protein"].unique()

    for prot_train in prot_list:
        db_protein_train = DB[DB.protein == prot_train].reset_index(drop=True).copy()
        print('==============================================================')
        print('       PROTEIN training:', prot_train, len(db_protein_train))
        print('==============================================================')

        topla = pd.DataFrame()
        for prot_test in [p for p in prot_list if p != prot_train]:
            db_protein_test = DB[DB.protein == prot_test].reset_index(drop=True).copy()
            print(f'testing... {prot_test}')

            train_x_scaled, train_y, test_x_scaled, threshold_prot, counting = train_test_prepare_cross(
                db_protein_train, db_protein_test, features, target, undersample
            )
            if len(db_protein_test) != len(test_x_scaled):
                sys.exit('protein test length different than scaled test x.')

            # --- Choose model ---
            if method == "mlr":
                model = LinearRegression(fit_intercept=True)
            elif method == "rfr":
                model = RandomForestRegressor(
                    max_depth=75, min_samples_leaf=4, min_samples_split=10,
                    n_estimators=100, random_state=12234
                )
            elif method == "xgb":
                model = XGBRegressor(
                    n_estimators=300, learning_rate=0.05, max_depth=6,
                    min_child_weight=3, subsample=0.8, colsample_bytree=0.8,
                    gamma=0, reg_alpha=0.1, reg_lambda=1.0,
                    objective='reg:squarederror', random_state=12234, verbosity=0
                )
            else:
                raise ValueError("method must be 'mlr', 'rfr', or 'xgb'")

            # --- Train & predict ---
            model.fit(train_x_scaled, train_y)
            db_protein_test[predictor_name] = model.predict(test_x_scaled).round(3)

            debug.loc[len(debug)] = [
                prot_train, threshold_prot, prot_test,
                counting[0], counting[1], counting[2], counting[3],
                len(train_x_scaled), len(db_protein_test)
            ]

            # --- Stats ---
            r, rho = calculate_stats(db_protein_test, target, predictor_name)
            cp_list.append([predictor_name, prot_train, prot_test, r, rho])

            # --- Collect rows to save ---
            db_protein_test['tested_protein'] = prot_test
            topla = pd.concat([topla, db_protein_test[['tested_protein','variant',target,predictor_name]]])

        # Save per-trained-protein predictions under .../train_one_predict_rest/
        out_file = os.path.join(path_save_trained, f"train_{prot_train}_predict_rest.csv")
        topla['trained_protein'] = prot_train
        topla.to_csv(out_file, index=False)

    # --- Save stats and debug ---
    cross_predictions = pd.DataFrame(cp_list, columns=['model','trained_protein','tested_protein','pearson','spearman'])
    cross_predictions.to_csv(stats_path, index=False)
    debug.to_csv(os.path.join(path_log, f"CROSS_debug_{predictor_name}.csv"), index=False)

    return cross_predictions
##########################             qafisplit1            ###################################

def qafisplit1_Cross_Predictions(DB, path_base, predictor_name, features, position_features, target):
    """
    Two-step cross-protein prediction:
    Step 1: MLR for per-position median prediction (on training protein only)
    Step 2: XGBoost using main features + median_feature (applied to test protein)

    Parameters
    ----------
    DB : pd.DataFrame
        Input dataframe containing protein mutation data.
    path_base : str
        Base path for saving results.
    predictor_name : str
        Name of the predictor (used for subdirectories).
    features : list
        Main feature columns used in the model.
    position_features : list
        Position-level features for median prediction.
    target : str
        Target variable name.
    """

    from xgboost import XGBRegressor
    import os

    # Set paths automatically
    path_save = os.path.join(path_base, predictor_name, "train_one_predict_rest")
    path_log = os.path.join(path_base, predictor_name, "log")
    os.makedirs(path_save, exist_ok=True)
    os.makedirs(path_log, exist_ok=True)

    # Protein list from DB
    prot_list = DB.protein.unique()

    cp_list = []
    debug = []
    
    for prot_train in prot_list:
        db_protein_train = DB[DB.protein == prot_train].reset_index(drop=True).copy()
        print('==============================================================')
        print('       PROTEIN training:', prot_train, len(db_protein_train))
        print('==============================================================')

        # === Step 1: Train position-level median model on training protein ===
        train_pos_level = db_protein_train.groupby('pos', as_index=False).agg(
            {**{f: 'first' for f in position_features}, target: 'median'})
        X_pos = train_pos_level[position_features].values
        y_pos = train_pos_level[target].values
        model_pos = LinearRegression().fit(X_pos, y_pos)
        train_pos_level['median_pos_predicted'] = model_pos.predict(X_pos)
        # Add predicted median as a new feature
        db_protein_train = db_protein_train.merge(
            train_pos_level[['pos', 'median_pos_predicted']], on='pos', how='left')
        topla = pd.DataFrame()
        prot_test_list = [p for p in prot_list if p != prot_train]

        for prot_test in prot_test_list:
            db_protein_test = DB[DB.protein == prot_test].reset_index(drop=True).copy()
            print(f'testing... {prot_test}')

            # === Predict per-position median in test protein using the training model ===
            test_pos_level = db_protein_test.groupby('pos', as_index=False).first()
            test_pos_level['median_pos_predicted'] = model_pos.predict(
                test_pos_level[position_features].values)
            db_protein_test = db_protein_test.merge(
                test_pos_level[['pos', 'median_pos_predicted']], on='pos', how='left')

            features_plus_median = features + ['median_pos_predicted']
            X_train = db_protein_train[features_plus_median].values
            y_train = db_protein_train[target].values
            X_test = db_protein_test[features_plus_median].values

            # === Step 2: XGBoost main model ===
            model_final = XGBRegressor(
                n_estimators=300,
                learning_rate=0.03,
                max_depth=3,
                min_child_weight=5,
                subsample=0.8,
                colsample_bytree=0.8,
                gamma=0,
                reg_alpha=0.5,
                reg_lambda=1.0,
                objective='reg:squarederror',
                random_state=12234,
                verbosity=0
            )

            model_final.fit(X_train, y_train)
            y_pred = model_final.predict(X_test)
            db_protein_test[predictor_name] = np.round(y_pred, 3)

            r, rho = calculate_stats(db_protein_test, target, predictor_name)
            cp_list.append([predictor_name, prot_train, prot_test, r, rho])

            db_protein_test['tested_protein'] = prot_test
            topla = pd.concat([topla, db_protein_test[['tested_protein', 'variant', target, predictor_name]]])

        topla['trained_protein'] = prot_train
        topla.to_csv(f'{path_save}/train_{prot_train}_predict_rest.csv', index=False)

    cross_predictions = pd.DataFrame(cp_list, columns=['model', 'trained_protein', 'tested_protein', 'pearson', 'spearman'])
    cross_predictions.to_csv(f'{path_log}/CROSS_debug_{predictor_name}.csv', index=False)

    return cross_predictions

##########################             qafisplit2            ###################################

def qafisplit2_Cross_Predictions(DB, path_base, predictor_name,
                                       features, position_features, target):
    """
    Two-step cross-protein prediction:
    Step 1: MLR for per-position median prediction (on training protein only)
    Step 2: XGBoost for residual modeling: (target - predicted_median)
    Final prediction: median_prediction + residual_prediction

    Parameters
    ----------
    DB : pd.DataFrame
        Input dataframe containing protein mutation data.
    path_base : str
        Base directory for saving results.
    predictor_name : str
        Name of the predictor (used for subdirectories).
    features : list
        Feature columns for the main model (used in residual prediction).
    position_features : list
        Position-level features for median prediction.
    target : str
        Target variable name.
    """

    import numpy as np
    import pandas as pd
    import os
    from xgboost import XGBRegressor
    from sklearn.linear_model import LinearRegression

    # Set paths automatically
    path_save = os.path.join(path_base, predictor_name, "train_one_predict_rest")
    path_log = os.path.join(path_base, predictor_name, "log")
    os.makedirs(path_save, exist_ok=True)
    os.makedirs(path_log, exist_ok=True)

    # Protein list from DB
    prot_list = DB.protein.unique()

    cp_list = []
    debug = []

    for prot_train in prot_list:
        db_protein_train = DB[DB.protein == prot_train].reset_index(drop=True).copy()
        print('==============================================================')
        print('       PROTEIN training:', prot_train, len(db_protein_train))
        print('==============================================================')

        # === Step 1: Train position-level median model on training protein ===
        train_pos_level = db_protein_train.groupby('pos', as_index=False).agg(
            {**{f: 'first' for f in position_features}, target: 'median'})
        X_pos = train_pos_level[position_features].values
        y_pos = train_pos_level[target].values
        model_pos = LinearRegression().fit(X_pos, y_pos)
        train_pos_level['median_pos_predicted'] = model_pos.predict(X_pos)

        # Add predicted median as a new feature
        db_protein_train = db_protein_train.merge(
            train_pos_level[['pos', 'median_pos_predicted']], on='pos', how='left')
        # Residual = true value - median prediction
        db_protein_train['residual'] = db_protein_train[target] - db_protein_train['median_pos_predicted']

        topla = pd.DataFrame()
        prot_test_list = [p for p in prot_list if p != prot_train]

        for prot_test in prot_test_list:
            db_protein_test = DB[DB.protein == prot_test].reset_index(drop=True).copy()
            print(f'testing... {prot_test}')

            # === Predict per-position median in test protein using the training model ===
            test_pos_level = db_protein_test.groupby('pos', as_index=False).first()
            test_pos_level['median_pos_predicted'] = model_pos.predict(
                test_pos_level[position_features].values)
            db_protein_test = db_protein_test.merge(
                test_pos_level[['pos', 'median_pos_predicted']], on='pos', how='left')

            # === Step 2: Train XGBoost on residuals (only features are used, not median) ===
            X_train = db_protein_train[features].values
            y_train = db_protein_train['residual'].values
            X_test = db_protein_test[features].values

            model_final = XGBRegressor(
                n_estimators=300,
                learning_rate=0.03,
                max_depth=3,
                min_child_weight=5,
                subsample=0.8,
                colsample_bytree=0.8,
                gamma=0,
                reg_alpha=0.5,
                reg_lambda=1.0,
                objective='reg:squarederror',
                random_state=12234,
                verbosity=0
            )

            model_final.fit(X_train, y_train)
            y_residual_pred = model_final.predict(X_test)

            # === Final prediction = median prediction + residual prediction ===
            db_protein_test[predictor_name] = np.round(
                db_protein_test['median_pos_predicted'] + y_residual_pred, 3)

            r, rho = calculate_stats(db_protein_test, target, predictor_name)
            cp_list.append([predictor_name, prot_train, prot_test, r, rho])

            db_protein_test['tested_protein'] = prot_test
            topla = pd.concat([topla, db_protein_test[['tested_protein', 'variant', target, predictor_name]]])

        topla['trained_protein'] = prot_train
        topla.to_csv(f'{path_save}/train_{prot_train}_predict_rest.csv', index=False)

    cross_predictions = pd.DataFrame(
        cp_list,
        columns=['model', 'trained_protein', 'tested_protein', 'pearson', 'spearman']
    )
    cross_predictions.to_csv(f'{path_log}/CROSS_debug_{predictor_name}.csv', index=False)

    return cross_predictions

#######################################################################################################################
import os

def pipeline_cross_preds(DB, target, predictor_name=None, path_base=None, cross_predictions=None, howmany=None):
    """
    Run the full pipeline:
    1. Create directories
    2. Generate per-protein tested CSVs
    3. Run cross-prediction median model
    4. Count protein occurrences

    Parameters:
    - DB (DataFrame): Input dataset containing proteins and variants.
    - target (str): Target column name.
    - predictor_name (str): Predictor column name.
    - path_base (str): Base path for saving outputs.
    - cross_predictions (DataFrame): Cross-prediction statistics (pearson, spearman, etc.).
    - howmany (int): Number of top predictors to use for the median model.
    - top_n (int): Number of top proteins to return in occurrence counts (default: 30).

    Returns:
    - stats_table (DataFrame): Results of the median model cross-predictions.
    - protein_counts (DataFrame): Count of protein occurrences (top N).
    """

    # ---- Step 1: Create directories ----
    path_save_trained = f'{path_base}/{predictor_name}/train_one_predict_rest/'
    path_save_tested  = f'{path_base}/{predictor_name}/tested_protein_csv/'
    os.makedirs(path_save_trained, exist_ok=True)
    os.makedirs(path_save_tested, exist_ok=True)

    # ---- Step 2: Generate per-protein CSVs ----
    prot_list = DB.protein.unique()
    tested_protein_csvs(DB, prot_list, target, predictor_name, path_save_trained, path_save_tested)

    # ---- Step 3: Run cross-prediction median model ----
    proteins_to_be_tested = list(DB.protein.unique())
    stats_table = function_cross_preds_median(
        path_save_tested, cross_predictions, predictor_name, target, proteins_to_be_tested, howmany
    )

    # ---- Step 4: Count protein occurrences ----
    input_array = stats_table['median_of_which_proteins'].values
    protein_counts = count_protein_occurrences(input_array)

    return stats_table, protein_counts



##########################################combine part##############################
def run_pspall(DB, target, features, position_features=None,
               predictor_name=None,
               undersample=False,
               zero_low_pLDDT=False):

    base_models = ["psp1_mlr", "psp2_mlr", "psp2_xgb", "psp2_rfr"]
    split_models = ["pspsplit1", "pspsplit2", "pspsplit2_obs", "pspsplit2_fusion"]

    # psp1 baseline features (14)
    psp1_features = [
        "Blosum62", "PSSM", "Shannon's entropy", "Shannon's entropy of seq. neighbours",
        "pLDDT", "pLDDT bin", "colasi", "fraction cons. 3D neighbor",
        "fanc", "fbnc", "M.J. potential", "access.dependent vol.", "neco", "laar"
    ]

    if predictor_name == "psp1_mlr":
        DB_Pred = pspbase(
            DB, target, psp1_features,
            predictor_name=predictor_name,
            method="mlr",
            undersample=True,        # special setting for psp1
            zero_low_pLDDT=True      # special setting for psp1
        )

    elif predictor_name in base_models:  # psp2_mlr / psp2_xgb / psp2_rfr
        DB_Pred = pspbase(
            DB, target, features,
            predictor_name=predictor_name,
            method=predictor_name.split("_")[1],  # auto-detect mlr/xgb/rfr
            undersample=False,
            zero_low_pLDDT=False
        )

    elif predictor_name in split_models:  # split models
        DB_Pred = pspsplit(
            DB, target, features, position_features,
            predictor_name=predictor_name,
            undersample=False,
            zero_low_pLDDT=False
        )

    else:
        raise ValueError(
            f"Unknown predictor_name: {predictor_name}\n"
            f"Choose from: {', '.join(base_models + split_models)}"
        )

    return DB_Pred

###########################################################################

def run_cross_pred(DB, path_base, predictor_name,
                   features, target, position_features=None,
                   undersample=False, zero_low_pLDDT=False):

    # Predictor options
    valid_predictors = [
        "qafi1_mlr", "qafi2_mlr", "qafi2_xgb", "qafi2_rfr",
        "qafisplit1", "qafisplit2"
    ]

    if predictor_name not in valid_predictors:
        raise ValueError(
            f"Invalid predictor_name: {predictor_name}\n"
            f"Choose from: {', '.join(valid_predictors)}"
        )

    # Define qafi1 baseline features (14)
    qafi1_features = [
        'Blosum62', 'PSSM', "Shannon's entropy", "Shannon's entropy of seq. neighbours",
        'pLDDT', 'pLDDT bin', 'colasi', 'fraction cons. 3D neighbor',
        'fanc', 'fbnc', 'M.J. potential', 'access.dependent vol.', 'neco', 'laar'
    ]

    # Base predictors
    if predictor_name == "qafi1_mlr":
        method = "mlr"
        cross_pred = qafibase_Cross_Predictions(
            DB=DB,
            path_base=path_base,
            predictor_name=predictor_name,
            features=qafi1_features,   # force qafi1 features
            target=target,
            undersample=True,          # qafi1 only
            method=method,
            zero_low_pLDDT=True        # qafi1 only
        )

    elif predictor_name in ["qafi2_mlr", "qafi2_xgb", "qafi2_rfr"]:
        method = predictor_name.split("_")[1]
        cross_pred = qafibase_Cross_Predictions(
            DB=DB,
            path_base=path_base,
            predictor_name=predictor_name,
            features=features,         # extended features
            target=target,
            undersample=False,         # default
            method=method,
            zero_low_pLDDT=False       # default
        )

    elif predictor_name == "qafisplit1":
        cross_pred = qafisplit1_Cross_Predictions(
            DB=DB,
            path_base=path_base,
            predictor_name=predictor_name,
            features=features,
            position_features=position_features,
            target=target
        )

    elif predictor_name == "qafisplit2":
        cross_pred = qafisplit2_Cross_Predictions(
            DB=DB,
            path_base=path_base,
            predictor_name=predictor_name,
            features=features,
            position_features=position_features,
            target=target
        )

    return cross_pred



######################################QAFI-PRED##################
import os
import warnings
import pandas as pd
import numpy as np
from os.path import join
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression

def qafi1_train_and_predict(
    model_name,
    DB,
    proteins_train,
    df_test_base,
    features,
    structural_features,
    target_column,
    save_each_pred_dir
):
    print(f"📦 Training and predicting with {len(proteins_train)} models for {model_name}...")
    model_columns = []

    for prot_train in proteins_train:
        print(f'🔁 Training on {prot_train} ...')

        # Extract training data for the current protein
        db_train = DB[DB['protein'] == prot_train].copy()

        # Set structural feature values to 0 if pLDDT bin is 0
        db_train.loc[db_train['pLDDT bin'] == 0, structural_features] = 0.0

        # Prepare features and target
        train_x = db_train[features].values
        train_y = db_train[target_column].values

        # Normalize features using MinMaxScaler
        scaler = MinMaxScaler()
        train_x_scaled = scaler.fit_transform(train_x)
        test_x_scaled = scaler.transform(df_test_base[features].values)

        # Train a Linear Regression model
        model = LinearRegression()
        model.fit(train_x_scaled, train_y)

        # Predict on test set
        preds = np.round(model.predict(test_x_scaled), 3)

        # Save predictions with appropriate column name
        col_name = f'{model_name}_{prot_train}'
        model_columns.append(col_name)
        df_pred = df_test_base[['uniprot', 'variant', 'first', 'pos', 'second']].copy()
        df_pred[col_name] = preds

        # Save predictions to file
        pred_file_path = join(save_each_pred_dir, f'train_{prot_train}_predict_combined_df1.csv')
        df_pred.to_csv(pred_file_path, index=False)
        print(f'✅ Saved: {pred_file_path}')

    return model_columns


import pandas as pd
from os.path import join

def merge_predictions_and_compute_median(
    model_name,
    proteins_train,
    df_test_base,
    save_each_pred_dir,
    output_base
):
    """
    Merge predictions from all training proteins and compute the median prediction.
    
    Parameters:
        model_name (str): Model name used as prefix in prediction columns.
        proteins_train (list): List of training protein names.
        df_test_base (pd.DataFrame): Base test dataframe.
        save_each_pred_dir (str): Directory containing individual prediction CSVs.
        output_base (str): Output directory to save the final merged result.
    
    Returns:
        pd.DataFrame: Merged dataframe with median prediction column added.
    """
    print("🔗 Merging predictions and computing median...")

    # Initialize the merged dataframe using the common identifier columns
    df_merged = df_test_base[['uniprot', 'variant', 'first', 'pos', 'second']].copy()

    # Iteratively read and merge each model's prediction
    for prot_train in proteins_train:
        col_name = f'{model_name}_{prot_train}'
        pred_path = join(save_each_pred_dir, f'train_{prot_train}_predict_combined_df1.csv')
        df_model = pd.read_csv(pred_path)

        # Keep only necessary columns for merging
        df_model = df_model[['uniprot', 'variant', col_name]]

        # Left join with the merged dataframe
        df_merged = df_merged.merge(df_model, on=['uniprot', 'variant'], how='left')

    # Compute median across model predictions
    model_columns = [f'{model_name}_{prot}' for prot in proteins_train]
    df_merged[f'{model_name}'] = df_merged[model_columns].median(axis=1)


    return df_merged

########################################################################
from xgboost import XGBRegressor
from sklearn.preprocessing import MinMaxScaler
from os.path import join
import pandas as pd
import numpy as np

def qafi2_train_and_predict_xgb(
    model_name,
    DB,
    proteins_train,
    df_test_base,
    features,
    target_column,
    save_each_pred_dir
):
    """
    Train an XGBoost model for each protein and predict on the test set.
    Save each model's prediction as a separate CSV file.

    Parameters:
        model_name (str): Name used for column naming and output file paths.
        DB (pd.DataFrame): Training dataset containing all proteins.
        proteins_train (list): List of protein identifiers used for training.
        df_test_base (pd.DataFrame): The test dataframe to predict on.
        features (list): Feature column names used for training and testing.
        target_column (str): Name of the target variable to predict.
        save_each_pred_dir (str): Directory to save individual prediction results.

    Returns:
        list: List of column names of the individual model predictions.
    """
    print(f"📦 Training and predicting with {len(proteins_train)} models for {model_name}...")
    model_columns = []

    for prot_train in proteins_train:
        print(f'🔁 Training on {prot_train} ...')

        # Select training data for this protein
        db_train = DB[DB['protein'] == prot_train].copy()
        train_x = db_train[features].values
        train_y = db_train[target_column].values

        # Scale training and test features
        scaler = MinMaxScaler()
        train_x_scaled = scaler.fit_transform(train_x)
        test_x_scaled = scaler.transform(df_test_base[features].values)

        # Initialize and train XGBoost model
        model = XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            min_child_weight=3,
            subsample=0.8,
            colsample_bytree=0.8,
            gamma=0,
            reg_alpha=0.1,
            reg_lambda=1.0,
            objective='reg:squarederror',
            random_state=12234,
            verbosity=0,
            tree_method='hist'
        )
        model.fit(train_x_scaled, train_y)

        # Predict on the test set
        preds = model.predict(test_x_scaled)
        col_name = f'{model_name}_{prot_train}'
        model_columns.append(col_name)

        # Save predictions
        df_pred = df_test_base[['uniprot', 'variant', 'first', 'pos', 'second']].copy()
        df_pred[col_name] = preds
        pred_file_path = join(save_each_pred_dir, f'train_{prot_train}_predict_combined_df1.csv')
        df_pred.to_csv(pred_file_path, index=False)
        print(f'✅ Saved: {pred_file_path}')

    return model_columns
###########################################################################################
def qafisplit1_train_and_predict(
    model_name,
    DB,
    proteins_train,
    df_test_base,
    position_features,
    features,
    target_column,
    save_each_pred_dir
):
    """
    Perform two-step prediction for each training protein:
    Step 1: Position-level median prediction using Linear Regression
    Step 2: Residual prediction using XGBoost
    Saves combined prediction and median to file for each protein.
    """
    from sklearn.linear_model import LinearRegression
    from xgboost import XGBRegressor
    from os.path import join

    print(f"📦 Running split model prediction for {len(proteins_train)} proteins...")

    for prot_train in proteins_train:
        print(f'🔁 Processing {prot_train}...')

        # === Step 1: Position-level Linear Regression ===
        db_train = DB[DB['protein'] == prot_train].copy()
        db_train['protein_pos'] = db_train['protein'] + '_' + db_train['pos'].astype(str)

        train_pos_level = db_train.groupby('protein_pos', as_index=False).agg(
            {**{f: 'first' for f in position_features}, 'pos': 'first', target_column: 'median'}
        )

        model_pos = LinearRegression().fit(
            train_pos_level[position_features],
            train_pos_level[target_column]
        )
        train_pos_level['median_pos_predicted'] = model_pos.predict(train_pos_level[position_features])
        db_train = db_train.merge(train_pos_level[['protein_pos', 'median_pos_predicted']], on='protein_pos', how='left')

        # Apply to test
        test_pos_level = df_test_base.groupby('protein_pos', as_index=False).first()
        test_pos_level['median_pos_predicted'] = model_pos.predict(test_pos_level[position_features])
        df_test = df_test_base.merge(test_pos_level[['protein_pos', 'median_pos_predicted']], on='protein_pos', how='left')

        df_median = df_test[['uniprot', 'pos', 'variant', 'median_pos_predicted']].copy()

        # === Step 2: XGBoost on residual ===
        features_plus = features + ['median_pos_predicted']
        X_train = db_train[features_plus].values
        y_train = db_train[target_column].values
        X_test = df_test[features_plus].values

        model_xgb = XGBRegressor(
            n_estimators=300, learning_rate=0.03, max_depth=3,
            min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
            gamma=0, reg_alpha=0.5, reg_lambda=1.0,
            objective='reg:squarederror', random_state=12234,
            verbosity=0, tree_method='hist'
        )
        model_xgb.fit(X_train, y_train)

        df_pred = df_test[['uniprot', 'variant', 'first', 'pos', 'second']].copy()
        df_pred[f'{model_name}_{prot_train}'] = model_xgb.predict(X_test)

        # Merge median + final prediction
        df_combined = df_pred.merge(df_median, on=['uniprot', 'pos', 'variant'], how='left')

        # Save combined output
        output_path = join(save_each_pred_dir, f'train_{prot_train}_predict_combined_df1_full.csv')
        df_combined.to_csv(output_path, index=False)
        print(f'✅ Saved: {output_path}')


def merge_qafisplit1_predictions(model_name, save_each_pred_dir, output_base):
    """
    Merge all prediction CSVs for the split model, compute per-variant medians
    of final and median predictions, and save final merged result.
    """
    from glob import glob
    import pandas as pd
    from os.path import join

    print("📊 Merging all prediction files...")

    all_csv_paths = glob(join(save_each_pred_dir, 'train_*_predict_combined_df1_full.csv'))
    all_dfs = []

    for path in all_csv_paths:
        df = pd.read_csv(path)
        model_col = [col for col in df.columns if col.startswith(model_name)][0]

        df_sub = df[['uniprot', 'variant', 'pos', 'first', 'second', model_col, 'median_pos_predicted']].copy()
        df_sub = df_sub.rename(columns={
            model_col: 'final_prediction',
            'median_pos_predicted': 'median_prediction'
        })
        all_dfs.append(df_sub)

    df_all = pd.concat(all_dfs, axis=0)

    df_median = df_all.groupby(['uniprot', 'variant', 'pos', 'first', 'second'], as_index=False).agg({
        'final_prediction': 'median',
        'median_prediction': 'median'
    }).rename(columns={
        'final_prediction': f'{model_name}',
        'median_prediction': f'{model_name}_median'
    })

    final_output_file = join(output_base, f'{model_name}.csv')
    df_median.to_csv(final_output_file, index=False)
    print(f'✅ Final merged file saved: {final_output_file}')
#########################################################################################
def qafisplit2_train_and_predict(
    model_name,
    DB,
    proteins_train,
    df_test_base,
    position_features,
    features,
    target_column,
    save_each_pred_dir
):
    """
    For each training protein:
    Step 1: Predict position-level medians using Linear Regression
    Step 2: Predict residuals using XGBoost
    Final prediction = median + residual
    Save prediction results to individual files.
    """
    from sklearn.linear_model import LinearRegression
    from xgboost import XGBRegressor
    from os.path import join

    print(f"📦 Running two-step residual prediction for {len(proteins_train)} proteins...")

    for prot_train in proteins_train:
        print(f'🔁 Processing {prot_train}...')

        # === Step 1: Linear Regression for position median prediction ===
        db_train = DB[DB['protein'] == prot_train].copy()
        db_train['protein_pos'] = db_train['protein'] + '_' + db_train['pos'].astype(str)

        train_pos_level = db_train.groupby('protein_pos', as_index=False).agg(
            {**{f: 'first' for f in position_features}, 'pos': 'first', target_column: 'median'}
        )
        model_pos = LinearRegression().fit(
            train_pos_level[position_features],
            train_pos_level[target_column]
        )
        train_pos_level['median_pos_predicted'] = model_pos.predict(train_pos_level[position_features])
        db_train = db_train.merge(train_pos_level[['protein_pos', 'median_pos_predicted']], on='protein_pos', how='left')

        test_pos_level = df_test_base.groupby('protein_pos', as_index=False).first()
        test_pos_level['median_pos_predicted'] = model_pos.predict(test_pos_level[position_features])
        df_test = df_test_base.merge(test_pos_level[['protein_pos', 'median_pos_predicted']], on='protein_pos', how='left')

        # === Step 2: XGBoost on residuals ===
        db_train['residual'] = db_train[target_column] - db_train['median_pos_predicted']

        features_plus = features + ['median_pos_predicted']
        X_train = db_train[features_plus].values
        y_train = db_train['residual'].values
        X_test = df_test[features_plus].values

        model_residual = XGBRegressor(
            n_estimators=300, learning_rate=0.03, max_depth=3,
            min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
            gamma=0, reg_alpha=0.5, reg_lambda=1.0,
            objective='reg:squarederror', random_state=12234,
            verbosity=0, tree_method='hist'
        )
        model_residual.fit(X_train, y_train)

        residual_pred = model_residual.predict(X_test)
        final_pred = df_test['median_pos_predicted'].values + residual_pred

        # Save combined prediction results
        df_pred = df_test[['uniprot', 'variant', 'first', 'pos', 'second']].copy()
        df_pred[f'{model_name}_{prot_train}'] = final_pred
        df_pred['residual_predicted'] = residual_pred
        df_pred['median_pos_predicted'] = df_test['median_pos_predicted']

        output_path = join(save_each_pred_dir, f'train_{prot_train}_predict_combined_df1_full.csv')
        df_pred.to_csv(output_path, index=False)
        print(f'✅ Saved: {output_path}')


def merge_qafisplit2_predictions(model_name, save_each_pred_dir, output_base):
    """
    Merge individual prediction files from step 1+2 (median + residual).
    Compute median across all training proteins for each mutation.
    Output includes:
        - Final prediction (median + residual)
        - Median prediction only
        - Residual prediction only
    """
    from glob import glob
    import pandas as pd
    from os.path import join

    print("📊 Merging all prediction files...")

    all_csv_paths = glob(join(save_each_pred_dir, 'train_*_predict_combined_df1_full.csv'))
    all_dfs = []

    for path in all_csv_paths:
        df = pd.read_csv(path)
        model_col = [col for col in df.columns if col.startswith(model_name)][0]

        df_sub = df[['uniprot', 'variant', 'pos', 'first', 'second',
                     model_col, 'median_pos_predicted', 'residual_predicted']].copy()

        df_sub = df_sub.rename(columns={
            model_col: 'final_prediction',
            'median_pos_predicted': 'median_prediction',
            'residual_predicted': 'residual_prediction'
        })

        all_dfs.append(df_sub)

    df_all = pd.concat(all_dfs, axis=0)

    df_median = df_all.groupby(['uniprot', 'variant', 'pos', 'first', 'second'], as_index=False).agg({
        'final_prediction': 'median',
        'median_prediction': 'median',
        'residual_prediction': 'median'
    }).rename(columns={
        'final_prediction': f'{model_name}',
        'median_prediction': f'{model_name}_median',
        'residual_prediction': f'{model_name}_residual'
    })

    final_output_file = join(output_base, f'{model_name}.csv')
    df_median.to_csv(final_output_file, index=False)
    print(f'✅ Final merged file saved: {final_output_file}')
###################################################################################

import os
import pandas as pd
from os.path import join

def run_qafi_model(model_name, uniprot_id, DB=None, df_build_proteins=None,
                   df_test_base=None, weight_name=None, path_base=None):
    """
    Unified runner for QAFI models (qafi1, qafi2, qafisplit1, qafisplit2, qafisplit3).

    Parameters
    ----------
    model_name : str
        One of ['qafi1', 'qafi2', 'qafisplit1', 'qafisplit2', 'qafisplit3'].
    uniprot_id : str
        UniProt ID of the test protein.
    DB : pd.DataFrame, optional
        Main training dataset (required for qafi1, qafi2, qafisplit1, qafisplit2).
    df_build_proteins : pd.DataFrame, optional
        Table of training proteins for each model (columns = model_name).
    df_test_base : pd.DataFrame, optional
        Base test DataFrame (required for qafi1, qafi2, qafisplit1, qafisplit2).
    weight_name : str, optional
        Name of weight folder (required for qafisplit3).
    path_base : str, default="output/QAFI_predictions"
        Base output directory for saving predictions.

    Returns
    -------
    final_save_path : str or None
        Path to the final saved CSV file (None if not applicable).
    """

    # === Setup paths ===
    output_base = join(path_base, uniprot_id, model_name)
    save_each_pred_dir = join(output_base, 'predictions')
    os.makedirs(save_each_pred_dir, exist_ok=True)

    # === Training protein selection ===
    if model_name in ['qafi1', 'qafi2', 'qafisplit1', 'qafisplit2']:
        if DB is None or df_build_proteins is None:
            raise ValueError("❌ DB and df_build_proteins are required for this model type")

        if model_name not in df_build_proteins.columns:
            raise ValueError(f"❌ model_name {model_name} not found in df_build_proteins")
        proteins_train = df_build_proteins[model_name].dropna().tolist()
        target_column = 'score_log_normalized'

    # === Define shared features (for qafi2, qafisplit1, qafisplit2) ===
    shared_features = [
        'Blosum62', 'PSSM', "Shannon's entropy", "Shannon's entropy of seq. neighbours",
        'pLDDT', 'pLDDT bin', 'colasi', 'fraction cons. 3D neighbor', 'fanc', 'fbnc',
        'M.J. potential', 'access.dependent vol.', 'neco',
        'PSSM_pdff', "Shannon's entropy_pdff", "Shannon's entropy of seq. neighbours_pdff", 
        'pLDDT_pdff', 'colasi_pdff', 'fraction cons. 3D neighbor_pdff', 'fanc_pdff', 
        'fbnc_pdff', 'e_native_pdff', 
        'neco2', 'neco3', 'neco12',
        'lpar', 'NCE-NR','laar'
    ]

    shared_position_features = [
        'PSSM', "Shannon's entropy", "Shannon's entropy of seq. neighbours",
        'pLDDT', 'pLDDT bin', 'colasi', 'fraction cons. 3D neighbor', 
        'fanc', 'fbnc', 'NCE-NR'
    ]

    # === QAFI1 ===
    if model_name == 'qafi1':
        structural_features = [
            'colasi', 'fraction cons. 3D neighbor', 'fanc', 'fbnc',
            'M.J. potential', 'access.dependent vol.', 'laar'
        ]
        features = [
            'Blosum62', 'PSSM', "Shannon's entropy", "Shannon's entropy of seq. neighbours", 'neco',
            'pLDDT', 'pLDDT bin'
        ] + structural_features

        qafi1_train_and_predict(
            model_name=model_name,
            DB=DB,
            proteins_train=proteins_train,
            df_test_base=df_test_base,
            features=features,
            structural_features=structural_features,
            target_column=target_column,
            save_each_pred_dir=save_each_pred_dir
        )
        df_merged = merge_predictions_and_compute_median(
            model_name=model_name,
            proteins_train=proteins_train,
            df_test_base=df_test_base,
            save_each_pred_dir=save_each_pred_dir,
            output_base=output_base
        )

    # === QAFI2 ===
    elif model_name == 'qafi2':
        qafi2_train_and_predict_xgb(
            model_name=model_name,
            DB=DB,
            proteins_train=proteins_train,
            df_test_base=df_test_base,
            features=shared_features,
            target_column=target_column,
            save_each_pred_dir=save_each_pred_dir
        )
        df_merged = merge_predictions_and_compute_median(
            model_name=model_name,
            proteins_train=proteins_train,
            df_test_base=df_test_base,
            save_each_pred_dir=save_each_pred_dir,
            output_base=output_base
        )

    # === QAFISPLIT1 ===
    elif model_name == 'qafisplit1':
        qafisplit1_train_and_predict(
            model_name=model_name,
            DB=DB,
            proteins_train=proteins_train,
            df_test_base=df_test_base,
            position_features=shared_position_features,
            features=shared_features,
            target_column=target_column,
            save_each_pred_dir=save_each_pred_dir
        )
        df_merged = merge_qafisplit1_predictions(
            model_name=model_name,
            save_each_pred_dir=save_each_pred_dir,
            output_base=output_base
        )

    # === QAFISPLIT2 ===
    elif model_name == 'qafisplit2':
        qafisplit2_train_and_predict(
            model_name=model_name,
            DB=DB,
            proteins_train=proteins_train,
            df_test_base=df_test_base,
            position_features=shared_position_features,
            features=shared_features,
            target_column=target_column,
            save_each_pred_dir=save_each_pred_dir
        )
        df_merged = merge_qafisplit2_predictions(
            model_name=model_name,
            save_each_pred_dir=save_each_pred_dir,
            output_base=output_base
        )
    # === QAFISPLIT3 ===
    elif model_name == 'qafisplit3':
        proteins_train = DB['protein'].unique().tolist()  

        qafisplit2_train_and_predict(  
            model_name=model_name,
            DB=DB,
            proteins_train=proteins_train,
            df_test_base=df_test_base,
            position_features=shared_position_features,
            features=shared_features,
            target_column="score_log_normalized",
            save_each_pred_dir=save_each_pred_dir
        )
        df_merged = merge_qafisplit3_predictions( 
            model_name=model_name,
            save_each_pred_dir=save_each_pred_dir,
            output_base=output_base,
            sim_metric="pearson"
        )

    # === Save final result if DataFrame is returned ===
    if isinstance(df_merged, pd.DataFrame):
        final_save_path = join(output_base, f'{model_name}.csv')
        if model_name in ['qafi1', 'qafi2']:
            df_merged[['uniprot', 'variant', 'first', 'pos', 'second', f'{model_name}']].to_csv(final_save_path, index=False)
        else:
            df_merged.to_csv(final_save_path, index=False)
        print(f'✅ Final merged median prediction saved to: {final_save_path}')
        return final_save_path
    else:
        print(f'⚠️ No DataFrame returned for {model_name}, merge function likely handled saving internally.')
        return None
#########################################
def merge_qafisplit3_predictions(model_name, save_each_pred_dir, output_base, sim_metric="pearson"):
    """
    Merge all prediction CSVs for qafisplit3.
    Instead of simple median, use similarity-based weighted fusion:
    - Compute similarity between each training protein's predictions
    - Proteins with higher average similarity get higher weights
    - Final prediction = weighted sum of all training proteins
    """
    from glob import glob
    import pandas as pd
    import numpy as np
    from os.path import join
    from scipy.spatial.distance import pdist, squareform
    from scipy.stats import pearsonr

    print("📊 Merging all prediction files with similarity-based fusion...")

    # === Step 1: Load all predictions ===
    all_csv_paths = glob(join(save_each_pred_dir, 'train_*_predict_combined_df1_full.csv'))
    prot_preds = {}
    base_df = None

    for path in all_csv_paths:
        df = pd.read_csv(path)
        model_col = [col for col in df.columns if col.startswith(model_name)][0]
        prot_name = path.split('train_')[1].split('_predict')[0]

        if base_df is None:
            base_df = df[['uniprot','variant','pos','first','second']].copy()

        prot_preds[prot_name] = df[model_col].values

    pred_matrix = pd.DataFrame(prot_preds)  # rows=variants, cols=proteins

    # === Step 2: Compute similarity between proteins ===
    if sim_metric == "pearson":
        corr = pred_matrix.corr().fillna(0).values  # NxN correlation
        sim_scores = corr
    elif sim_metric == "cosine":
        # cosine similarity = 1 - cosine distance
        from sklearn.metrics.pairwise import cosine_similarity
        sim_scores = cosine_similarity(pred_matrix.T)
    else:
        raise ValueError("Unknown sim_metric, choose 'pearson' or 'cosine'")

    # === Step 3: Compute weights ===
    avg_sim = sim_scores.mean(axis=1)  # average similarity per protein
    weights = avg_sim / avg_sim.sum()  # normalize
    weight_dict = dict(zip(pred_matrix.columns, weights))
    print("🔑 Fusion weights:", weight_dict)

    # === Step 4: Weighted fusion ===
    final_pred = np.zeros(len(base_df))
    for i, prot in enumerate(pred_matrix.columns):
        final_pred += pred_matrix[prot].values * weights[i]

    # === Step 5: Save output ===
    df_out = base_df.copy()
    df_out[f'{model_name}'] = final_pred

    final_output_file = join(output_base, f'{model_name}.csv')
    df_out.to_csv(final_output_file, index=False)
    print(f'✅ Final weighted-fusion file saved: {final_output_file}')

