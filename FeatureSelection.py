#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 17:15:03 2024

@author: pavlomysak
"""
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.feature_selection import SelectKBest, RFE, VarianceThreshold, SelectFromModel, mutual_info_regression
from sklearn.model_selection import cross_val_predict, KFold
import statsmodels.api as sm
import warnings

#import caffeine
#caffeine.on(display=False)


################################
############################
## DATA GENERATION FUNCTION
############################

def get_data(distribution="normal", rows=100, columns=10, n_info_cols=0.5, noise_ratio=0.2, 
             center_around_zero=True, uniform_variance=True, random_seed=None, mixed_distributions=None,
             n_polynomial_terms=0, n_interaction_terms=0):
    """
    Generate synthetic datasets with controlled properties, including mixed distributions, varied centers, variances,
    polynomial terms, and interaction terms.

    Parameters:
        distribution (str): The base distribution to use ("normal", "uniform", "exponential", or "mixed").
            If "mixed", specify `mixed_distributions` as a dictionary, e.g., {"normal": 0.3, "exponential": 0.7}.
        rows (int): Number of rows (samples).
        columns (int): Total number of columns (features).
        n_info_cols (float): Ratio of informative columns (0 to 1).
        noise_ratio (float): Proportion of noise to inject into the dataset.
        center_around_zero (bool): If True, all features are centered around 0. Otherwise, centers vary randomly.
        uniform_variance (bool): If True, all features have the same variance. Otherwise, variances vary randomly.
        random_seed (int): Seed for reproducibility.
        mixed_distributions (dict): Dictionary specifying the ratio of features for each distribution if 
            `distribution="mixed"`, e.g., {"normal": 0.3, "exponential": 0.7}.
        n_polynomial_terms (int): Number of polynomial terms (e.g., squared or cubic) to include.
        n_interaction_terms (int): Number of interaction terms to add (random pairs of features).

    Returns:
        data (pd.DataFrame): The generated dataset.
        metadata (dict): Metadata about the dataset.
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    # Define parameters for random centers and variances
    if not center_around_zero:
        centers = np.random.uniform(-100, 100, columns)
    else:
        centers = np.zeros(columns)

    if not uniform_variance:
        variances = np.random.uniform(1, 50, columns)
    else:
        variances = np.ones(columns)

    # Feature generation map
    feature_map = {
        "normal": lambda size, mean, var: np.random.normal(loc=mean, scale=np.sqrt(var), size=size),
        "uniform": lambda size, mean, var: np.random.uniform(low=mean - var, high=mean + var, size=size),
        "exponential": lambda size, mean, var: np.random.exponential(scale=max(mean + var, 1e-5), size=size),
    }

    # Mixed distribution handling
    if distribution == "mixed":
        if mixed_distributions is None:
            raise ValueError("When `distribution='mixed'`, you must provide `mixed_distributions` as a dictionary.")
        total_columns = columns
        # Scale mixed distribution ratios to match the number of columns
        mixed_distributions_scaled = {k: int(v * total_columns) for k, v in mixed_distributions.items()}

        # Adjust for rounding errors
        total_assigned = sum(mixed_distributions_scaled.values())
        difference = total_columns - total_assigned

        if difference != 0:
            largest_dist = max(mixed_distributions_scaled, key=mixed_distributions_scaled.get)
            mixed_distributions_scaled[largest_dist] += difference

        # Generate distribution labels based on the scaled values
        distribution_labels = []
        for dist, count in mixed_distributions_scaled.items():
            distribution_labels.extend([dist] * count)
        np.random.shuffle(distribution_labels)
    else:
        distribution_labels = [distribution] * columns

    # Generate features
    features = []
    for i, dist in enumerate(distribution_labels):
        if dist not in feature_map:
            raise ValueError(f"Unsupported distribution: {dist}")
        features.append(feature_map[dist]((rows,), centers[i], variances[i]))
    X = np.column_stack(features)

    # Create informative features and labels
    n_info_cols = int(n_info_cols * columns)
    
    if n_info_cols > columns:
        n_info_cols = columns
        
    info_indices = np.random.choice(columns, n_info_cols, replace=False)
    y = np.dot(X[:, info_indices], np.random.uniform(-2.5, 2.5, size=n_info_cols)) + np.random.normal(0, 0.1, size=rows)

    # Polynomial terms
    if n_polynomial_terms > 0:
        poly_indices = np.random.choice(columns, n_polynomial_terms, replace=False)
        for idx in poly_indices:
            degree = np.random.choice([2, 3])  # Randomly choose squared or cubic
            X = np.column_stack((X, X[:, idx] ** degree))

    # Interaction terms
    if n_interaction_terms > 0:
        interaction_indices = np.random.choice(columns, (n_interaction_terms, 2), replace=False)
        for idx_pair in interaction_indices:
            X = np.column_stack((X, X[:, idx_pair[0]] * X[:, idx_pair[1]]))

    # Noise
    n_noise_cols = int(noise_ratio * columns)
    noise = np.random.normal(0, 1, size=(rows, n_noise_cols))
    X[:, :n_noise_cols] = noise

    # Prepare metadata
    metadata = {
        "distribution": distribution,
        "rows": rows,
        "columns": X.shape[1],  # Original number of columns
        "n_info_cols": n_info_cols,
        "info_cols_indices": info_indices.tolist(),  # Indices of informative columns from the original features
        "noise_ratio": noise_ratio,
        "center_around_zero": center_around_zero,
        "uniform_variance": uniform_variance,
        #"centers": centers.tolist(),
        "mean": np.round(np.mean(X.mean(axis=0).tolist()),2),
        "variance": np.round(np.mean(X.var(axis=0).tolist()),2),
        "distributions": distribution_labels,
        "n_polynomial_terms": n_polynomial_terms,
        "n_interaction_terms": n_interaction_terms,
        "polynomial_terms_indices": list(poly_indices) if n_polynomial_terms > 0 else [],
        "interaction_terms_indices": interaction_indices.tolist() if n_interaction_terms > 0 else [],
    }

    # Return Data as DataFrame
    data = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
    data["target"] = y

    return data, metadata


def data_analysis(cv, model, feature_selection_algos, data, metadata):
    """
    Run analysis for feature selection algorithms and evaluate RMSE, Feature Precision, and Feature Recall.

    Parameters:
    - cv: sklearn CV object, Cross-validation strategy
    - model: sklearn model, The base model to use for predictions
    - feature_selection_algos: list of tuples, Feature selection algorithms (name, selector)
    - data: pd.DataFrame, Dataset containing features and target
    - metadata: dict, Metadata containing info on feature indices for "info_cols"

    Returns:
    - dict, Final results with RMSE, Precision, and Recall
    """

    # Extract features and target
    X, y = data.drop('target', axis=1), data['target']
    info_cols = metadata["info_cols_indices"]

    # Initialize results dictionary
    results = {
        "Baseline": {
            "Lasso_RMSE": None,
            "Ridge_RMSE": None,
        },
        "Feature_Selection": {}
    }

    # Baseline Models - Lasso and Ridge
    print("Running Baseline Models")
    lasso_pred = cross_val_predict(Lasso(), X, y, cv=cv)
    results["Baseline"]["Lasso_RMSE"] = np.round(np.sqrt(np.mean((y - lasso_pred) ** 2)),2)

    ridge_pred = cross_val_predict(Ridge(), X, y, cv=cv)
    results["Baseline"]["Ridge_RMSE"] = np.round(np.sqrt(np.mean((y - ridge_pred) ** 2)))

    # Feature Selection Analysis
    for name, selector in feature_selection_algos:
        print(f'Running for {name}')
        
        # Suppress warning if no features are selected
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            selector.fit(X, y)
        
        X_transformed = selector.transform(X)

        # Identify selected features
        if hasattr(selector, 'get_support'):
            selected_features = X.columns[selector.get_support()]
        elif hasattr(selector, 'get_feature_names_out'):
            selected_features = selector.get_feature_names_out(input_features=X.columns)
        else:
            selected_features = []

        # If no features are selected, skip to the next algorithm
        if len(selected_features) == 0:
            print(f"No features selected by {name}. Skipping this feature selection method.")
            results["Feature_Selection"][name] = {
                "RMSE": None,
                "Precision": None,
                "Recall": None,
                "FeatureCorrectness": None
            }
            continue

        # Evaluate RMSE
        y_pred = cross_val_predict(model, X_transformed, y, cv=cv)
        rmse = np.sqrt(np.mean((y - y_pred) ** 2))

        # Feature Precision and Recall
        common_features = [
            feature for feature in selected_features if feature in X.columns[info_cols]
        ]
        precision = len(common_features) / max(1, len(selected_features))  # Avoid division by zero
        recall = len(common_features) / len(info_cols)

        # Store results
        results["Feature_Selection"][name] = {
            "RMSE": np.round(rmse,2),
            "Precision": np.round(precision,2),
            "Recall": np.round(recall,2),
            "FeatureCorrectness": 0 if (precision + recall) == 0 else np.round(2 * (precision * recall) / (precision + recall), 2)
        }

    return results


################################
############################
## PRE-LOOP DEFINITIONS
############################

# Defining ranges for parameter randomization
distributions = ["normal", "uniform", "exponential", "mixed"]
row_range = (50, 100000)  
column_range = (2, 250)  
info_ratio_range = (0.1, 0.9)  
noise_ratio_range = (0.0, 0.5)
center_around_zero_options = [True, False]
uniform_variance_options = [True, False]
random_seed_range = (1, 1000)
polynomial_terms_range = (0, 5)
interaction_terms_range = (0, 5)

# Defining LR Model and Cross Validation

model = LinearRegression()
cv = KFold(n_splits=5, shuffle=True)


#####################################
#################################
## GENERATING AND GATHERING DATA
#################################

datasets_dict = {}
metadata_dict = {}
performance_dict = {}
final_dt = []


for i in range(125):  

    ############################
    ##### Generating Data
    #########################
    #########################


    dist = random.choice(distributions)
    rows = np.round(random.uniform(*row_range),0).astype(int)
    columns = np.round(random.triangular(low=column_range[0],
                                high=column_range[1],
                                mode=random.uniform(5, 45)),0).astype(int)
    n_info_ratio = random.uniform(*info_ratio_range)
    noise_ratio = random.uniform(*noise_ratio_range)
    center_around_zero = random.choice(center_around_zero_options)
    uniform_variance = random.choice(uniform_variance_options)
    random_seed = random.randint(*random_seed_range)
    n_polynomial_terms = random.randint(*polynomial_terms_range)
    n_interaction_terms = random.randint(*interaction_terms_range)
    
    if n_polynomial_terms > columns:
        n_polynomial_terms = columns
    if n_interaction_terms > (columns * (columns - 1) // 2):
        n_interaction_terms = (columns * (columns - 1) // 2)
    
    if dist == "mixed":
        num_mixed_types = random.randint(2, 3)
        mixed_distributions = {d: random.uniform(0.1, 0.5) for d in random.sample(distributions[:-1], num_mixed_types)}
        total_ratio = sum(mixed_distributions.values())
        mixed_distributions = {k: v / total_ratio for k, v in mixed_distributions.items()}
    else:
        mixed_distributions = None

    n_info_cols = int(n_info_ratio * columns)
    data, metadata = get_data(
        distribution=dist,
        rows=rows,
        columns=columns,
        n_info_cols=n_info_cols,
        noise_ratio=noise_ratio,
        center_around_zero=center_around_zero,
        uniform_variance=uniform_variance,
        random_seed=random_seed,
        mixed_distributions=mixed_distributions,
        n_polynomial_terms=n_polynomial_terms,
        n_interaction_terms=n_interaction_terms
    )
    
    datasets_dict.update({i:data})
    metadata_dict.update({i:metadata})
    
    ############################
    ##### Prepping Analysis
    #########################
    #########################
    
    if n_info_cols < columns-5:
    
        feature_selection_algos = [
            ('SelectKBest', SelectKBest(mutual_info_regression,
                                        k = n_info_cols)),
            ('RFE', RFE(estimator = Lasso(),
                        n_features_to_select = n_info_cols,
                        step = 1)),
            ('VarianceThreshold', VarianceThreshold(threshold = metadata["variance"])), 
            ('SelectFromModel_Lasso', SelectFromModel(estimator = Lasso(),
                                                      max_features = n_info_cols)),
            ('SelectFromModel_Tree', SelectFromModel(estimator = DecisionTreeRegressor(),
                                                     max_features = n_info_cols))
                                    ]
    
    else:
        
        feature_selection_algos = [
            ('SelectKBest', SelectKBest(mutual_info_regression,
                                        k = abs(n_info_cols-5))),
            ('RFE', RFE(estimator = Lasso(),
                        n_features_to_select = abs(n_info_cols-5),
                        step = 1)),
            ('VarianceThreshold', VarianceThreshold(threshold = data.var().quantile(0.25))), ##### PROBLEM?
            ('SelectFromModel_Lasso', SelectFromModel(estimator = Lasso(),
                                                      max_features = abs(n_info_cols-5))),
            ('SelectFromModel_Tree', SelectFromModel(estimator = DecisionTreeRegressor(),
                                                     max_features = abs(n_info_cols-5)))
                                    ]
    
    ############################
    ##### Analysis and Performance
    #########################
    #########################
    
    performance = data_analysis(cv = cv, model = model,
                                feature_selection_algos = feature_selection_algos,
                                data = data, metadata = metadata)
    
    performance_dict.update({i:performance})
    
    ############################
    ##### Labelling
    #########################
    #########################
    
    label = None
    highest_score = float('-inf')
    lowest_rmse = float('inf')

    for fs, data in performance["Feature_Selection"].items():
        
        if data["RMSE"] is None or data["FeatureCorrectness"] is None:
            continue

        if data["FeatureCorrectness"] > highest_score:
            highest_score = data["FeatureCorrectness"]
            lowest_rmse = data["RMSE"]
            label = fs
            
        # If there's a tie in FeatureCorrectness, use RMSE to decide
        elif data["FeatureCorrectness"] == highest_score and data["RMSE"] < lowest_rmse:
            lowest_rmse = data["RMSE"]
            label = fs

    final_dt.append({"data_id": i,
                      "label":label,
                      "nrows":metadata["rows"],
                      "ncols": metadata["columns"],
                      "n_info_cols": metadata["n_info_cols"],
                      "noise_ratio": metadata["noise_ratio"],
                      "cetered": metadata["center_around_zero"],
                      "uniform_var": metadata["uniform_variance"],
                      "n_polys": metadata["n_polynomial_terms"],
                      "n_interactions": metadata["n_interaction_terms"],
                      "distribution": metadata["distribution"]})
    
    print(f"DONE WITH {i+1} DATASET(S)")


print(f"Generated and analyzed {len(datasets_dict)} unique datasets.")
mtdt_data = pd.DataFrame(final_dt)



mtdt_data["label"].value_counts()



###################################
###############################
## RUNNING META-LEARNING LAYER
###############################

mll_dt = pd.get_dummies(mtdt_data, prefix= "dist", columns = ["distribution"], drop_first=True).drop(["data_id"], axis=1)
mll_dt["dims_ratio"] = mll_dt["ncols"]/mll_dt["nrows"]
mll_dt["info_col_ratio"] = mll_dt["n_info_cols"]/mll_dt["ncols"]
mll_dt["poly_ratio"] = mll_dt["n_polys"]/mll_dt["ncols"]
mll_dt["interaction_ratio"] = mll_dt["n_interactions"]/mll_dt["ncols"]

mll_dt = mll_dt.drop(["ncols", "nrows", "n_info_cols", "n_polys", "n_interactions"], axis=1)


sm.MNLogit(endog = mll_dt["label"],
           exog = mll_dt.drop(["label"], axis=1).astype(float)
           ).fit().summary()

# Looking at only RFE
sm.Logit(endog= pd.get_dummies(mll_dt["label"])["RFE"],
         exog = mll_dt.drop(["label"], axis=1).astype(float)).fit().summary()







