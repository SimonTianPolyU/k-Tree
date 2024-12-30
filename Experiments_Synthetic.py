import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from math import sqrt
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from PkTree import PkTree
from AkTree import AkTree
from PkRF import PkRF
from sklearn.preprocessing import StandardScaler
import time
from sklearn.datasets import make_regression

def generate_data(n_feautres, n_samples, seed=42):
    np.random.seed(seed)

    X,Y = make_regression(n_samples=n_samples, n_features=n_feautres, n_informative= int(n_feautres*0.8),
                            n_targets=1, noise=0.3, effective_rank=max(int(n_feautres*0.8),1), tail_strength=0.5,random_state=42
                          )

    return X, Y

def generate_and_split_data(P, N, seed=None):

    X, y = generate_data(P, N, seed)

    # Splitting into training (80%) and test (20%)
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2)

    # Splitting the training and validation with a ratio of 3:1
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25)

    return X_train, X_val, X_test, y_train, y_val, y_test, P, N

seed = 42
P_values = [1,3,5,7,9,11,13,15,17]
N_values = [100,300,500,700]
datasets = [generate_and_split_data(P, N, seed) for P in P_values for N in N_values]

def train_kNN_with_validation(X_train, y_train, X_val, y_val, N_train):
    k_range = list(range(max(1, int(sqrt(N_train)) - 4), int(sqrt(N_train)) + 5))
    best_k = None
    best_score = float('inf')

    # Normalizing the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    for k in k_range:
        model = KNeighborsRegressor(n_neighbors=k, metric='manhattan')
        model.fit(X_train_scaled, y_train)
        score = mean_squared_error(y_val, model.predict(X_val_scaled))
        if score < best_score:
            best_score = score
            best_k = k

    return {'n_neighbors': best_k}, best_score

def train_CART_with_validation(X_train, y_train, X_val, y_val):
    min_samples_per_leaf_options = [5, 10, 15, 20]
    max_depth_options = [4, 5, 6, 7, 8, 9, 10]
    best_min_samples = None
    best_max_depth = None

    best_score = float('inf')

    for min_samples in min_samples_per_leaf_options:
        for max_depth in max_depth_options:
            model = DecisionTreeRegressor(min_samples_leaf=min_samples, max_depth=max_depth)
            model.fit(X_train, y_train)
            score = mean_squared_error(y_val, model.predict(X_val))

            if score < best_score:
                best_score = score
                best_min_samples = min_samples
                best_max_depth = max_depth

    return {'min_samples_leaf': best_min_samples, 'max_depth': best_max_depth}, best_score

def train_RF_with_validation(X_train, y_train, X_val, y_val):
    min_samples_per_leaf_options = [5, 10, 15, 20]
    max_depth_options = [4, 5, 6, 7, 8, 9, 10]
    n_estimators_options = [100,200]
    best_min_samples = None
    best_max_depth = None

    best_model = None
    best_score = float('inf')

    for min_samples in min_samples_per_leaf_options:
        for max_depth in max_depth_options:
            for n_estimators in n_estimators_options:
                model = RandomForestRegressor(min_samples_leaf=min_samples, max_depth=max_depth, n_estimators=n_estimators)
                model.fit(X_train, y_train)
                score = mean_squared_error(y_val, model.predict(X_val))
                if score < best_score:
                    best_score = score
                    best_min_samples = min_samples
                    best_max_depth = max_depth
                    best_n_estimators = n_estimators

    return {'min_samples_leaf': best_min_samples, 'max_depth': best_max_depth, 'n_estimators': best_n_estimators}, best_score

def train_PkTree_with_validation(X_train, y_train, X_val, y_val, N_train):
    k_range = [int(sqrt(N_train)) - i for i in range(4, -5, -2)]  # Adjusted k range
    min_samples_per_leaf_options = [5, 10, 15, 20]
    max_depth_options = [4, 5, 6, 7, 8, 9, 10]
    best_k = None
    best_min_samples = None
    best_max_depth = None

    best_score = float('inf')

    start_time = time.time()

    for k in k_range:
        for min_samples in min_samples_per_leaf_options:
            for max_depth in max_depth_options:
                dt_params = {'min_samples_leaf': min_samples, 'max_depth': max_depth}
                model = PkTree(dt_params=dt_params, k=k)
                model.fit(X_train, y_train)
                predictions_val = model.predict(X_val)[0]
                score = mean_squared_error(y_val, predictions_val)
                if score < best_score:
                    best_score = score
                    best_k = k
                    best_min_samples = min_samples
                    best_max_depth = max_depth

                current_time = time.time() - start_time
                if current_time > 1200:
                    return {'min_samples_leaf': best_min_samples, 'max_depth': best_max_depth, 'k': best_k}, 'not enough time'

    return {'min_samples_leaf': best_min_samples, 'max_depth': best_max_depth, 'k': best_k}, best_score

def train_PkRF_with_validation(X_train, y_train, X_val, y_val, N_train):
    k_range = [int(np.sqrt(N_train)) - i for i in range(4, -5, -2)]
    min_samples_per_leaf_options = [5, 10, 15, 20]
    max_depth_options = [4, 5, 6, 7, 8, 9, 10]
    total_features = X_train.shape[1]
    max_features_options = [int(total_features * p) for p in [1.0]]
    max_features_options = list(set([max(1, mf) for mf in max_features_options]))
    n_estimators_options = [100,200]

    best_score = float('inf')
    best_params = None

    start_time = time.time()

    for k in k_range:
        for min_samples in min_samples_per_leaf_options:
            for max_depth in max_depth_options:
                for max_features in max_features_options:
                    for n_estimators in n_estimators_options:
                        dt_params = {'min_samples_leaf': min_samples, 'max_depth': max_depth}
                        model = PkRF(n_estimators=n_estimators, dt_params=dt_params, k=k, max_features=max_features)
                        model.fit(X_train, y_train)
                        predictions_val = model.predict(X_val)[0]
                        score = mean_squared_error(y_val, predictions_val)
                        if score < best_score:
                            best_score = score
                            best_params = {
                                'k': k,
                                'min_samples_leaf': min_samples,
                                'max_depth': max_depth,
                                'max_features': max_features,
                                'n_estimators': n_estimators
                            }

                        current_time = time.time() - start_time
                        if current_time > 1200:
                            return best_params, 'not enough time'

    return best_params, best_score

def train_AkTree_with_validation(X_train, y_train, X_val, y_val, N_train):
    k_range = [int(sqrt(N_train)) - i for i in range(4, -5, -2)]  # Adjusted k range
    min_samples_per_leaf_options = [5, 10, 15, 20]
    max_depth_options = [4, 5, 6, 7, 8, 9, 10]
    best_k = None
    best_min_samples = None
    best_max_depth = None

    best_score = float('inf')

    for k in k_range:
        for min_samples in min_samples_per_leaf_options:
            for max_depth in max_depth_options:
                model = AkTree(min_samples=min_samples, max_depth=max_depth, k=k)
                model.fit(X_train, y_train)
                predictions_val = model.predict_all(X_val)[0]
                score = mean_squared_error(y_val, predictions_val)
                if score < best_score:
                    best_score = score
                    best_k = k
                    best_min_samples = min_samples
                    best_max_depth = max_depth

    return {'min_samples_leaf': best_min_samples, 'max_depth': best_max_depth, 'k': best_k}, best_score

results = []


for dataset in datasets:
    X_train, X_val, X_test, y_train, y_val, y_test, P, N = dataset
    print(P,N)
    X_train_final = np.concatenate((X_train, X_val))
    y_train_final = np.concatenate((y_train, y_val))

    # Training each model with validation set

    kNN_params, kNN_best_score = train_kNN_with_validation(X_train, y_train, X_val, y_val, len(X_train))
    print("kNN done")

    CART_params, CART_best_score = train_CART_with_validation(X_train, y_train, X_val, y_val)
    print("CART done")

    RF_params, RF_best_score = train_RF_with_validation(X_train, y_train, X_val, y_val)
    print("RF done")

    PkTree_params, PkTree_best_score = train_PkTree_with_validation(X_train, y_train, X_val, y_val, len(X_train))
    print("PkTree done")

    PkRF_params, PkRF_best_score = train_PkRF_with_validation(X_train, y_train, X_val, y_val, len(X_train))
    print("PkRF done")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_final)
    X_test_scaled = scaler.transform(X_test)

    start_time = time.time()
    kNN_model = KNeighborsRegressor(n_neighbors=kNN_params['n_neighbors'], metric='manhattan')
    kNN_model.fit(X_train_scaled, y_train_final)
    kNN_training_time = time.time() - start_time

    start_time = time.time()
    CART_model = DecisionTreeRegressor(min_samples_leaf=CART_params['min_samples_leaf'], max_depth=CART_params['max_depth'])
    CART_model.fit(X_train_final, y_train_final)
    CART_training_time = time.time() - start_time

    start_time = time.time()
    RF_model = RandomForestRegressor(min_samples_leaf=RF_params['min_samples_leaf'], max_depth=RF_params['max_depth'], n_estimators=RF_params['n_estimators'])
    RF_model.fit(X_train_final, y_train_final)
    RF_training_time = time.time() - start_time

    start_time = time.time()
    PkTree_model = PkTree(
        dt_params={'min_samples_leaf': PkTree_params['min_samples_leaf'], 'max_depth': PkTree_params['max_depth']},
        k=PkTree_params['k'])
    PkTree_model.fit(X_train_final, y_train_final)
    PkTree_training_time = time.time() - start_time
    # print('down')

    start_time = time.time()
    PkRF_model = PkRF(n_estimators=PkRF_params['n_estimators'],
        dt_params={'min_samples_leaf': PkRF_params['min_samples_leaf'], 'max_depth': PkRF_params['max_depth']},
        k=PkRF_params['k'], max_features=PkRF_params['max_features'])
    PkRF_model.fit(X_train_final, y_train_final)
    PkRF_training_time = time.time() - start_time

    # Testing and calculating MSE
    start_time = time.time()
    kNN_predictions = kNN_model.predict(X_test_scaled)
    kNN_prediction_time = time.time() - start_time
    kNN_mse = mean_squared_error(y_test, kNN_predictions)

    start_time = time.time()
    CART_predictions = CART_model.predict(X_test)
    CART_prediction_time = time.time() - start_time
    CART_mse = mean_squared_error(y_test, CART_predictions)

    start_time = time.time()
    RF_predictions = RF_model.predict(X_test)
    RF_prediction_time = time.time() - start_time
    RF_mse = mean_squared_error(y_test, RF_predictions)

    start_time = time.time()
    PkTree_predictions, PkTree_neighbor_proportions_in_adjacent = PkTree_model.predict(X_test)
    PkTree_prediction_time = time.time() - start_time
    PkTree_mse = mean_squared_error(y_test, PkTree_predictions)

    start_time = time.time()
    PkRF_predictions, PkRF_neighbor_proportions_in_adjacent = PkRF_model.predict(X_test)
    PkRF_prediction_time = time.time() - start_time
    PkRF_mse = mean_squared_error(y_test, PkRF_predictions)

    result = {
        'features': P,
        'size': N,
        'kNN_params': kNN_params,
        'kNN_best_score': kNN_best_score,
        'kNN_mse': kNN_mse,
        'kNN_training_time': kNN_training_time,
        'kNN_prediction_time': kNN_prediction_time,
        'CART_params': CART_params,
        'CART_best_score': CART_best_score,
        'CART_mse': CART_mse,
        'CART_training_time': CART_training_time,
        'CART_prediction_time': CART_prediction_time,
        'RF_params': RF_params,
        'RF_best_score': RF_best_score,
        'RF_mse': RF_mse,
        'RF_training_time': RF_training_time,
        'RF_prediction_time': RF_prediction_time,
        'PkTree_params': PkTree_params,
        'PkTree_best_score': PkTree_best_score,
        'PkTree_mse': PkTree_mse,
        'PkTree_training_time': PkTree_training_time,
        'PkTree_prediction_time': PkTree_prediction_time,
        'PkTree_neighbor_proportions_in_adjacent': PkTree_neighbor_proportions_in_adjacent,
        'PkRF_params': PkRF_params,
        'PkRF_best_score': PkRF_best_score,
        'PkRF_mse': PkRF_mse,
        'PkRF_training_time': PkRF_training_time,
        'PkRF_prediction_time': PkRF_prediction_time,
        'PkRF_neighbor_proportions_in_adjacent': PkRF_neighbor_proportions_in_adjacent
    }
    results.append(result)

if results:
    df_results = pd.DataFrame(results)
    output_file = f'model_synthetic_results.xlsx'
    df_results.to_excel(output_file, index=False)
    print(f"Results saved to {output_file}")

