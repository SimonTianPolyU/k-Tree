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
import time
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_diabetes
                
datasets_details = {

    'airfoil_self_noise': {
        'file': 'airfoil_self_noise.csv',
        'features': ['frequency','aoa',	'cl',	'fsv',	'ssdt'],
        'target': 'sspl'
    },

    'auction_verification_time': {
        'file': 'auction_verification_time.csv',
        'features': ['process_b1_capacity',	'process_b2_capacity',	'process_b3_capacity',	'process_b4_capacity', 'property_price',	'property_product',	'property_winner'],
        'target':	'verification_time'
    },

    'auto_mpg': {
        'file': 'auto_mpg.csv',
        'features': ['cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model_year', 'origin_1',
                     'origin_2', 'origin_3'],
        'target': 'mpg'
    },

    'average_localization_error': {
        'file': 'average_localization_error.csv',
        'features': ['anchor_ratio', 'trans_range', 'node_density', 'iterations'],
        'target': 'ale'
    },

    'computer_hardware': {
        'file': 'computer_hardware.csv',
        'features': ['MYCT', 'MMIN', 'MMAX', 'CACH', 'CHMIN', 'CHMAX', 'PRP'],
        'target': 'ERP'
    },

    'concrete_compressive':
        {
            'file': 'concrete_compressive.csv',
            'features': ['Cement', 'Slag', 'Fly_ash', 'Water', 'SP', 'Coarse_Aggr.', 'Fine_Aggr.'],
            'target': 'Compressive(Mpa)'
        },

    'concrete_flow':
        {
            'file': 'concrete_flow.csv',
            'features': ['Cement', 'Slag', 'Fly_ash', 'Water', 'SP', 'Coarse_Aggr.', 'Fine_Aggr.'],
            'target': 'FLOW(cm)'
        },

    'concrete_slump': {
        'file': 'concrete_slump.csv',
        'features': ['Cement', 'Slag', 'Fly_ash', 'Water', 'SP', 'Coarse_Aggr.', 'Fine_Aggr.'],
        'target': 'SLUMP(cm)'
    },

    'energy_efficiency':
        {
            'file': 'energy_efficiency.csv',
            'features': ['X1',	'X2',	'X3',	'X4',	'X5',	'X6',	'X7',	'X8'],
            'target': 	'Y1'
        },

    'forest_fires':
        {
            'file': 'forest_fires.csv',
            'features': ['X', 'Y', 'month', 'day', 'FFMC', 'DMC', 'DC', 'ISI', 'temp', 'RH', 'wind', 'rain'],
            'target': 'area'
        },

    'heart_failure':
        {
            'file': 'heart_failure.csv',
            'features': ['age',	'anaemia',	'creatinine_phosphokinase',	'diabetes',	'ejection_fraction',	'high_blood_pressure',	'platelets',	'serum_creatinine',	'serum_sodium',	'sex',	'smoking',	'time'],
            'target': 'DEATH_EVENT'
        },

    'Intrusion_detection_in_WSNs':
        {
            'file': 'Intrusion_detection_in_WSNs.csv',
            'features': ['Area'	,'Sensing_Range'	,'Transmission_Range'	,'Number_of_Sensor_nodes'],
            'target': 'Number_of_Barriers'
        },

    'istanbul_stock_exchange':
        {
            'file': 'istanbul_stock_exchange.csv',
            'features': ['SP', 'DAX', 'FTSE', 'NIKKEI', 'BOVESPA', 'EU', 'EM'],
            'target': 'ISE'
        },

    'qsar_aquatic_toxicity': {
        'file': 'qsar_aquatic_toxicity.csv',
        'features': ['TPSA(Tot)', 'SAacc', 'H-050', 'MLOGP', 'RDCHI', 'GATS1p', 'nN', 'C-040'],
        'target': 'quantitative'
    },

    'qsar_fish_toxicity': {
        'file': 'qsar_fish_toxicity.csv',
        'features': ['CIC0', 'SM1_Dz(Z)', 'GATS1i', 'NdsCH', 'NdssC', 'MLOGP'],
        'target': 'quantitative'
    },

    'real_estate_valuation': {
        'file': 'real_estate_valuation.csv',
        'features': ['td', 'ha', 'dttnms', 'nocs', 'latitude', 'longitude'],
        'target': 'hpoua'
    },

    'wine_quality_red':
    {
            'file': 'wine_quality_red.csv',
            'features': ['fixed_acidity', 'volatile_acidity', 'citric_acid', 'residual_sugar', 'chlorides', 'free_sulfur_dioxide', 'total_sulfur_dioxide', 'density', 'pH', 'sulphates', 'alcohol'],
            'target': 'quality'
        },

    'yacht_hydrodynamics': {
        'file': 'yacht_hydrodynamics.csv',
        'features': ['LP', 'PC', 'LDR', 'BDR', 'LBR', 'FN'],
        'target': 'RR'
    }

}

base_path = os.getenv('DATASET_BASE_PATH', './Datasets/')
datasets = {}

diabetes_data = load_diabetes()

datasets['diabetes'] = {'data': diabetes_data.data, 'target': diabetes_data.target}

data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\\s+", skiprows=22, header=None)
boston_data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
boston_target = raw_df.values[1::2, 2]

datasets['boston'] = {'data': boston_data, 'target': boston_target}

# Loading datasets
for name, details in datasets_details.items():
    try:
        df = pd.read_csv(base_path + name + '/' + details['file'])
        X = df[details['features']]
        y = df[details['target']]
        datasets[name] = {'data': X.values, 'target': y.values}
    except Exception as e:
        print(f"Error loading {name}: {e}")


def split_data(X, y, random_state=None):
    # Splitting into training (80%) and test (20%)
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )

    # Further splitting the training into training (60% of 80%) and validation (20% of 80%)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.25, random_state=random_state
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


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
    n_estimators_options = [100, 200]
    best_min_samples = None
    best_max_depth = None
    best_n_estimators = None

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
                current_validation_time = time.time() - start_time

                if current_validation_time > 1200:
                    return {'min_samples_leaf': best_min_samples, 'max_depth': best_max_depth, 'k': best_k}, 'not enough time'

    return {'min_samples_leaf': best_min_samples, 'max_depth': best_max_depth, 'k': best_k}, best_score

def train_PkRF_with_validation(X_train, y_train, X_val, y_val, N_train):
    k_range = [int(np.sqrt(N_train)) - i for i in range(4, -5, -2)]
    min_samples_per_leaf_options = [5, 10, 15, 20]
    max_depth_options = [4, 5, 6, 7, 8, 9, 10]
    max_features_options = [X_train.shape[1]]
    n_estimators_options = [100, 200]

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

                        current_validation_time = time.time() - start_time
                        if current_validation_time > 1200:
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

    start_time = time.time()
    for k in k_range:
        for min_samples in min_samples_per_leaf_options:
            for max_depth in max_depth_options:
                # dt_params = {'min_samples_leaf': min_samples, 'max_depth': max_depth}
                model = AkTree(min_samples=min_samples, max_depth=max_depth, k=k)
                model.fit(X_train, y_train)
                predictions_val = model.predict_all(X_val)[0]
                score = mean_squared_error(y_val, predictions_val)
                if score < best_score:
                    best_score = score
                    best_k = k
                    best_min_samples = min_samples
                    best_max_depth = max_depth

                current_validation_time = time.time() - start_time
                if current_validation_time > 1200:
                    # If the time exceeds 20 minutes, return the best parameters found so far
                    return {'min_samples_leaf': best_min_samples, 'max_depth': best_max_depth, 'k': best_k}, 'not enough time'

    return {'min_samples_leaf': best_min_samples, 'max_depth': best_max_depth, 'k': best_k}, best_score


results = []
seed = 42

for name, data in datasets.items():
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(data['data'], data['target'],random_state=seed)
    P, N = X_train.shape[1], len(X_train) + len(X_val) + len(X_test)  # Update P and N

    print(name,P,N)

    X_train_final = np.concatenate((X_train, X_val))
    y_train_final = np.concatenate((y_train, y_val))

    # Training each model with validation set
    kNN_params, kNN_best_score = train_kNN_with_validation(X_train, y_train, X_val, y_val, len(X_train))
    print("kNN done")

    CART_params, CART_best_score = train_CART_with_validation(X_train, y_train, X_val, y_val)
    print("CART done")

    RF_params, RF_best_score = train_RF_with_validation(X_train, y_train, X_val, y_val)
    print("RF done")

    # Similar for PkTree, AkTree, and SkTree
    PkTree_params, PkTree_best_score = train_PkTree_with_validation(X_train, y_train, X_val, y_val, len(X_train))
    print("PkTree done")

    AkTree_params, AkTree_best_score = train_AkTree_with_validation(X_train, y_train, X_val, y_val, len(X_train))
    print("AkTree done")

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
    RF_model = RandomForestRegressor(min_samples_leaf=RF_params['min_samples_leaf'], max_depth=RF_params['max_depth'], n_estimators=100)
    RF_model.fit(X_train_final, y_train_final)
    RF_training_time = time.time() - start_time

    start_time = time.time()
    PkTree_model = PkTree(
        dt_params={'min_samples_leaf': PkTree_params['min_samples_leaf'], 'max_depth': PkTree_params['max_depth']},
        k=PkTree_params['k'])
    PkTree_model.fit(X_train_final, y_train_final)
    PkTree_training_time = time.time() - start_time

    start_time = time.time()
    AkTree_model = AkTree(min_samples=AkTree_params['min_samples_leaf'], max_depth=AkTree_params['max_depth'],
                          k=AkTree_params['k'])
    AkTree_model.fit(X_train_final, y_train_final)
    AkTree_training_time = time.time() - start_time

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
    AkTree_predictions, AkTree_neighbor_proportions_in_adjacent = AkTree_model.predict_all(X_test)
    AkTree_prediction_time = time.time() - start_time
    AkTree_mse = mean_squared_error(y_test, AkTree_predictions)

    start_time = time.time()
    PkRF_predictions, PkRF_neighbor_proportions_in_adjacent = PkRF_model.predict(X_test)
    PkRF_prediction_time = time.time() - start_time
    PkRF_mse = mean_squared_error(y_test, PkRF_predictions)

    result = {
        'dataset': name,
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
        'AkTree_params': AkTree_params,
        'AkTree_best_score': AkTree_best_score,
        'AkTree_mse': AkTree_mse,
        'AkTree_training_time': AkTree_training_time,
        'AkTree_prediction_time': AkTree_prediction_time,
        'AkTree_neighbor_proportions_in_adjacent': AkTree_neighbor_proportions_in_adjacent,
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
    output_file = f'model_results_real_datasets.xlsx'
    df_results.to_excel(output_file, index=False)
    print(f"Results saved to {output_file}")








