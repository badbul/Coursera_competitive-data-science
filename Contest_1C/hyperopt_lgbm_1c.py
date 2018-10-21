from datetime import datetime 
import pickle
import numpy as np
import lightgbm as lgb
from hyperopt import hp
from hyperopt import fmin, Trials, tpe
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import cross_val_score, KFold
import sys
import json

RANDOM_STATE = 0
# Metrics
def mape(y_pred, y_test):
    denominator = y_test
    diff = np.abs(y_test - y_pred) / denominator
    diff[denominator == 0] = 0.0
    return np.mean(diff)

def metrics(y_pred, y_test): 
    mae = mean_absolute_error(y_pred, y_test)
    mape_ = mape(y_pred, y_test)
    return mae, mape_

def cv_mape(estimator, X, y, cv=3):
    skf = KFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE)
    mape_=[]
    for train_index, test_index in skf.split(X, y):
        estimator.fit(X.iloc[train_index], y.iloc[train_index])
        mape_.append(mape(np.expm1(estimator.predict(X.iloc[test_index])),np.expm1(y.iloc[test_index])))
    return mape_

def main(device="cpu"):
    # Read X, y
    with open('X_train.pkl', 'rb') as X_pkl:
        X_train = pickle.load(X_pkl)
    with open('y_train.pkl', 'rb') as y_pkl:
        y_train = pickle.load(y_pkl)

    with open('X_valid.pkl', 'rb') as X_pkl:
        X_valid = pickle.load(X_pkl)
    with open('y_valid.pkl', 'rb') as y_pkl:
        y_valid = pickle.load(y_pkl)

    start_t = datetime.now()
    print('start fitting..\n')

    params_space = {
                    'n_estimators': hp.choice('n_estimators', (500,1000,1500,2000)), 
                    'max_depth': hp.choice('max_depth', (6,10,15,25,30)),
                    'num_leaves': hp.choice('num_leaves', (50,100,150,175,200)),
                    'learning_rate': hp.uniform('learning_rate', 0.001, 0.2),
                    'reg_alpha': hp.uniform('reg_alpha', 0, 10),
                    'reg_lambda': hp.uniform('reg_lambda', 0, 10),
                    'min_data_in_leaf': hp.choice('min_data_in_leaf', (2,5,10,15,20,30)),
                    }

    def hyperopt_objective(params):
        
        model = lgb.LGBMRegressor(
                                    seed=RANDOM_STATE,
                                    n_estimators = params['n_estimators'],
                                    max_depth = params['max_depth'],
                                    num_leaves = params['num_leaves'],
                                    learning_rate = params['learning_rate'],
                                    reg_alpha = params['reg_alpha'],
                                    reg_lambda = params['reg_lambda'],
                                    min_data_in_leaf = params['min_data_in_leaf'],
                                    objective='rmse',
                                    nthread=-1,
                                    device=device
                                  )

        model.fit(X_train, y_train)
        lgb_pred = model.predict(X_valid)
        
        return round(np.sqrt(mean_squared_error(y_valid, lgb_pred)),4)


    trials = Trials()
    best_params = fmin(
                        hyperopt_objective,
                        space=params_space,
                        algo=tpe.suggest,
                        max_evals=50,
                        trials=trials,
                        verbose = 2
                      )
    # write best params to json
    with open('lgbm_params.json', 'w') as f:
        json.dump(best_params,f)
    pickle.dump(trials, open("trials.p", "wb"))
    
    stop_t = datetime.now()
    print('results: {}\n'.format(trials.best_trial['result']))
    print(device+'_time: {}'.format(stop_t-start_t))
    


if __name__ == '__main__':
    if len(sys.argv) < 1:
        print ("Warning: Insert device type as first argument")
    else: 
        main(sys.argv[1])
    #main()
    sys.exit()
