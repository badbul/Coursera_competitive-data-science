from datetime import datetime 
import pickle
import numpy as np
import xgboost as xgb
from hyperopt import hp
from hyperopt import fmin, Trials, tpe
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import cross_val_score, KFold
import sys
import json

RANDOM_STATE = 0

class xgbmodel:
    def __init__(self, xgb, params):
        self.params = params
        self.model = None
        self.xgb = xgb
        
    def fit(self, X, y):
        xgbtrain = self.xgb.DMatrix(X, y)
        self.model = self.xgb.train(self.params, xgbtrain) 
        
    def predict(self, X_test):
        return self.model.predict(self.xgb.DMatrix(X_test))

def main():
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
                    'num_round': hp.choice('num_round', (500,1000,1200,1500)), 
                    'max_depth': hp.choice('max_depth', (5,10,12,15,20)),
                    'eta': hp.uniform('eta', 0.01, 0.35),
                    'alpha': hp.uniform('reg_alpha', 0, 10),
                    'lambda': hp.uniform('reg_lambda', 0, 10),
                    'min_child_weight': hp.uniform('min_child_weight', 0.5, 10),
                    }

    def hyperopt_objective(params):

        xgb_params = {  'max_depth':params['max_depth'], 
                        'num_round': params['num_round'],
                        'min_child_weight': params['min_child_weight'],
                        'eta': params['eta'], 
                        'alpha': params['alpha'],
                        'lambda': params['lambda'],
                        'seed':1,
                        'silent':0,
                        'eval_metric':'rmse'
                      }
        
        model = xgbmodel(xgb, xgb_params)

        model.fit(X_train, y_train)
        xgb_pred = model.predict(X_valid)
        
        return round(np.sqrt(mean_squared_error(y_valid, xgb_pred)),4)


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
    with open('xgbm_params.json', 'w') as f:
        json.dump(best_params,f)
    pickle.dump(trials, open("trials_xgb.p", "wb"))
    
    stop_t = datetime.now()
    print('results: {}\n'.format(trials.best_trial['result']))
    print('perf_time: {}'.format(stop_t-start_t))    


if __name__ == '__main__':
    main()
    sys.exit()
