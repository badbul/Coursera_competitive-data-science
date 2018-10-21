# coding=utf-8
import pandas as pd
import numpy as np
import tqdm
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression, Lasso, ElasticNet, SGDRegressor, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA

RANDOM_STATE = 0

class stack_model:
    def __init__(self, base_models, meta_models, X_train, y_train, X_valid, y_valid, X_test, y_test):
         self.X_train_1 = X_train[:int(X_train.shape[0]*0.65)]
         self.y_train_1 = y_train[:int(y_train.shape[0]*0.65)]
         self.X_train_2 = X_train[int(X_train.shape[0]*0.65):]
         self.y_train_2 = y_train[int(y_train.shape[0]*0.65):]
         self.X_valid = X_valid
         self.y_valid = y_valid
         self.X_test = X_test 
         self.y_test = y_test
         self.b_models = base_models
         self.m_models = meta_models

    # Вootstrap split
    def get_bootstrap_samples(self, X, y, n_samples, seed):  
        X_train = []
        y_train = []
        # генерируем матрицу индексов подвыборок из генеральной выборки
        np.random.seed(seed*1000)
        indices = np.random.randint(0, y.size, (n_samples, y.size)) 
        # массив выборок
        for i in range(n_samples):
            X_train.append(X[indices[i]])
            y_train.append(y[indices[i]])
        return X_train, y_train

   
    def base_model(self, X_train, y_train):
        model_lst=[]
        for model in self.b_models:
            for X_trn, y_trn in tqdm.tqdm(zip(X_train, y_train), total=len(X_train)): 
                for model in self.b_models:
                    model.fit(X_trn, y_trn)
                    model_lst.append(model)
            return model_lst  

    def meta_model(self, X_train_meta, y_train_meta):
        model_lst=[]
        for model in self.m_models:
            for X_trn, y_trn in zip(X_train_meta, y_train_meta): 
                for model in self.m_models:
                    model.fit(X_trn, y_trn)
                    model_lst.append(model)
            return model_lst  


    def fit(self):

        print('Start fitting..\n\n')

        print('Train base model..')
        X_train_1, y_train_1 = self.get_bootstrap_samples(self.X_train_1, self.y_train_1, 5, 0)
        B_models = self.base_model(X_train_1, y_train_1)

        print('Predict meta features..')
        MF2 = [m.predict(self.X_train_2) for m in B_models]
        MF2 = np.array(MF2).T

        MFV = [m.predict(self.X_valid) for m in B_models]
        MFV = np.array(MFV).T

        MFT = [m.predict(self.X_test) for m in B_models]
        MFT = np.array(MFV).T

        print('Fit/Validate meta model..')
        #--train
        M_models = self.meta_model([MF2], [self.y_train_2])
        #---validate
        valid_pred = [m.predict(MFV) for m in M_models]
        #----average prediction
        val_pred_avg = np.vstack(valid_pred).T.mean(axis=1)
        print('val_rmse:{}'.format(round(np.sqrt(mean_squared_error(self.y_valid, val_pred_avg)),4)))

        print('Fit/test full meta model..')
        #--train
        M_models = self.meta_model([np.vstack([MF2, MFV])], [np.vstack([self.y_train_2,
                                                                        self.y_valid]
                                                                      )]) 
        #---test
        test_pred = [m.predict(MFT) for m in M_models]
        test_pred_avg = np.vstack(test_pred).T.mean(axis=1)



        return val_pred_avg, test_pred_avg
        


        
    
