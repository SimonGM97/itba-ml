from itba_ml.utils.logging_helper import get_logger
import pandas as pd
import numpy as np

# Non-deep learning ML models
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor

# Deep learning
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam

import pickle
from collections.abc import Iterable
from typing import Any, List
from copy import deepcopy
import os


LOGGER = get_logger(
    name=__name__,
    level='DEBUG'
)


class Regressor:
    save_attrs = [
        'algorithm',
        'hyper_parameters',
        'fitted'
    ]
    
    def __init__(
        self,
        algorithm: str = None,
        hyper_parameters: dict = None,
    ) -> None:
        self.model = None
        self.algorithm = algorithm
        self.hyper_parameters = None
        if hyper_parameters is not None:
            self.hyper_parameters = self.prepare_hyper_parameters(deepcopy(hyper_parameters))
            
        self.fitted = False
        
    def prepare_hyper_parameters(
        self,
        hyper_parameters: dict
    ) -> None:
        if self.algorithm == 'random_forest':
            names = list(hyper_parameters.keys()).copy()
            for param_name in names:
                if 'random_forest.' in param_name:
                    correct_name = param_name.replace('random_forest.', '')
                    hyper_parameters[correct_name] = hyper_parameters.pop(param_name)
            
            hyper_parameters.update(**{
                'n_jobs': -1,
                'random_state': 23111997
            })
        
        if self.algorithm == 'lightgbm':
            names: List[str] = list(hyper_parameters.keys()).copy()
            for param_name in names:
                if 'lightgbm.' in param_name:
                    correct_name = param_name.replace('lightgbm.', '')
                    hyper_parameters[correct_name] = hyper_parameters.pop(param_name)
                
            hyper_parameters.update(**{
                "objective": 'regression',
                "importance_type": 'split',
                "random_state": 23111997,
                "n_jobs": -1,
                "verbose": -1
            })
                
        return hyper_parameters
                
    def build(
        self,
        train_target: pd.Series = None, 
        train_features: pd.DataFrame = None
    ) -> None:
        if self.algorithm == 'random_forest':
            self.model = RandomForestRegressor(**self.hyper_parameters)
            
        elif self.algorithm == 'lightgbm':
            self.model = LGBMRegressor(**self.hyper_parameters)
            
        elif self.algorithm == 'lstm':
            # Define a Sequential model
            self.model = Sequential()
            
            # Add first layer
            self.model.add(LSTM(
                self.hyper_parameters['units'],
                input_shape=(train_features.shape[1], train_features.shape[0]), # train_features.shape[2]
                dropout=self.hyper_parameters['dropout'], 
                recurrent_dropout=self.hyper_parameters['recurrent_dropout']
            ))
            
            # Add subsequent LSTM layers
            if self.hyper_parameters['layers'] > 1:
                mult_list = (np.geomspace(1, 2, self.hyper_parameters['layers']+1)[::-1] - 1)[:-1]
                for mult in mult_list[1:]:
                    self.model.add(LSTM(
                        int(self.hyper_parameters['units'] * mult), 
                        dropout=self.hyper_parameters['dropout'], 
                        recurrent_dropout=self.hyper_parameters['recurrent_dropout']
                    ))
            
            # Add Dense layer and compile
            self.model.add(Dense(1))
            
            # Describe Optimizer
            optimizer = Adam(
                learning_rate=self.hyper_parameters['lstm.learning_rate']
            )
            
            self.model.compile(
                loss="mse", # "mae"
                optimizer=optimizer
            )
        
        self.fitted = False
            
    def fit(
        self,
        train_target: pd.Series = None, 
        train_features: pd.DataFrame = None
    ) -> None:
        if self.algorithm == 'random_forest':
            self.model.fit(
                train_features.values.astype(float), 
                train_target.values.astype(float)
            )
        
        elif self.algorithm == 'lightgbm':
            self.model.fit(
                train_features.values.astype(float), 
                train_target.values.astype(float)
            )
            
        elif self.algorithm == 'lstm':
            early_stopping = EarlyStopping(patience=5)
            history = self.model.fit(
                train_features, 
                train_target, 
                epochs=20,
                batch_size=self.hyper_parameters['lstm.batch_size'],
                verbose=0,
                callbacks=[early_stopping],
                # use_multi_gpus=True
            )
            loss = history.history["loss"][-1]
            
        self.fitted = True
            
    def predict(
        self,
        train_target: pd.Series = None,
        forecast_features: pd.DataFrame = None, 
        forecast_dates: Iterable = None
    ):
        if self.algorithm == 'random_forest':
            return self.model.predict(
                forecast_features.values.astype(float)
            )
        
        elif self.algorithm == 'lightgbm':
            return self.model.predict(
                forecast_features.values.astype(float)
            )
        
        elif self.algorithm == 'lstm':
            return self.model.predict(
                forecast_features.values.astype(float)
            )
        
    def save(
        self,
        as_champion: bool = True
    ):
        save_attrs = {
            attr_name: attr_value for attr_name, attr_value in self.__dict__.items()
            if attr_name in self.save_attrs
        }
        
        if as_champion:
            LOGGER.info('Saving new champion.')
            
            # Define paths
            model_path = os.path.join('..', 'models', 'regressor', 'champion', 'champion.pickle')
            attrs_path = os.path.join('..', 'models', 'regressor', 'champion', 'attrs.pickle')
            
            
        else:
            LOGGER.info('Saving new challenger.')
            
            # Define paths
            model_path = os.path.join('..', 'models', 'regressor', 'challenger', 'challenger.pickle')
            attrs_path = os.path.join('..', 'models', 'regressor', 'challenger', 'attrs.pickle')
                
        # Save self.model            
        with open(model_path, 'wb') as file:
            pickle.dump(self.model, file)

        # Save attrs            
        with open(attrs_path, 'wb') as file:
            pickle.dump(save_attrs, file)
            
    def load(
        self,
        champion: bool = True
    ):
        if champion:
            LOGGER.info('Loading champion.')
            
            # Define paths
            model_path = os.path.join('..', 'models', 'regressor', 'champion', 'champion.pickle')
            attrs_path = os.path.join('..', 'models', 'regressor', 'champion', 'attrs.pickle')
            
            
        else:
            LOGGER.info('Loading challenger.')
            
            # Define paths
            model_path = os.path.join('..', 'models', 'regressor', 'challenger', 'challenger.pickle')
            attrs_path = os.path.join('..', 'models', 'regressor', 'challenger', 'attrs.pickle')
                
        # Load self.model            
        with open(model_path, 'rb') as file:
            self.model = pickle.load(file)

        # Load attrs            
        with open(attrs_path, 'rb') as file:
            attrs: dict = pickle.load(file)
            
        for attr_name, attr_value in attrs.items():
            setattr(self, attr_name, attr_value)                