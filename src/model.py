import logging
from typing import Tuple
import pandas as pd

from sklearn.linear_model import LinearRegression
from abc import ABC, abstractmethod

class Model(ABC):
    '''
    Abstract base class for model.
    '''
    @abstractmethod
    def train(self,x_train,y_train,**kwargs) -> None:
        '''
        Train the model.
        Args:
            x_train: Dataframe containing the features.
            y_train: Dataframe containing the labels.
        '''
        pass

class LinearRegressionModel(Model):
    '''
    Linear regression model.
    '''
    def train(self,x_train,y_train,**kwargs) -> None:
        '''
        Train the model.
        Args:
            x_train: Dataframe containing the features.
            y_train: Dataframe containing the labels.
        '''
        try:
            self.model = LinearRegression()
            self.model.fit(x_train,y_train)
        except Exception as e:
            logging.error(f"Failed to train model: {e}")
            raise e

    def predict(self,x_test) -> Tuple[pd.DataFrame,pd.DataFrame]:
        '''
        Predict the model.
        Args:
            x_test: Dataframe containing the features.
        Returns:
            y_pred: Dataframe containing the predicted labels.
        '''
        try:
            y_pred = self.model.predict(x_test)
            return y_pred
        except Exception as e:
            logging.error(f"Failed to predict model: {e}")
            raise e