import os
import sys
import dill
import numpy as np
import pandas as pd

from src.exception import CustomException
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
from src.logger import logging


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(X_train, y_train, X_test, y_test, models, params):
    try:
        model_report = {}
        trained_model = {}
        for name, model in models.items():
            param = params[name]  # getting the params using model name

            gs = GridSearchCV(estimator=model, param_grid=param, cv=3, n_jobs=-1)
            gs.fit(X_train, y_train)

            best_model = gs.best_estimator_

            y_test_pred = best_model.predict(X_test)
            score = r2_score(y_test, y_test_pred)

            model_report[name] = score
            trained_model[name] = best_model

            logging.info(f"{name} training successfull and data reported.")

        return model_report, trained_model

    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)
