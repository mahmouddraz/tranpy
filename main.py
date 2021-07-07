# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 17:21:21 2019

@author: draz
"""

import os
from dataset import model_grid, DataClass
from ml_modules import predict_with_svm, \
    predict_with_mlp, \
    predict_with_clf, \
    predict_with_ensemble, \
    predict_with_nn
import yaml
import warnings
import dalex as dx

warnings.filterwarnings(action='ignore', category=FutureWarning)


def get_config_data():
    """get the input data from config file
    Arguments: None
    Return: a dict with the configuration parameters
    """
    with open('input.yaml', 'r') as file:
        return yaml.load(file, Loader=yaml.FullLoader)


def main(_config_data, _algorithm, path):
    """the main function takes the configuration parameters as input
    and returns the predictions event consequences

    Arguments:
        :param _config_data: 'dict', input configuration parameters
        :param path: the directory where the training dataset is saved
        :param _algorithm: the selected ML algorithm
    Return:
         predictive measures of systems stability

    """
    if _config_data['key_data'] == 1:
        model_grid(_config_data)

    data_set = DataClass(_config_data['grid'], os.path.join(path, 'data'))
    
    
    if _algorithm == "svm":
        res = predict_with_svm(X_train=data_set.X_train,
                               y_train=data_set.y_train,
                               X_test=data_set.X_test,
                               y_test=data_set.y_test,
                               reduced = _config_data['reduced'],
                               grid =_config_data['grid'],
                               cross_validation=False)

    elif _algorithm == "mlp":
        res = predict_with_mlp(X_train=data_set.X_train,
                               y_train=data_set.y_train,
                               X_test=data_set.X_test, 
                               y_test=data_set.y_test,
                               reduced = _config_data['reduced'],
                               grid =_config_data['grid'],
                               cross_validation=False)

    elif _algorithm == "clf":
        res = predict_with_clf(X_train=data_set.X_train,
                               y_train=data_set.y_train,
                               X_test=data_set.X_test,
                               y_test=data_set.y_test,
                               reduced = _config_data['reduced'],
                               grid =_config_data['grid'],
                               cross_validation=False)
    elif _algorithm == "ensb":
        res = predict_with_ensemble(X_train=data_set.X_train,
                               y_train=data_set.y_train,
                               X_test=data_set.X_test,
                               y_test=data_set.y_test,
                               data_set =data_set,
                               reduced = _config_data['reduced'],
                               grid =_config_data['grid'],
                               config_data = _config_data,
                               cross_validation=False)
    
    elif 'nn' in _algorithm:
        res = predict_with_nn(data_set, _algorithm, _config_data,grid =_config_data['grid'])
    else:
        raise Exception('no algorithm is selected')

    return res


path_dir = os.path.dirname(os.path.realpath(__file__))

if __name__ == '__main__':
    config_data = get_config_data()
    results = []
    for algorithm in config_data['ml_algorithm']:
        res_alg = main(config_data, algorithm, path_dir)
        results.append(res_alg)

