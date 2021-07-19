# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 10:08:05 2019

@author: draz
"""
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import pandas as pd
import os
import numpy as np
from models import Model
import time
import _pickle as c_pickle


def model_grid(config_data):
    """This function takes the the configuration parameters and returns
    the grid model data.
    Args:
        config_data: the configuration parameters
    Returns:
        model: data and of the grid
    """
    start = time.time()
    for grid in config_data['grid']:
        model = Model(grid, simulation_time=config_data['simulation_time'])
        model.model()
        model.run_model(number_of_events=config_data['number_of_events'],
                        max_load_change=config_data['max_load_change'],
                        fault_clearing_time__cycles=config_data['fault_clearing_time__cycles'],
                        path=os.path.join(config_data['path'], 'results'))

        with open(os.path.join(config_data['path'], 'results', grid,
                               'pickles', '%s.pickle' % grid), 'wb') as data:
            c_pickle.dump(grid.Data, data)
    print('time = ', time.time() - start)
    return model


def get_dataset(grid, path):
    """This function takes the the configuration parameters and returns
    the grid model data.
    generated training data set resulted.
    Args:
        grid: string, grid model
        path: the directory where the pickle files are saved
    Returns:
        X_train: training input dataset in a 'numpy array'
        y_train: 'numpy array', training labels
        X_test:  'numpy array', test input data set
        y_test:  'numpy array', test labels dataset
        train:   'DataFrame', the training data together with labels
        test:    'DataFrame', the test data together with labels
        val:     'DataFrame', the validation data together with labels
        data:    'DataFrame', the all data before splitting

    """
    data_set = {'voltage_angel_data_' + grid: []}
    data_set.update({'events': None})
    with open(os.path.join(path, '%s.pickle' % grid), 'rb') as output:
        data_all = c_pickle.load(output)
        data_set['voltage_angel_data_' + grid] = data_all.bus_data_post_fault[1]
        data_set['events'] = data_all.df_events

    voltage_angel_data = data_set['voltage_angel_data_' + grid]

    v_removal = []
    a_removal = []
    for va_data in voltage_angel_data:
        dict_temp = list(va_data.values())[-1]
        voltages = []
        phases = []
        for key in dict_temp.keys():
            if 'm:u' in key:
                voltages.append(dict_temp[key])
            elif 'm:ph' in key:

                phases.append(dict_temp[key])

            else:
                continue
        v_removal.append(voltages)
        a_removal.append(phases)

    voltage_date_after_fault_removal = pd.DataFrame(v_removal)
    angel_date_after_fault_removal = pd.DataFrame(a_removal)

    data = pd.concat([voltage_date_after_fault_removal, angel_date_after_fault_removal], axis=1)
    stability = data_set['events'][['system_stability']]

    le = preprocessing.LabelEncoder()
    le.fit(stability['system_stability'].unique())
    labels = le.transform(stability)
    X_train, X_test, y_train, y_test = train_test_split(data,
                                                        labels, test_size=0.20, shuffle=False)

    data = pd.concat([data, pd.DataFrame(labels)], axis=1)
    data.columns = ['F_%s' % f for f in range(len(data.columns) - 1)] + ['stable-unstable']
    train, test = train_test_split(data, test_size=0.2)
    train, val = train_test_split(train, test_size=0.2)
    data_to_dump = [X_train, X_test, y_train, y_test, train, test, val, data]
    with open(os.path.join(path.replace('data', ''), 'data_set', '%s.pickle' % grid), 'wb') as data:
        c_pickle.dump(data_to_dump, data)
    return X_train, X_test, y_train, y_test, train, test, val, data


class DataClass:
    """data class contains the results of the simulation
    """
    def __init__(self, grid, path):
        """the instance contains the training data
        set which is used for training"""
        self.data_set = data_set = get_dataset(grid, path)
        self.X_train = data_set[0]
        self.X_test = data_set[1]
        self.y_train = pd.DataFrame(data_set[2])
        self.y_test = pd.DataFrame(data_set[3])
        self.train = data_set[4]
        self.test = data_set[5]
        self.val = data_set[6]
        self.data = data_set[7]
