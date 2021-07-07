import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import shap
import dalex as dx
from timeit import default_timer as timer
from tensorflow.keras.callbacks import Callback
from keras import backend as K
from sklearn import tree
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import accuracy_score
from sklearn.ensemble import VotingClassifier
from tensorflow import feature_column
from explainers import SmartGridExplainer
from eli5.sklearn import PermutationImportance
from eli5 import explain_weights_df
from collections import Counter

feature_names = []
bus=1
for i in range(77):
    if (i % 2 == 0):
        feature_names.append('bus_'+str(bus)+':u')
    else:
        feature_names.append('bus_'+str(bus)+':phi')
        bus += 1

index_featurer_names = {}
bus=1
for i in range(77):
    if (i % 2 == 0):
        index_featurer_names[i] = ('bus_'+str(bus)+':u')
    else:
        index_featurer_names[i] = ('bus_'+str(bus)+':phi')
        bus += 1

def predict_with_svm(X_train,
                     y_train,
                     X_test,
                     y_test,
                     reduced,
                     grid,
                     cross_validation=False):
    """the following function is used when the ML method support vector
    machine is activated in the config file
    :arg X_train: The training input dataset in a 'numpy array'
    :arg y_train: The training labels in a 'numpy array'
    :arg X_test: The test input data set in a 'numpy array'
    :arg y_test: The test labels dataset in a 'numpy array'
    :arg cross_validation:  True or False 'Default=False'
    :returns res: confusion matrix between the predictions and labels
    :returns TypeError: if arguments data sets is not numpy array
     """
    scores = []
    if reduced ==0:
        n=1
    else: 
        n=50
    
    for i in range(n):

        svm_classifier = SVC(kernel='rbf', probability=True)

        start = timer()
        svm_classifier.fit(X_train, y_train)
        end = timer()
        trainingtime = end - start

        start = timer()
        y_predict_smv = svm_classifier.predict(X_test)

        end = timer()
        predictiontime = end - start
        print(classification_report(y_test, y_predict_smv))
        print(confusion_matrix(y_test, y_predict_smv))
        print(trainingtime, ' ', predictiontime)

        if cross_validation is True:
            parameters = {"alpha": [6]}
            n = 6
            gs_cv_block_smv = GridSearchCV(SVC(kernel='rbf'), parameters,
                                           scoring="neg_mean_squared_error",
                                           n_jobs=-1, return_train_score=True,
                                           cv=KFold(n_splits=n, shuffle=False))

            gs_cv_block_smv.fit(X_train, y_train)
            y_hat = gs_cv_block_smv.predict(X_test)
            print("Test RMSE Block: %.2f" % np.sqrt(mse(y_test, y_hat)), "\n")

            # Cross Validation Shuffle
            gs_cv_shuffle = GridSearchCV(svm_classifier, parameters,
                                         scoring="neg_mean_squared_error",
                                         cv=KFold(n_splits=n, shuffle=True),
                                         n_jobs=-1, return_train_score=True)
            gs_cv_shuffle.fit(X_train, y_train)
            print(gs_cv_shuffle.best_params_)
            y_hat_shuffle = gs_cv_shuffle.predict(X_test)
            print("Test RMSE Shuffle: %.2f" %
                  np.sqrt(mse(y_test, y_hat_shuffle)), "\n")

        svmxai = SmartGridExplainer(
            X_train=X_train, X_test=X_test, model=svm_classifier, grid=grid)

        spobj = svmxai.generateGlobalLimeExplanations()

        shaps = svmxai.generateGlobalShapExplanations()

        
        #Get SHAPS
        topshaps = svmxai.get_top_SHAP_important_features(shaps,X_test.shape[1] )
        names = index_featurer_names
        shap_feats_score = []
        list1 =  [x[0] for x in topshaps]
        list1 = list(map(lambda x: names[x], list1))

        list2 =  [x[1] for x in topshaps]
        result = list(zip(list1,list2))

            


        indexes = []
        for m in range(1, X_test.shape[1]):
            indexes.append(svmxai.get_top_important_features_SHAP(shaps, m))

        indexeslime = []
        for m in range(1, X_test.shape[1]):
            indexeslime.append(svmxai.get_top_important_features_LIME(spobj, m))

        scoreins = []

        if(reduced == 1):
            for index in indexeslime:
                print('Number of n: ', len(index))
                predictions = svmxai.predict_with_modified_features(
                    index, False)
                accuracy = classification_report(
                    y_test, predictions, output_dict=True)
                scoreins.append(accuracy['accuracy'])
                print(accuracy['accuracy'])
        scores.append(scoreins)
    if(reduced==1):
        print(i)
        accuracies = np.array(scores)
        result = accuracies.mean(axis=0)
        df = pd.DataFrame(result)
        filepath = 'svm_score_lime.xlsx'
        df.to_excel(filepath, index=False)

    #Permutation Importances
    perm = PermutationImportance(svm_classifier).fit(X_test, y_test)
    importances = explain_weights_df(perm, feature_names=feature_names,targets=['unstable','stable'])

    #Break Down Plot
    # newengland_svm_exp = dx.Explainer(svm_classifier, X_test, y_test, 
    #               label = "Smart Grid New England Pipeline")
    
    # instance = newengland_svm_exp.predict_parts(X_test.iloc[1], 
    #          type = 'break_down')
    # instance.plot(max_vars = 30)
    

    
    
    
    




    
    
    return classification_report(y_test, y_predict_smv), confusion_matrix(y_test, y_predict_smv)


def predict_with_mlp(X_train,
                     y_train,
                     X_test,
                     y_test,
                     reduced,
                     grid,
                     cross_validation=False):
    """this function is used when the ML mlp in the config file is chosen
    :arg X_train: The training input dataset in a 'numpy array'
    :arg y_train: The training labels in a 'numpy array'
    :arg X_test: The test input data set in a 'numpy array'
    :arg y_test: The test labels dataset in a 'numpy array'
    :arg cross_validation:  True or False 'Default=False'

    :returns: confusion matrix between the predictions and labels

    :raises: TypeError: if arguments data sets is not numpy array
    """

    scores = []
    if reduced ==0:
        n=1
    else: 
        n=50
    
    for i in range(n):

        mlp = MLPClassifier()
        print(X_train.shape)
        print(X_test.shape)

        start = timer()
        mlp.fit(X_train, y_train)
        end = timer()
        trainingTime = end-start

        start = timer()
        y_predict_mlp = mlp.predict(X_test)
        end = timer()
        predictionTime = end - start
        print(classification_report(y_test, y_predict_mlp))
        print(confusion_matrix(y_test, y_predict_mlp))
        print(trainingTime, ' ', predictionTime)
        if cross_validation is True:
            parameters = {"alpha": [6]}
            n = 6
            gs_cv_block_mlp = GridSearchCV(MLPClassifier(), parameters,
                                           scoring="neg_mean_squared_error",
                                           cv=KFold(n_splits=n),
                                           n_jobs=-1, return_train_score=True)

            gs_cv_block_mlp.fit(X_train, y_train)
            y_hat = gs_cv_block_mlp.predict(X_test)
            print("Test RMSE Block: %.2f" % np.sqrt(mse(y_test, y_hat)), "\n")

            # Cross Validation
            gs_cv_shuffle = GridSearchCV(mlp, parameters,
                                         scoring="neg_mean_squared_error",
                                         cv=KFold(n_splits=n, shuffle=True),
                                         n_jobs=-1, return_train_score=True)
            gs_cv_shuffle.fit(X_train, y_train)
            print(gs_cv_shuffle.best_params_)
            y_hat_shuffle = gs_cv_shuffle.predict(X_test)
            print("Test RMSE Shuffle: %.2f" %
                  np.sqrt(mse(y_test, y_hat_shuffle)), "\n")

        # perm = PermutationImportance(mlp).fit(X_test, y_test)
        # importances = explain_weights_df(perm, feature_names=feature_names,targets=['unstable','stable'])
        # print(importances)

        # df = pd.DataFrame(importances)
        # filepath = 'permutationimportance_mlp.xlsx'
        # df.to_excel(filepath, index=False)

        print('MLP START')
        mlpxai = SmartGridExplainer(
            X_train=X_train, X_test=X_test, model=mlp, grid=grid)
        mlpxai.generateLocalLimeExplanation(2)

        spobj = mlpxai.generateGlobalLimeExplanations()

        shap = mlpxai.generateGlobalShapExplanations()

        

    
        indexes = []
        for m in range(1, X_test.shape[1]):
            indexes.append(
                mlpxai.get_top_important_features_SHAP(shap, m))
        

        indexeslime = []
        for m in range(1, X_test.shape[1]):
            indexeslime.append(mlpxai.get_top_important_features_LIME(spobj, m))

        scoreins = []

        print()
        if(reduced == 1):
            for index in indexeslime:
                print('Number of n: ', len(index))
                predictions = mlpxai.predict_with_modified_features(
                    index, False)
                accuracy = classification_report(
                    y_test, predictions, output_dict=True)
                scoreins.append(accuracy['accuracy'])
                print(classification_report(y_test, predictions))
        
        scores.append(scoreins)
    print(i)
    if(reduced==1):
        result = np.array(scores)
        result = result.mean(axis=0)
        df = pd.DataFrame(result)
        filepath = 'mlp_score_lime.xlsx'
        df.to_excel(filepath, index=False)
        print('MLP END')


    

def predict_with_clf(X_train,
                     y_train,
                     X_test,
                     y_test,
                     reduced,
                     grid,
                     cross_validation=False):
    """the transient stability can be predicted
    by decision trees using this function. It takes the training, test data set as
    input and returns the confusion matrix between the predicted values and
    the true.

    Args:
        X_train: The training input dataset in a 'numpy array'
        y_train: The training labels in a 'numpy array'
        X_test: The test input data set in a 'numpy array'
        y_test: The test labels dataset in a 'numpy array'
        cross_validation:  True or False 'Default=False'

    Returns:
        confusion matrix between the predictions and labels

     Raises:
        TypeError: if arguments data sets is not numpy array
    """
    scores = []
    accscores = []
    if reduced ==0:
        n=1
    else: 
        n=50
    for i in range(n):
        clf = tree.DecisionTreeClassifier()
        start = timer()
        clf.fit(X_train, y_train)
        end = timer()
        trainingTime = end - start

        start = timer()
        y_predict_clf = clf.predict(X_test)
        end = timer()
        predictionTime = end-start
        accuracy = classification_report(
                        y_test, y_predict_clf, output_dict=True)
        accscores.append(accuracy['accuracy'])
        print(classification_report(y_test, y_predict_clf))
        print(confusion_matrix(y_test, y_predict_clf))
        print(trainingTime, ' ', predictionTime)
        if cross_validation is True:
            parameters = {"alpha": [6]}
            n = 6
            gs_cv_block_clf = GridSearchCV(tree.DecisionTreeClassifier(), parameters,
                                           scoring="neg_mean_squared_error",
                                           cv=KFold(n_splits=n, shuffle=False),
                                           n_jobs=-1, return_train_score=True)
            gs_cv_block_clf.fit(X_train, y_train)
            y_hat = gs_cv_block_clf.predict(X_test)
            print("Test RMSE Block: %.2f" % np.sqrt(mse(y_test, y_hat)), "\n")

            # Cross Validation Shuffle
            gs_cv_shuffle = GridSearchCV(clf, parameters,
                                         scoring="neg_mean_squared_error",
                                         cv=KFold(n_splits=n, shuffle=True),
                                         n_jobs=-1, return_train_score=True)
            gs_cv_shuffle.fit(X_train, y_train)
            print(gs_cv_shuffle.best_params_)
            y_hat_shuffle = gs_cv_shuffle.predict(X_test)
            print("Test RMSE Shuffle: %.2f" %
                  np.sqrt(mse(y_test, y_hat_shuffle)), "\n")

        print('CLF START')
        clfxai = SmartGridExplainer(
            X_train=X_train, X_test=X_test, model=clf, grid=grid)
        clfxai.generateLocalLimeExplanation(2)

        spobj = clfxai.generateGlobalLimeExplanations()

        shap = clfxai.generateGlobalShapExplanations()

        

        # perm = PermutationImportance(clf).fit(X_test, y_test)
        # importances = explain_weights_df(perm, feature_names=feature_names,targets=['unstable','stable'])
        # df = pd.DataFrame(importances)
        # filepath = 'permutationimportance_clf.xlsx'
        # df.to_excel(filepath, index=False)

        

        indexes = []
        for m in range(1, X_test.shape[1]):
            indexes.append(clfxai.get_top_important_features_SHAP(shap, m))
        
        indexeslime = []
        for m in range(1, X_test.shape[1]):
            indexeslime.append(clfxai.get_top_important_features_LIME(spobj, m))
        

            

        scoreins = []
        print()
        if(reduced == 1):
            for index in indexeslime:
                print('Number of n: ', len(index))
                predictions = clfxai.predict_with_modified_features(
                    important_columns=index, important_removed=False)
                accuracy = classification_report(
                    y_test, predictions, output_dict=True)
                scoreins.append(accuracy['accuracy'])
                print(accuracy['accuracy'])
        scores.append(scoreins)
    
    if(reduced==1):
        print(i)
        accuracies = np.array(scores)
        print()
        result = accuracies.mean(axis=0)
        df = pd.DataFrame(result)
        filepath = 'clf_score_lime.xlsx'
        df.to_excel(filepath, index=False)

    print('CLF END')
    


def df_to_dataset(df, shuffle=True, batch_size=32):
    """this function takes any data set in a form
    of pandas data frames (e.g. train, test)
    and convert it to tensors.
    :arg df: a data frame with string columns names
    :arg shuffle: 'Default=True'
    :arg batch_size: 'Default=32'
    :returns feature and labels batches as 'tensors'
    """
    df = df.copy()
    labels = df.pop('stable-unstable')

    ds = tf.data.Dataset.from_tensor_slices((dict(df), labels))

    if shuffle:
        ds = ds.shuffle(buffer_size=len(df))
    dataset = ds.batch(batch_size)

    return dataset


def get_feature_layer(i):
    """this function creates a feature
    layer to be used later in Keras model.
    :returns Keras Keras feature layer
    :raises does not raise any
    """
    feature_columns = []
    columns = ['F_%s' % f for f in range(i)]

    for header in columns:
        feature_columns.append(feature_column.numeric_column(header))

    feature_layer = tf.keras.layers.DenseFeatures(feature_columns)
    return feature_layer


def construct_model(config_data, algorithm, train_ds):
    """this function creates the DL model
    Args:
        does not need any
    Returns:
        Keras Sequential model
    Raises:
        does not raise any
    """

    print(train_ds.shape)
    if algorithm == 'dnn':
        return tf.keras.Sequential([
            get_feature_layer((train_ds.shape[1] - 1)),
            tf.keras.layers.Dense(10, activation=tf.nn.relu),
            tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)
        ])
    elif algorithm == 'rnn':
        return tf.keras.Sequential([
            tf.keras.layers.LSTM(150, activation='tanh',
                                 input_shape=(config_data['n_steps'],
                                              train_ds.shape[1] - 1),
                                 return_sequences=True),
            tf.keras.layers.LSTM(150, activation='tanh',
                                 return_sequences=True),
            tf.keras.layers.LSTM(150, activation='tanh',
                                 return_sequences=True),
            tf.keras.layers.Dense(units=1, activation='sigmoid')
        ])


def predict_with_nn(dataset, algorithm, config_data, grid):
    """this function calls the above mentioned functions,
    compiles the Keras model and perform the training.

    Args:
        dataset: The training input and labels dataset in a 'pandas data frame'
        algorithm: the selected algorithm
        config_data: configuration parameters

    Returns:
        confusion matrix between the predictions and labels

     Raises:
        TypeError: if arguments data sets is not data frame
    """

    reduced = config_data['reduced']
    scores = []
    if reduced ==0:
        n=1
    else: 
        n=50
    for i in range(n):
        tf.keras.backend.clear_session()
        # Predicting with reduced Dataset

        # Predicting with normal Dataset
        print('Test')
        train_ds = df_to_dataset(dataset.train, shuffle=False,
                                 batch_size=config_data['batch_size'])
        val_ds = df_to_dataset(dataset.val,
                               shuffle=False, batch_size=config_data['batch_size'])

        model = construct_model(config_data, algorithm, dataset.train)
        model.compile(optimizer=config_data['optimizer'], metrics=['accuracy'],
                      loss=tf.losses.binary_crossentropy),

        #callbacks_list = [StopTraining()]
        if algorithm == 'dnn':

            start = timer()
            model.fit(train_ds, validation_data=val_ds,
                      epochs=config_data['epochs'])
            end = timer()
            trainingtime = end-start
            model._name = "DNN"

            test_ds = dataset.test

            test_ds = df_to_dataset(test_ds,
                                    shuffle=False, batch_size=config_data['batch_size'])

            start = timer()
            model.evaluate(test_ds)
            end = timer()
            predictiontime = end-start
            print(trainingtime, predictiontime)

        elif algorithm == 'rnn':

            X_train = np.array(dataset.X_train.values)

            X_test = np.array(dataset.X_test.values)

            trainX = np.reshape(X_train, (dataset.X_train.shape[0], config_data['n_steps'],
                                          dataset.X_train.shape[1]))
            testX = np.reshape(X_test, (dataset.X_test.shape[0], config_data['n_steps'],
                                        dataset.X_test.shape[1]))

            start = timer()
            model.fit(trainX, dataset.y_train, epochs=config_data['epochs'])
            end = timer()
            trainingtime = end - start

            start = timer()
            model.evaluate(testX, dataset.y_test)
            end = timer()
            predictiontime = end - start

            print(trainingtime, predictiontime)
            model._name = "RNN"

        print(model._name, ' START')
        nnxai = SmartGridExplainer(
            X_train=np.array(dataset.X_train.values), X_test=np.array(dataset.X_test.values), model=model, grid=grid)
        nnxai.generateLocalLimeExplanation(2)

        spobj = nnxai.generateGlobalLimeExplanations()

        shap = nnxai.generateGlobalShapExplanations()

        


        

        #Permutation Importance for dnn
        # dnn_clf = tf.keras.wrappers.scikit_learn.KerasClassifier(
        #                         dnn_model_ensemble,
        #                         epochs=10,
        #                         verbose=False)
        # dnn_clf._estimator_type = "classifier"
        # dnn_clf.fit(dataset.X_train, dataset.y_train)
        # perm = PermutationImportance(dnn_clf).fit(dataset.X_test, dataset.y_test)
        # importances = explain_weights_df(perm, feature_names=feature_names,targets=['unstable','stable'])
        # print(importances)

        # df = pd.DataFrame(importances)
        # filepath = 'permutationimportance_dnn.xlsx'
        # df.to_excel(filepath, index=False)

        

        if(reduced == 1 and model.name == 'DNN'):
                # Reduce features
            print("DNN REDUCED")
            indexes = []
            for m in range(1, dataset.X_test.shape[1]):
                indexes.append(
                    nnxai.get_top_important_features_SHAP(shap, m))

            scoreins = []

            for index in indexes:
                ds = dataset.test
                test_ds = ds.copy()

                test_ds.columns = range(test_ds.shape[1])
                important_columns = index
                unimportant_columns = [
                    item for item in test_ds if item not in important_columns]
                print()

                # for col in important_columns:
                #       test_ds[col].values[:] = 0
                print("Evaluating with n: ", len(index))
                for col in unimportant_columns[:-1]:
                    test_ds[col].values[:] = 0

                test_ds.columns = ['F_%s' % f for f in range(test_ds.shape[1])]
                test_ds.columns = [*test_ds.columns[:-1], 'stable-unstable']

                test_ds = df_to_dataset(test_ds, shuffle=False,
                                        batch_size=config_data['batch_size'])

                accuracy = model.evaluate(test_ds, return_dict=True)
                scoreins.append(accuracy['accuracy'])
          
            
            

        elif(reduced == 1 and model.name == 'RNN'):
            print('REDUCED RNN')

            indexes = []

            for m in range(1, dataset.X_test.shape[1]):
                indexes.append(
                    nnxai.get_top_important_features_SHAP(shap, m))
            scoreins = []
            
            for index in indexes:
                X_train = np.array(dataset.X_train.values)
                X_test = np.array(dataset.X_test.values)
                print("Evaluating with n: ", len(index))
                test_ds = dataset.X_test
                test_ds.columns = range(test_ds.shape[1])
                important_columns = index
                unimportant_columns = [
                    item for item in test_ds if item not in important_columns]

                #X_test[:, [important_columns]] = 0
                X_test[:, [unimportant_columns]] = 0

                testX = np.reshape(X_test, (dataset.X_test.shape[0], config_data['n_steps'],
                                            dataset.X_test.shape[1]))

                accuracy = model.evaluate(testX,dataset.y_test, return_dict=True)
                scoreins.append(accuracy['accuracy'])

    if(reduced==1):
        scores.append(scoreins)
        result = np.array(scores)
        result = result.mean(axis=0)
        df = pd.DataFrame(result)
        filepath = 'rnn_score_lime.xlsx'
        df.to_excel(filepath, index=False)
        return model



def predict_with_ensemble(X_train,
                     y_train,
                     X_test,
                     y_test,
                     data_set,
                     reduced,
                     grid,
                     config_data,
                     cross_validation=False):
    print('predicting with ensemble')
    scores = []
    if reduced ==0:
        n=1
    else: 
        n=50
    for i in range(n):
        svm_classifier = SVC(kernel='rbf', probability=True)
        
        mlp = MLPClassifier()
        
        clf = tree.DecisionTreeClassifier()

        
        dnn_clf = tf.keras.wrappers.scikit_learn.KerasClassifier(
                                dnn_model_ensemble,
                                epochs=10,
                                verbose=False)
        dnn_clf._estimator_type = "classifier"

        
        
        estimators = [('svm',svm_classifier),('MLP',mlp),('dt',clf),('dnn',dnn_clf)]

        
        

        ensb = VotingClassifier(estimators,voting='soft',flatten_transform=True)
        start = timer()
        ensb.fit(X_train, y_train)
        end = timer()
        trainingtime = end-start
        start = timer()
        preds =  ensb.predict(X_test)
        end = timer()
        predictiontime = end-start
        print(classification_report(y_test, preds))
        print(confusion_matrix(y_test, preds))
        print(trainingtime,predictiontime)

        


        ensbxai = SmartGridExplainer(X_train=X_train, X_test=X_test, model=ensb, grid=grid)
        ensbxai.generateLocalLimeExplanation(2)
        spobj = ensbxai.generateGlobalLimeExplanations()
        shap = ensbxai.generateGlobalShapExplanations()

        

        

        if(reduced == 1):

            indexes = []
            for m in range(1, X_test.shape[1]):
                indexes.append(
                    ensbxai.get_top_important_features_SHAP(shap, m))

            scoreins = []
            print()
            if(reduced == 1):
                for index in indexes:
                    print('Number of n: ', len(index))
                    predictions = ensbxai.predict_with_modified_features(
                        index, False)
                    accuracy = classification_report(
                        y_test, predictions, output_dict=True)
                    scoreins.append(accuracy['accuracy'])
                    print(accuracy['accuracy'])
            scores.append(scoreins)
    if(reduced==1):
        print(i)
        accuracies = np.array(scores)
        print()
        result = accuracies.mean(axis=0)
        df = pd.DataFrame(result)
        filepath = 'ensb_score.xlsx'
        df.to_excel(filepath, index=False)

    perm = PermutationImportance(ensb).fit(X_test, y_test)
    importances = explain_weights_df(perm, feature_names=feature_names,targets=['unstable','stable'])
    df = pd.DataFrame(importances)
    filepath = 'permutationimportance_ensb.xlsx'
    df.to_excel(filepath, index=False)

    print('ENSB END')
    return classification_report(y_test, preds), confusion_matrix(y_test, preds)

    
def dnn_model_ensemble():
    tf.keras.backend.clear_session()
    model= tf.keras.Sequential([
                       tf.keras.layers.Dense(10,activation='relu',input_shape=[77]),
                       tf.keras.layers.Dense(1,activation='sigmoid')
    ])
    model.compile(
                optimizer='Adam',
                loss='binary_crossentropy',
                metrics=['accuracy'])
    return model


def rnn_model_ensemble():
    model = tf.keras.Sequential([
            tf.keras.layers.LSTM(150, activation='tanh',
                                 input_shape=(1,77),
                                 return_sequences=True),
            tf.keras.layers.LSTM(150, activation='tanh',
                                 return_sequences=True),
            tf.keras.layers.LSTM(150, activation='tanh',
                                 return_sequences=False),
            tf.keras.layers.Dense(units=1, activation='sigmoid')
    ])
    model.compile(
            optimizer='Adam',
            loss='binary_crossentropy',
            metrics=['accuracy'])
    return model






class ModelCallbacks(tf.keras.callbacks.Callback):
    def __init__(self):
        def on_epoch_begin(self, epoch, logs={}):
            self.starttime = time()

        def on_epoch_end(self, epoch, logs={}):
            self.logs.append(time() - self.starttime)

        def on_epoch_end(self, logs=None):
            if logs.get('acc') >= 0.8:
                self.model.stop_training = True
                print("Required accuracy is reached 99.00%, "
                      "cancelling training!")
