import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import Callback
from sklearn import tree
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import mean_squared_error as mse
from tensorflow import feature_column
from time import time


def predict_with_svm(X_train,
                     y_train,
                     X_test,
                     y_test,
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

    svm_classifier = SVC(kernel='rbf')
    svm_classifier.fit(X_train, y_train)
    y_predict_smv = svm_classifier.predict(X_test)
    print(classification_report(y_test, y_predict_smv))
    print(confusion_matrix(y_test, y_predict_smv))
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
        print("Test RMSE Shuffle: %.2f" % np.sqrt(mse(y_test, y_hat_shuffle)), "\n")

    return classification_report(y_test, y_predict_smv), confusion_matrix(y_test, y_predict_smv)


def predict_with_mlp(X_train,
                     y_train,
                     X_test,
                     y_test,
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
    mlp = MLPClassifier()
    mlp.fit(X_train, y_train)
    y_predict_mlp = mlp.predict(X_test)
    print(classification_report(y_test, y_predict_mlp))
    print(confusion_matrix(y_test, y_predict_mlp))
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
        print("Test RMSE Shuffle: %.2f" % np.sqrt(mse(y_test, y_hat_shuffle)), "\n")
    return classification_report(y_test, y_predict_mlp), confusion_matrix(y_test, y_predict_mlp)


def predict_with_clf(X_train,
                     y_train,
                     X_test,
                     y_test,
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
    clf = tree.DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    y_predict_clf = clf.predict(X_test)
    print(classification_report(y_test, y_predict_clf))
    print(confusion_matrix(y_test, y_predict_clf))
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
        print("Test RMSE Shuffle: %.2f" % np.sqrt(mse(y_test, y_hat_shuffle)), "\n")
    return classification_report(y_test, y_predict_clf), confusion_matrix(y_test, y_predict_clf)


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


def get_feature_layer():
    """this function creates a feature
    layer to be used later in Keras model.
    :returns Keras Keras feature layer
    :raises does not raise any
    """
    feature_columns = []
    columns = ['F_%s' % f for f in range(16)]
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
    if algorithm == 'dnn':
        return tf.keras.Sequential([
            get_feature_layer(),
            tf.keras.layers.Dense(10, activation=tf.nn.relu),
            tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)
        ])
    elif algorithm == 'rnn':
        return tf.keras.Sequential([
            tf.keras.layers.LSTM(150, activation='tanh',
                                 input_shape=(config_data['n_steps'],
                                              train_ds.shape[1] - 1),
                                 return_sequences=True),
            tf.keras.layers.LSTM(150, activation='tanh', return_sequences=True),
            tf.keras.layers.LSTM(150, activation='tanh', return_sequences=True),
            tf.keras.layers.Dense(units=1, activation='sigmoid')
        ])


def predict_with_nn(dataset, algorithm, config_data):
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
    tf.keras.backend.clear_session()
    train_ds = df_to_dataset(dataset.train, shuffle=False,
                             batch_size=config_data['batch_size'])
    val_ds = df_to_dataset(dataset.val,
                           shuffle=False, batch_size=config_data['batch_size'])
    model = construct_model(config_data, algorithm, dataset.train)
    model.compile(optimizer=config_data['optimizer'], metrics=['accuracy'],
                  loss=tf.losses.binary_crossentropy),
    # callbacks_list = [StopTraining()]
    if algorithm == 'dnn':
        model.fit(train_ds, validation_data=val_ds, epochs=config_data['epochs'])
    elif algorithm == 'rnn':
        X_train = np.array(dataset.X_train.values)
        X_test = np.array(dataset.X_test.values)
        trainX = np.reshape(X_train, (dataset.X_train.shape[0], config_data['n_steps'],
                                      dataset.X_train.shape[1]))
        testX = np.reshape(X_test, (dataset.X_test.shape[0], config_data['n_steps'],
                                    dataset.X_test.shape[1]))
        model.fit(trainX, dataset.y_train, epochs=config_data['epochs'])
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
