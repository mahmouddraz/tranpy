Models Module
=============

The models module provides machine learning classifiers for power system stability prediction.

.. currentmodule:: tranpy.models

Base Classes
------------

.. autoclass:: BaseClassifier
   :members:
   :undoc-members:
   :show-inheritance:

Classical Models
----------------

.. autoclass:: RandomForestClassifier
   :members:
   :show-inheritance:

.. autoclass:: SVMClassifier
   :members:
   :show-inheritance:

.. autoclass:: DecisionTreeClassifier
   :members:
   :show-inheritance:

.. autoclass:: AdaBoostClassifier
   :members:
   :show-inheritance:

.. autoclass:: GradientBoostingClassifier
   :members:
   :show-inheritance:

.. autoclass:: ExtraTreesClassifier
   :members:
   :show-inheritance:

Naive Bayes Models
~~~~~~~~~~~~~~~~~~

.. autoclass:: GaussianNB
   :members:
   :show-inheritance:

Nearest Neighbors
~~~~~~~~~~~~~~~~~

.. autoclass:: KNeighborsClassifier
   :members:
   :show-inheritance:

Gaussian Process
~~~~~~~~~~~~~~~~

.. autoclass:: GaussianProcessClassifier
   :members:
   :show-inheritance:

Linear Models
~~~~~~~~~~~~~

.. autoclass:: LogisticRegression
   :members:
   :show-inheritance:

.. autoclass:: RidgeClassifier
   :members:
   :show-inheritance:

.. autoclass:: SGDClassifier
   :members:
   :show-inheritance:

Discriminant Analysis
~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: LinearDiscriminantAnalysis
   :members:
   :show-inheritance:

.. autoclass:: QuadraticDiscriminantAnalysis
   :members:
   :show-inheritance:

Boosting Models
~~~~~~~~~~~~~~~

.. autoclass:: XGBClassifier
   :members:
   :show-inheritance:

.. autoclass:: LGBMClassifier
   :members:
   :show-inheritance:

Neural Networks
---------------

.. autoclass:: DNNClassifier
   :members:
   :show-inheritance:

.. autoclass:: RNNClassifier
   :members:
   :show-inheritance:

Ensemble Models
---------------

.. autoclass:: EnsembleClassifier
   :members:
   :show-inheritance:

Pretrained Models
-----------------

.. autofunction:: load_pretrained

.. autofunction:: list_pretrained_models

.. autofunction:: save_model

.. autofunction:: load_model
