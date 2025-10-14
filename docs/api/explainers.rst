Explainers Module
=================

The explainers module provides model interpretability and explainability tools.

.. currentmodule:: tranpy.explainers

Base Classes
------------

.. autoclass:: BaseExplainer
   :members:
   :undoc-members:
   :show-inheritance:

SHAP Explainer
--------------

.. autoclass:: SHAPExplainer
   :members:
   :show-inheritance:

   **Example Usage:**

   .. code-block:: python

      from tranpy.explainers import SHAPExplainer

      explainer = SHAPExplainer(model, X_train, X_test)
      shap_values = explainer.explain_global()
      explainer.plot_summary(shap_values)

LIME Explainer
--------------

.. autoclass:: LIMEExplainer
   :members:
   :show-inheritance:

   **Example Usage:**

   .. code-block:: python

      from tranpy.explainers import LIMEExplainer

      explainer = LIMEExplainer(model, X_train, X_test)
      explanation = explainer.explain_instance(X_test[0])

DALEX Explainer
---------------

.. autoclass:: DALEXExplainer
   :members:
   :show-inheritance:

   **Example Usage:**

   .. code-block:: python

      from tranpy.explainers import DALEXExplainer

      explainer = DALEXExplainer(model, X_train, y_train, X_test)
      explainer.model_parts()
      explainer.plot_feature_importance()
