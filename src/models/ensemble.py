# src/models/ensemble.py
import numpy as np
from sklearn.linear_model import LogisticRegression

class StackingEnsemble:
    def __init__(self):
        self.meta_clf = LogisticRegression(max_iter=1000)

    def fit_meta(self, preds_list, y):
        """
        preds_list: list of (N, num_classes) arrays from base models (on holdout val set)
        y: ground truth labels
        """
        X_meta = np.concatenate(preds_list, axis=1)
        self.meta_clf.fit(X_meta, y)

    def predict(self, preds_list):
        X_meta = np.concatenate(preds_list, axis=1)
        return self.meta_clf.predict(X_meta)

    def predict_proba(self, preds_list):
        X_meta = np.concatenate(preds_list, axis=1)
        return self.meta_clf.predict_proba(X_meta)
