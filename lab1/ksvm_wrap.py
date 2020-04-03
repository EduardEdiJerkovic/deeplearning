import torch.nn as nn
import numpy as np
import torch
from sklearn import svm

import data


class KSVM(nn.Module):
    def __init__(self, X, Y_, param_svm_c=1, param_svm_gamma='auto'):
        self.svm = svm.SVC(C=param_svm_c, kernel='rbf', gamma=param_svm_gamma)
        self.svm.fit(X, Y_)


    def predict(self, X):
        return self.svm.predict(X)

    def get_scores(self, X):
        self.svm.get
    # Vraća klasifikacijske mjere
    # (engl. classification scores) podataka X;
    # ovo će vam trebati za računanje prosječne preciznosti.

    def support(self):
        return self.svm.support_
    # Indeksi podataka koji su odabrani za potporne vektore
