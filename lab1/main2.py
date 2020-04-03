import numpy as np
import matplotlib.pyplot as plt

from data import sample_gmm_2d, graph_surface, graph_data, myDummyDecision, eval_perf_multi
from fcann2 import fcann2_train, fcann2_classify


def fcann2_decfun(w, b):
    def classify(X):
        return np.argmax(fcann2_classify(X, w, b), axis=1)
    return classify


if __name__ == "__main__":
    X, Y_ = sample_gmm_2d(6, 2, 10)
    meanX = X.mean()
    stdX = X.std()
    X = (X - meanX) / stdX
    w, b = fcann2_train(X, Y_, param_lambda=1e-5, param_delta=1e-5) #param_niter

    # graph the decision surface
    rect = (np.min(X, axis=0), np.max(X, axis=0))
    graph_surface(fcann2_decfun(w, b), rect, offset=0.5)


    #print(fcann2_classify(X, w, b))
    # graph the data points
    graph_data(X, Y_, np.argmax(fcann2_classify(X, w, b), axis=1), special=[])

    # graph_data(X, Y_, list(map(lambda x: np.argmax(x), fcann2_classify(X, w, b))), special=[])

    plt.show()

    # Finish

