import numpy as np
import matplotlib.pyplot as plt
import torch

import data
import pt_deep

def deep_decfun(ptlr):
  def classify(X):
    return np.argmax(pt_deep.evaluation(ptlr, X), axis=1)
  return classify


def test(X, Y, dims, func, param_niter=1e5, param_delta=0.001, plot=True):
    ptdeep = pt_deep.PTDeep(dims, func)

    # nauči parametre (X i Yoh_ moraju biti tipa torch.Tensor):
    pt_deep.train(ptdeep, X, Y, param_niter=param_niter, param_delta=param_delta)

    # dohvati vjerojatnosti na skupu za učenje
    probs = pt_deep.evaluation(ptdeep, X)

    # ispiši performansu (preciznost i odziv po razredima)
    if plot:
        rect = (np.min(X, axis=0), np.max(X, axis=0))
        data.graph_surface(deep_decfun(ptdeep), rect, offset=0.5)
        data.graph_data(X, Y, np.argmax(pt_deep.evaluation(ptdeep, X), axis=1), special=[])

        plt.show()

    return ptdeep, probs


def apr_print(values):
    print("Accuracy:", values[0], "Recall:", values[1], "Precision:", values[2])


if __name__ == "__main__":
    # inicijaliziraj generatore slučajnih brojeva
    np.random.seed(100)

    # instanciraj podatke X i labele Yoh_

    C = 2
    X, Y = data.sample_gauss_2d(C, 10)

    # Same as task 4
    test(X, Y, dims=[2, C], func=torch.relu, param_niter=1e5, param_delta=0.001)

    # Example of dimensions
    ptdeep1, result = test(X, Y, dims=[2, 5, C], func=torch.relu, param_niter=1, param_delta=0.001, plot=False)
    ptdeep1.count_params()

    # Extra tasks.
    X1, Y1 = data.sample_gmm_2d(4, 2, 40)
    X2, Y2 = data.sample_gmm_2d(6, 2, 10)

    m11, r11 = test(X1, Y1, dims=[2, 2], func=torch.relu)
    m12, r12 = test(X1, Y1, dims=[2, 10, 2], func=torch.relu)
    m13, r13 = test(X1, Y1, dims=[2, 10, 10, 2], func=torch.relu)

    n21, r21 = test(X2, Y2, dims=[2, 2], func=torch.relu)
    n22, r22 = test(X2, Y2, dims=[2, 10, 2], func=torch.relu)
    n23, r23 = test(X2, Y2, dims=[2, 10, 10, 2], func=torch.relu)

    apr_print(data.eval_perf_binary(np.argmax(r11, axis=1), Y1))
    apr_print(data.eval_perf_binary(np.argmax(r12, axis=1), Y1))
    apr_print(data.eval_perf_binary(np.argmax(r13, axis=1), Y1))

    apr_print(data.eval_perf_binary(np.argmax(r21, axis=1), Y2))
    apr_print(data.eval_perf_binary(np.argmax(r22, axis=1), Y2))
    apr_print(data.eval_perf_binary(np.argmax(r23, axis=1), Y2))


