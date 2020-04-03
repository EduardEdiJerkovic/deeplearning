import numpy as np


def fcann2_train(X, Y, nhidden=5, param_niter=1e5, param_delta=0.05, param_lambda=1e-3):

    w1 = np.random.randn(X.shape[1], nhidden) * 0.01
    b1 = np.random.randn(1, nhidden) * 0.01
    w2 = np.random.randn(nhidden, max(Y) + 1) * 0.01
    b2 = np.random.randn(1, max(Y) + 1) * 0.01

    Y_ = np.stack([1 - Y, Y], axis=1)

    for i in range(int(param_niter)):
        S1 = X @ w1 + b1
        H1 = relu(S1)
        S2 = H1 @ w2 + b2
        P = softmax(S2)

        Gs2 = P - Y_

        if i % 100 == 0:
            print(-np.sum(np.log(P[Y])) / len(X))

        gradW2 = Gs2.T @ H1
        gradb2 = np.sum(Gs2, axis=0, keepdims=True)
        #print(Gs2.shape, w2.T.shape)
        Gh1 = Gs2 @ w2.T
        Gs1 = Gh1 * (S1 > 0).astype(np.float32)

        gradW1 = Gs1.T @ X
        #print(gradW1.shape, w1.shape)
        gradb1 = np.sum(Gs1, axis=0, keepdims=True)

        #print(w1.shape, b1.shape, w2.shape, b2.shape)
        w1 = w1 - gradW1.T * param_delta - param_delta * param_lambda * w1
        b1 = b1 - param_delta * gradb1

        w2 = w2 - gradW2.T * param_delta - param_delta * param_lambda * w2
        b2 = b2 - param_delta * gradb2

        #print(w1.shape, b1.shape, w2.shape, b2.shape)

    return [w1, w2], [b1, b2]


def fcann2_classify(X, W, b):
    S1 = X @ W[0] + b[0]
    H1 = relu(S1)
    S2 = H1 @ W[1] + b[1]
    P = softmax(S2)

    return P


def relu(Y):
    return np.maximum(0, Y)


def softmax(Y):
    return np.exp(Y)/np.sum(np.exp(Y), axis=1, keepdims=True)