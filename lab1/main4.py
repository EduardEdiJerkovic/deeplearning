import numpy as np
import matplotlib.pyplot as plt

from pt_logreg import train, PTLogreg, evaluation
from data import sample_gauss_2d, class_to_onehot, graph_data, graph_surface

def logreg_decfun(ptlr):
  def classify(X):
    return np.argmax(evaluation(ptlr, X), axis=1)
  return classify

if __name__ == "__main__":
  # inicijaliziraj generatore slučajnih brojeva
  np.random.seed(100)

  # instanciraj podatke X i labele Yoh_
  X, Y = sample_gauss_2d(2, 10)

  #Yoh_ = class_to_onehot(Y)

  #X = torch.tensor(X)
  #Yoh_ = torch.tensor(Yoh_)

  # definiraj model:
  ptlr = PTLogreg(X.shape[1], max(Y) + 1)

  # nauči parametre (X i Yoh_ moraju biti tipa torch.Tensor):
  train(ptlr, X, Y, param_niter=1e5, param_delta=0.001)

  # dohvati vjerojatnosti na skupu za učenje
  probs = evaluation(ptlr, X)

  # ispiši performansu (preciznost i odziv po razredima)
  rect = (np.min(X, axis=0), np.max(X, axis=0))
  graph_surface(logreg_decfun(ptlr), rect, offset=0.5)
  graph_data(X, Y, np.argmax(evaluation(ptlr, X), axis=1), special=[])


  plt.show()

  # iscrtaj rezultate, decizijsku plohu