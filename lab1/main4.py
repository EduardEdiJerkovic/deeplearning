import numpy as np
from pt_logreg import train

if __name__ == "__main__":
  # inicijaliziraj generatore slučajnih brojeva
  np.random.seed(100)

  # instanciraj podatke X i labele Yoh_

  # definiraj model:
  ptlr = PTLogreg(X.shape[1], Yoh_.shape[1])

  # nauči parametre (X i Yoh_ moraju biti tipa torch.Tensor):
  train(ptlr, X, Yoh_, 1000, 0.5)

  # dohvati vjerojatnosti na skupu za učenje
  probs = eval(ptlr, X)

  # ispiši performansu (preciznost i odziv po razredima)

  # iscrtaj rezultate, decizijsku plohu