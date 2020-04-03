import torch.nn as nn

class PTLogreg(nn.Module):
    def __init__(self, D, C):
        """Arguments:
           - D: dimensions of each datapoint
           - C: number of classes
        """
        super(PTLogreg, self).init()
        self.W = torch.nn.Parameter(torch.zeros(C, D))
        self.b = torch.nn.Parameter(torch.zeros(C))

        # inicijalizirati parametre (koristite nn.Parameter):
        # imena mogu biti self.W, self.b
        # ...

    def forward(self, X):
        return nn.functional.softmax(torch.mm(X, self.W) + self.b)

        # unaprijedni prolaz modela: izračunati vjerojatnosti
        #   koristiti: torch.mm, torch.softmax
        # ...

    def get_loss(self, X, Yoh_):
        Y = self.forward(X)
        return - torch.mean(torch.sum(Yoh_ * torch.log(Y) + (1 - Yoh_) * torch.log(1 - Y)))
        # formulacija gubitka
        #   koristiti: torch.log, torch.mean, torch.sum
        # ...


def train(model, X, Yoh_, param_niter, param_delta):
    """Arguments:
       - X: model inputs [NxD], type: torch.Tensor
       - Yoh_: ground truth [NxC], type: torch.Tensor
       - param_niter: number of training iterations
       - param_delta: learning rate
    """
    func = nn.CrossEntropyLoss()
    optim = torch.optim.SGD(model.parameters(), lr=param_delta)

    for i in range(int(param_niter)):
        optim.zero_grad()
        out = model(X)
        loss = func(Yah_)


    # inicijalizacija optimizatora
    # ...

    # petlja učenja
    # ispisujte gubitak tijekom učenja
    # ...

def eval(model, X):
    """Arguments:
       - model: type: PTLogreg
       - X: actual datapoints [NxD], type: np.array
       Returns: predicted class probabilites [NxC], type: np.array
    """
    # ulaz je potrebno pretvoriti u torch.Tensor
    # izlaze je potrebno pretvoriti u numpy.array
    # koristite torch.Tensor.detach() i torch.Tensor.numpy()