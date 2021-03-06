import torch.nn as nn
import torch

class PTLogreg(nn.Module):
    def __init__(self, D, C):
        """Arguments:
           - D: dimensions of each datapoint
           - C: number of classes
        """
        super(PTLogreg, self).__init__()
        self.W = torch.nn.Parameter(torch.zeros(D, C))
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
    loss_func = nn.CrossEntropyLoss()
    optim = torch.optim.SGD(model.parameters(), lr=param_delta, weight_decay=1e-4)

    model.double()

    for i in range(int(param_niter)):
        optim.zero_grad()
        out = model(torch.tensor(X))
        loss = loss_func(out, torch.tensor(Yoh_).to(dtype=torch.int64))
        loss.backward()
        optim.step()

        if i % 100 == 0:
            print("Iteration:", i, "Loss:", loss.item())

    # inicijalizacija optimizatora
    # ...

    # petlja učenja
    # ispisujte gubitak tijekom učenja
    # ...


def evaluation(model, X):
    """Arguments:
       - model: type: PTLogreg
       - X: actual datapoints [NxD], type: np.array
       Returns: predicted class probabilites [NxC], type: np.array
    """
    return torch.Tensor.numpy(model(torch.tensor(X)).detach())
    # ulaz je potrebno pretvoriti u torch.Tensor
    # izlaze je potrebno pretvoriti u numpy.array
    # koristite torch.Tensor.detach() i torch.Tensor.numpy()