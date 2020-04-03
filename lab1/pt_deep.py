import torch.nn as nn
import torch


class PTDeep(nn.Module):
    def __init__(self, D, F):
        """Arguments:
           - D: dimensions of each datapoint
           - C: number of classes
        """
        super(PTDeep, self).__init__()
        learn_args = []
        bias_args = []
        for i in range(len(D) - 1):
            learn_args.append(nn.Parameter(nn.Parameter(torch.zeros(D[i], D[i+1]))))
            bias_args.append(nn.Parameter(nn.Parameter(torch.zeros(D[i+1]))))

        self.W = nn.ParameterList(learn_args)
        self.b = nn.ParameterList(bias_args)
        self.F = F

        # inicijalizirati parametre (koristite nn.Parameter):
        # imena mogu biti self.W, self.b
        # ...

    def forward(self, X):
        data = X
        for i in range(len(self.W)-1):
            data = self.F(torch.mm(data, self.W[i]) + self.b[i])
        return nn.functional.softmax(torch.mm(data, self.W[-1]) + self.b[-1])

        # unaprijedni prolaz modela: izračunati vjerojatnosti
        #   koristiti: torch.mm, torch.softmax
        # ...

    def get_loss(self, X, Yoh_):
        Y = self.forward(X)
        return - torch.mean(torch.sum(Yoh_ * torch.log(Y) + (1 - Yoh_) * torch.log(1 - Y)))
        # formulacija gubitka
        #   koristiti: torch.log, torch.mean, torch.sum
        # ...

    def count_params(self):
        count = 0
        for i in range(len(self.W)):
            print("W" + str(i) + ":", str(self.W[i].shape))
            print("W" + str(i) + ":", str(self.b[i].shape))

            count += self.W[i].shape[0] * self.W[i].shape[1] + len(self.b[i])

        print("Number of all parameters:", count)


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

        if i % 1000 == 0:
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

