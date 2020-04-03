import torch
import torch.nn as nn
import torch.optim as optim


## Definicija računskog grafa
# podaci i parametri, inicijalizacija parametara
a = torch.randn(1, requires_grad=True)
b = torch.randn(1, requires_grad=True)

X = torch.tensor([1, 2, 3, 4])
Y = torch.tensor([3, 5, 7, 10])

#X = torch.tensor([1, 2])
#Y = torch.tensor([3, 5])

# optimizacijski postupak: gradijentni spust
optimizer = optim.SGD([a, b], lr=0.1/(len(X) * 5))

for i in range(100):
    # afin regresijski model
    Y_ = a*X + b
    diff = (Y-Y_)

    # kvadratni gubitak
    loss = torch.sum(diff**2)

    # računanje gradijenata
    loss.backward()

    # korak optimizacije
    optimizer.step()

    print(f'a_grad:{a.grad}, b_grad:{b.grad}, a_calc:{sum(-2 * (Y-a*X - b) * X)}, b_calc:{sum(-2 * (Y-a*X - b))}')

    # Postavljanje gradijenata na nulu
    optimizer.zero_grad()

    print(f'step: {i}, loss:{loss}, Y_:{Y_}, a:{a}, b {b}')