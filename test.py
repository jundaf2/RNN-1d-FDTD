import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import fdtd_1d as fdtd
from model import RNN

domainLength = fdtd.Parameters.DOMAINLENGTH
dim = int(np.ceil(domainLength / fdtd.Parameters.DELTA_Z))
steps = np.linspace(0, domainLength, dim).flatten()
ez_rnn = np.zeros([1, dim])
ez_rnn = ez_rnn.transpose()

fig = plt.figure(1, figsize=(12, 5))
plt.ion()  # continuously plot

model = RNN()
model.load_state_dict(torch.load('mymodule.pt'))
model.eval()

Ez = []
temp2 = ez_rnn
Ez.append(temp2)
temp1 = ez_rnn
Ez.append(temp1)
h_state = torch.zeros(1, 1, dim)
for step in range(fdtd.Parameters.STEP_NUM-1):

    ez_rnn = np.append(fdtd.Parameters.BVL[step], Ez[-1][1:])
    x = torch.from_numpy(ez_rnn[np.newaxis, :, np.newaxis]).float()  # shape (batch, time_step, input_size)
    ez_rnn, h_state = model(x, h_state)  # rnn output
    # print(type(ez_rnn))
    ez_rnn = ez_rnn.detach().numpy().flatten()
    h_state = h_state.data
    Ez.append(ez_rnn)
    # print(type(ez_rnn))
    # print(ez_rnn)
    # plotting
    fig.clear()
    plt.plot(steps, ez_rnn, 'r-')
    plt.draw()
    plt.pause(0.0005)

plt.ioff()
plt.show()