
import torch
from torch import nn
import matplotlib.pyplot as plt
import fdtd_1d as fdtd
from model import RNN
import numpy as np


domainLength = fdtd.Parameters.DOMAINLENGTH
dim = int(np.ceil(domainLength / fdtd.Parameters.DELTA_Z))
# Hyper Parameters
TIME_STEP = dim  # rnn time step
INPUT_SIZE = 1  # rnn input size
LR = 0.02  # learning rate

# fdtd_rnn setting
ez_rnn = np.zeros([1, dim])
ez_rnn = ez_rnn.transpose()
# fdtd setting
Ez = []
ez = ez_rnn
temp2 = ez
Ez.append(temp2)
temp1 = ez
Ez.append(temp1)

# range of domain
steps = np.linspace(0, domainLength, dim).flatten()

rnn = RNN()
print(rnn)

optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)  # optimize all cnn parameters
loss_func = nn.MSELoss()
h_state = None  # for initial hidden state

fig = plt.figure(1, figsize=(12, 5))
plt.ion()  # continuously plot

for step in range(fdtd.Parameters.STEP_NUM-1):

    ez = fdtd.fdtd_1d(step, Ez)
    Ez.append(ez)
    input_bvl = np.append(fdtd.Parameters.BVL[step], Ez[-1][1:])
    x = torch.from_numpy(input_bvl[np.newaxis, :, np.newaxis]).float()  # shape (batch, time_step, input_size)
    y = torch.from_numpy(ez[np.newaxis, :]).float()

    prediction, h_state = rnn(x, h_state)  # rnn output
    # !! next step is important !!
    h_state = h_state.data  # repack the hidden state, break the connection from last iteration

    loss = loss_func(prediction, y)  # calculate loss
    print('epoch {} loss {}'.format(step, loss))
    optimizer.zero_grad()  # clear gradients for this training step
    loss.backward()  # backpropagation, compute gradients
    optimizer.step()  # apply gradients

    # plotting
    fig.clear()
    plt.plot(steps, ez.flatten(), 'r-')
    # print(prediction.data.numpy().flatten())
    plt.plot(steps, prediction.data.numpy().flatten(), 'b-')
    plt.pause(0.0005)

plt.ioff()
plt.close(1)

torch.save(rnn.state_dict(), 'mymodule.pt')


