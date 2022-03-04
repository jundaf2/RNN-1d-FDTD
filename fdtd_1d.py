import numpy as np
import matplotlib.pyplot as plt
import time
import matplotlib.animation as animation
from matplotlib import style


class Parameters:

    DOMAINLENGTH: float = 10
    FREQUENCY: float = 10e1
    DELTA_Z: float = 0.1
    DELTA_T: float = 0.01

    PERMEABILITY: float = 1
    PERMITIVITY: float = 1
    SPEED_LIGHT: float = 299_792_458.0  # [m/s] speed of light
    VACUUM_PERMEABILITY: float = 4e-7 * np.pi  # vacuum permeability
    VACUUM_PERMITIVITY: float = 1.0 / (
            VACUUM_PERMEABILITY * SPEED_LIGHT ** 2
    )  # vacuum permittivity

    STEP_NUM = 2000
    # Dirichlet BV
    BVL = np.zeros([STEP_NUM, 1])
    for i in range(1, STEP_NUM):
        BVL[i, 0] = np.sin(2*np.pi*i*DELTA_T+np.pi/2)
    BVR = np.zeros([STEP_NUM, 1]) + 0

'''
def main():
'''


def fdtd_1d(time_step, Ez):

    # add the boundary values
    temp2 = np.concatenate((np.zeros([1, 1]), Parameters.BVL[time_step].reshape([1, 1]), Ez[-2][1:], np.zeros([1, 1])))
    # set the first two frames

    temp1 = np.concatenate((np.zeros([1, 1]), Parameters.BVL[time_step].reshape([1, 1]), Ez[-1][1:], np.zeros([1, 1])))
    ez = 2 * temp1[1:-1] - temp2[1:-1] + (Parameters.DELTA_T ** 2) / \
         (Parameters.PERMEABILITY * Parameters.PERMITIVITY * Parameters.DELTA_Z ** 2) * \
         (temp1[2:] - 2 * temp1[1:-1] + temp1[0:-2])

    return ez


def test():
    Ez = []
    domainLength = Parameters.DOMAINLENGTH
    dim = int(np.ceil(domainLength / Parameters.DELTA_Z))
    ez = np.zeros([1, dim])
    ez[1, 1] = Parameters.BVL[1, 0]
    ez = ez.transpose()
    # add the boundary values
    ez[1, 1] = Parameters.BVL[2, 0]
    temp2 = ez
    Ez.append(temp2)
    # set the first two frames
    temp1 = ez
    Ez.append(temp1)

    fig = plt.figure(1, figsize=(12, 5))
    plt.ion()
    for i in range(2, Parameters.STEP_NUM):
        ez = fdtd_1d(i, Ez)
        Ez.append(ez)
        fig.clear()
        plt.plot(np.linspace(0, domainLength, dim).flatten(), ez.T.flatten())
        plt.show()
        plt.pause(0.005)
    plt.ioff()

'''
def main():
    style.use('fivethirtyeight')
    fig = plt.figure()
    ax1 = fig.add_subplot(1,1,1)
    plt.ion()

    domainLength = Parameters.DOMAINLENGTH
    dim = int(np.ceil(domainLength / Parameters.DELTA_Z))
    ez = np.zeros([1, dim])
    ez = ez.transpose()
    axis = np.linspace(0, domainLength, dim)

    # add the boundary values
    temp2 = np.concatenate((Parameters.BVL[0].reshape([1, 1]), ez))
    temp2 = np.concatenate((temp2, Parameters.BVR[0].reshape([1, 1])))
    # set the first two frames

    for i in range(1, 9999):
        temp1 = np.concatenate((Parameters.BVL[i].reshape([1, 1]), ez))
        temp1 = np.concatenate((temp1, Parameters.BVR[i].reshape([1, 1])))
        ez = 2 * temp1[1:-1] - temp2[1:-1] + (Parameters.DELTA_T ** 2) / \
             (Parameters.PERMEABILITY * Parameters.PERMITIVITY * Parameters.DELTA_Z ** 2) * \
             (temp1[2:] - 2 * temp1[1:-1] + temp1[0:-2])
        temp2 = temp1
        ax1.clear()
        ax1.plot(axis, ez)
        plt.pause(0.0005)
        
        
if __name__ == "__main__":
    main()
'''


