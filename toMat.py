import numpy as np
import scipy.io as io

resnet_data = np.load('resnet50.npz')
alexnet_data = np.load('alexnet.npz')
squeezenet_data = np.load('squeezenet.npz')

io.savemat('resnet.mat', mdict=resnet_data)
io.savemat('alexnet.mat', mdict=alexnet_data)
io.savemat('squeezenet.mat', mdict=squeezenet_data)

