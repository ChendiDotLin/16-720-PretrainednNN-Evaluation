import numpy as np
import scipy.io as io

resnet_data = np.load('resnet50.npz')
alexnet_data = np.load('alexnet.npz')
squeezenet_data = np.load('squeezenet.npz')
vggnet_data = np.load('vgg19_bn.npz')
densenet_data = np.load('densenet161.npz')

io.savemat('resnet.mat', mdict=resnet_data)
io.savemat('alexnet.mat', mdict=alexnet_data)
io.savemat('squeezenet.mat', mdict=squeezenet_data)
io.savemat('vggnet.mat', mdict=vggnet_data)
io.savemat('densenet.mat', mdict=densenet_data)
