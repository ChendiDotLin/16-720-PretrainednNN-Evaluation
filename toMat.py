import numpy as np
import scipy.io as io

resnet_data = np.load('resnet50.npz')
io.savemat('resnet.mat', mdict=resnet_data)
