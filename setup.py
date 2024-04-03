import torch
import sys
import numpy
import cv2
print(numpy.version.__version__)
print(cv2.__version__)
print(sys.version)
print("Pytorch version：")
print(torch.__version__)
print("CUDA Version: ")
print(torch.version.cuda)
print("cuDNN version is :")
print(torch.backends.cudnn.version())
