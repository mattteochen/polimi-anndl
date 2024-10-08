import sys
import keras
import pandas as pd
import sklearn as sk
import scipy as sp
import tensorflow as tf
import torch
import platform

print (f"Python Platform: {platform.platform ()}")
print (f"Tensor Flow Version: {tf.__version__}")
print(f"Keras Version: {keras.__version__}")
print ()

print (f"Python {sys.version}")
print (f"Pandas {pd.__version__}")
print (f"Scikit-Learn {sk.__version__}")
print (f"SciPy {sp.__version__}")
gpu = len (tf.config.list_physical_devices ('GPU'))>0
print ("tf: GPU is", "available" if gpu else "NOT AVAILABLE")
if not torch.backends.mps.is_available():
    if not torch.backends.mps.is_built():
        print("torch: MPS not available because the current PyTorch install was not "
              "built with MPS enabled.")
    else:
        print("torch: MPS not available because the current MacOS version is not 12.3+ "
              "and/or you do not have an MPS-enabled device on this machine.")

else:
	print('torch: MPS GPU available')