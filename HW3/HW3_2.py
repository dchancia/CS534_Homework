import glmnet_py
from glmnet import glmnet
import scipy.io
import numpy as np

# load data as numpy arrays
data = scipy.io.loadmat("HW3_1.mat")
X = data["X"]
y = np.float64(data["y"])

lambdas = np.logspace(1, 100, num=1000)

fit = glmnet(x=X, y=y, alpha=0.95, nlambda=1000)
glmnetPlot(fit, xvar="lambda", label=True)