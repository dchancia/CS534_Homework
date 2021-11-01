import glmnet_python
from glmnet import glmnet
import scipy.io
import numpy as np

# load data as numpy arrays
data = scipy.io.loadmat("HW3_1.mat")
X = data["X"]
y = np.array(data["y"]).astype("float64")

lambdas = np.logspace(1, 100, num=1000)

fit = glmnet(x=X.copy(), y=y.copy(), alp)
glmnetPlot(fit, xvar='lambda', label=True);