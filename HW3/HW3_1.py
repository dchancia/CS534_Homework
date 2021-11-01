import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import ElasticNet, LogisticRegression
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import mean_squared_error

# load data as numpy arrays
data = scipy.io.loadmat("HW3_1.mat")
X = data["X"]
lambdas = data["lambdas"]
y = data["y"]

# create list with lambdas
list_lambdas = []
for i in range(lambdas.size):
    list_lambdas.append(lambdas[0, i])

test_errors = []
val_errors = []
train_errors = []

outer = KFold(n_splits=5, shuffle=True, random_state=0)
inner = KFold(n_splits=4, shuffle=True, random_state=0)
for train_idx, test_idx in outer.split(X):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    param_grid = {"C": 1/np.array(list_lambdas)}
    reg = LogisticRegression(penalty='elasticnet', solver="saga", l1_ratio=0.95)
    search = GridSearchCV(reg, param_grid, scoring='neg_mean_squared_error', cv=inner, return_train_score=True, refit=True)
    search.fit(X_train, y_train)
    val_error = abs(search.cv_results_["mean_test_score"])
    std = search.cv_results_["std_test_score"]
    index = np.argmin(val_error) + np.max(np.argwhere(val_error[np.argmin(val_error):-1] <= val_error[np.argmin(val_error)]
                                                                   + std[np.argmin(val_error)]))
    plt.figure()
    plt.errorbar(list_lambdas, val_error, yerr=std, fmt='-', color='k', ecolor='r', elinewidth=0.2)
    plt.scatter(list_lambdas[np.argmin(val_error)], val_error[np.argmin(val_error)], c='b')
    plt.scatter(list_lambdas[index], val_error[index], c='g')
    plt.xscale("log")
    plt.xlabel("lambda")
    plt.ylabel("error")

    selected_model = search.best_estimator_
    y_hat = selected_model.predict(X_test)
    test_error = mean_squared_error(y_test, y_hat)
    test_errors.append(test_error)
    val_errors.append(abs(search.cv_results_['split0_test_score'][index]))
    val_errors.append(abs(search.cv_results_['split1_test_score'][index]))
    val_errors.append(abs(search.cv_results_['split2_test_score'][index]))
    val_errors.append(abs(search.cv_results_['split3_test_score'][index]))
    train_errors.append(abs(search.cv_results_['split0_train_score'][index]))
    train_errors.append(abs(search.cv_results_['split1_train_score'][index]))
    train_errors.append(abs(search.cv_results_['split2_train_score'][index]))
    train_errors.append(abs(search.cv_results_['split3_train_score'][index]))

print(test_errors)
print(val_errors)
print(train_errors)
errors = [test_errors, train_errors, val_errors]

plt.figure()
plt.boxplot(errors, showfliers=False)
plt.ylabel("error")

plt.show()
