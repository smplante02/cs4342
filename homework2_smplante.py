import numpy as np
import matplotlib.pyplot as plt

########################################################################################################################
# PROBLEM 2
########################################################################################################################
# Given a vector x of (scalar) inputs and associated vector y of the target labels, and given
# degree d of the polynomial, train a polynomial regression model and return the optimal weight vector.
def trainPolynomialRegressor(x, y, d):
    X = np.array
    for i in range(d+1):
        X = np.append(X, x**i, axis=0)
    m, n = X.shape

    # found in online formula
    X = X.reshape(d + 1, n)
    Xtranpose = np.transpose(X)

    # like method1 from PROBLEM 1
    return np.linalg.solve(X.dot(Xtranpose), X.dot(y))

########################################################################################################################
# PROBLEM 1
########################################################################################################################

# Given an array of faces (N x M x M, where N is number of examples and M is number of pixes along each axis),
# return a design matrix Xtilde ((M**2 + 1) x N) whose last row contains all 1s.
def reshapeAndAppend1s(faces):
    numberExamples = len(faces)
    n, m1, m2 = faces.shape
    onesRow = np.ones(numberExamples)
    facesTranspose = np.transpose(faces.reshape(n, m1 ** 2))
    return np.vstack((facesTranspose, onesRow))


# Given a vector of weights w, a design matrix Xtilde, and a vector of labels y, return the (unregularized)
# MSE.
def fMSE(wtilde, Xtilde, y):
    Xtranspose = np.transpose(Xtilde)
    yhat = Xtranspose.dot(wtilde)
    summation = np.sum((yhat - y) ** 2)
    return summation / (len(y) * 2)


# Given a vector of weights w, a design matrix Xtilde, and a vector of labels y, and a regularization strength
# alpha (default value of 0), return the gradient of the (regularized) MSE loss.
def gradfMSE(wtilde, Xtilde, y, alpha=0.):
    # make the bias for wtilde alpha
    wAlpha = np.copy(wtilde)
    wAlpha[-1] = alpha
    return Xtilde.dot(Xtilde.T.dot(wtilde) - y)/ len(y) + (alpha * wAlpha) / len(y)


# Given a design matrix Xtilde and labels y, train a linear regressor for Xtilde and y using the analytical solution.
def method1(Xtilde, y):
    Xtranspose = np.transpose(Xtilde)
    wtilde = np.linalg.solve(Xtilde.dot(Xtranspose), (Xtilde.dot(y)))
    return wtilde


# Given a design matrix Xtilde and labels y, train a linear regressor for Xtilde and y using gradient descent on fMSE.
def method2(Xtilde, y):
    return gradientDescent(Xtilde, y)


# Given a design matrix Xtilde and labels y, train a linear regressor for Xtilde and y using gradient descent on fMSE
# with regularization.
def method3(Xtilde, y):
    ALPHA = 0.1
    return gradientDescent(Xtilde, y, ALPHA)


# Helper method for method2 and method3.
def gradientDescent(Xtilde, y, alpha=0.):
    EPSILON = 3e-3  # Step size aka learning rate
    T = 5000  # Number of gradient descent iterations

    # setting random w value to start, from slides
    m, n = Xtilde.shape
    w = 0.01 * np.random.randn(m)

    for i in range(T):
        w = w - EPSILON * gradfMSE(w, Xtilde, y, alpha)
    return w


def visualization(solutionW):
    # not including b term in wtilde so go until -1 (last term)
    plt.imshow(np.reshape(solutionW[:-1], (48, 48)))
    plt.show()


if __name__ == "__main__":
    # Load data
    Xtilde_tr = reshapeAndAppend1s(np.load("age_regression_Xtr.npy"))
    ytr = np.load("age_regression_ytr.npy")
    Xtilde_te = reshapeAndAppend1s(np.load("age_regression_Xte.npy"))
    yte = np.load("age_regression_yte.npy")

    # training weights
    # analytical
    w1 = method1(Xtilde_tr, ytr)
    # visualization(w1)

    # grad descent i
    w2 = method2(Xtilde_tr, ytr)
    # visualization(w2)

    # grad descent ii
    w3 = method3(Xtilde_tr, ytr)
    # visualization(w3)


    print("Analytical Solution")
    print("Training MSE Cost: ", fMSE(w1, Xtilde_tr, ytr))
    print("Testing MSE Cost: ", fMSE(w1, Xtilde_te, yte))

    print("\nGradient Descent Solution")
    print("Training MSE Cost: ", fMSE(w2, Xtilde_tr, ytr))
    print("Testing MSE Cost: ", fMSE(w2, Xtilde_te, yte))

    print("\nRegularised Gradient Descent Solution")
    print("Training MSE Cost: ", fMSE(w3, Xtilde_tr, ytr))
    print("Testing MSE Cost: ", fMSE(w3, Xtilde_te, yte))


