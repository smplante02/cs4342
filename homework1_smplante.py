import numpy as np


def problem1(A, B):
    return A + B


def problem2(A, B, C):
    return np.subtract(A.dot(B), C)


def problem3(A, B, C):
    C = np.transpose(C)
    return (A * B) + C


def problem4(x, S, y):
    return np.dot(np.dot(np.transpose(x), S), y)


def problem5(A):
    a_rows, a_columns = A.shape
    return np.ones(a_rows, a_columns)


def problem6(A):
    a_rows, a_columns = A.shape
    for j in range(a_columns):
        for i in range(a_rows):
            if i == j:
                A[i][j] = 1
    return A


def problem7(A, alpha):
    a_rows, a_columns = A.shape
    B = np.eye(a_rows)
    return A + (B * alpha)


def problem8(A, i, j):
    return A[i][j]


def problem9(A, i):
    return np.sum(A, axis=i)


def problem10(A, c, d):
    return np.mean(A[c <= A <= d])


def problem11(A, k):
    eig = np.linalg.eig(A)
    vals, vectors = eig
    valsSorted = np.abs(np.sort(vals)[::-1])
    vectorsSorted = vectors[:, valsSorted.argsort()]

    return vectorsSorted[:k]


def problem12(A, x):
    return np.linalg.solve(A, x)


def problem13(x, k):
    return (np.repeat(x[:, np.newaxis], k, 1)).shape


def problem14(A):
    return np.random.permutation(A)


def main():
    # for testing purposes only
    a = np.array([[2, 2], [1, 1]])
    b = np.array([[1, 1], [2, 2]])
    c = np.array([[3, 3], [3, 3]])
    d = np.array([1, 2, 3])