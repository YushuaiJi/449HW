import numpy as np
from numpy.linalg import *
import matplotlib.pyplot as plt
from scipy.sparse import *

Inf = np.inf

#Code Problem1
def getMatrix(n=10, isDiagDom=True):
    '''
    Return an nxn matrix which is the discretization matrix
    from finite difference for a diffusion problem.
    To visualize A: use plt.spy(A.toarray())
    '''
    # Representation of sparse matrix and right-hand side
    assert n >= 2
    n -= 1
    diagonal = np.zeros(n + 1)
    lower = np.zeros(n)
    upper = np.zeros(n)

    # Precompute sparse matrix
    if isDiagDom:
        diagonal[:] = 1
    else:
        diagonal[:] = 1
    lower[:] = -0.5  # 1
    upper[:] = -0.5  # 1
    # Insert boundary conditions
    diagonal[0] = 1
    upper[0] = -0.5
    diagonal[n] = 1
    lower[-1] = -0.5

    A = diags(
        diagonals=[diagonal, lower, upper],
        offsets=[0, -1, 1], shape=(n + 1, n + 1),
        format='csr')

    return A


def getAuxMatrix(A):
    '''
    return D, L, U matrices for Jacobi or Gauss-Seidel
    D: array
    L, U: matrices
    '''
    # m = A.shape[0]
    D = csr_matrix.diagonal(A)
    L = -tril(A, k=-1)
    U = -triu(A, k=1)
    return D, L, U


def plotConvergence(x_true, x, k=2, scale='log'):
    '''
    Plot the error ||x[i] - x_true||_k against i
    x_true: (n,) vector
    x: (m, n) matrix, x[i] is the i-th iterate,
    x[0] = x0 is the initial guess
    m: total number of iterations
    scale: 'log' or 'linear'
    k: 1,2,..., of Inf
    '''
    numIter = x.shape[0]
    error = np.zeros(numIter)
    for i in range(numIter):
        error[i] = norm(x_true - x[i], k)
    if scale is 'log':
        plt.semilogy(error)
    else:
        plt.plot(error)



#Code Problem2
# %% compressed sparse row or csr
A = getMatrix(n=5)
# generate a random vector as the solution
x_true = np.random.randn(5)
b = A@x_true
# %%
# D is an array
D, L, U = getAuxMatrix(A)

numIter = 200 # total number of iterations
nDim = A.shape[0]
x = np.zeros((numIter+1, nDim))
# x[0] = np.zeros((nDim, ))
for k in range(numIter):
    # diagonal matrix (matrix multiplication) vector
    # = diagonal entry vector * vector elementwisely
    # D is a vector representing the diagonal entries
    x[k+1] = ((L+U)@x[k])/D + b/D
    if k % 10 is 0:
        # 1.0 after 80ish iteration is because
        # we have reached error of machine epsilon level
        print(norm(x[k+1] - x_true)/norm(x[k] - x_true) )


#rate of the convergence for Jacobi method 2
def jacobiIteration(A, b, x0=None, tol=1e-13, numIter=100):
    '''
    Jacobi iteraiton:
    A: nxn matrix
    b: (n,) vector
    x0: initial guess
    numIter: total number of iteration
    tol: algorithm stops if ||x^{k+1} - x^{k}|| < tol
    return: x
    x: solution array such that x[i] = i-th iterate
    '''
    n = A.shape[0]
    x = np.zeros((numIter + 1, n))
    if x0 is not None:
        x[0] = x0
    D, L, U = getAuxMatrix(A)
    for k in range(numIter):
        x[k + 1] = ((L + U) @ x[k]) / D + b / D
        if norm(x[k + 1] - x[k]) < tol:
            break

    return x[:k + 1]



plotConvergence(x_true,x[:5],scale = 'linear')
#set n = 10
#the Jacobi iteration right here, the number of iteration needed to reach t
#he same tolerance just it's like four times as big approximately
#four times as big. we need to increase our iterations.
#we also can test n = 100 and n = 1000

#Problem 3(optional)
# we must recongnize it is different from previous iteration, due to different dims of iteration equation Ma = D-L, x[k + 1] = (U @ x[k])/Ma+b/Ma.
def GaussSeidelIteration(A, b, x0=None, tol=1e-13, numIter=100):
    '''
    Jacobi iteraiton:
    A: nxn matrix
    b: (n,) vector
    x0: initial guess
    numIter: total number of iteration
    tol: algorithm stops if ||x^{k+1} - x^{k}|| < tol
    return: x
    x: solution array such that x[i] = i-th iterate
    '''
    n = A.shape[0]
    x = np.zeros((numIter + 1, n, n))
    if x0 is not None:
        x[0] = x0
    D, L, U = getAuxMatrix(A)
    for k in range(numIter):
        Ma = D-L
        x[k + 1] = (U @ x[k])/Ma+b/Ma
        #x[k + 1] = ((L + U) @ x[k]) / D + b / D
        if norm(x[k + 1] - x[k]) < tol:
            break

    return x[:k + 1]
