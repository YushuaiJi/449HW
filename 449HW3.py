import numpy as np
import matplotlib.pyplot as plt
# %%
'''
n is the number of interior points for a finite difference matrix
'''
n = 21
I = np.diag(np.ones(n)) 
A = 2*I - np.diag(np.ones(n-1), k=1) -  np.diag(np.ones(n-1), k=-1)
#print(A)
T = I - A/2 # T is the matrix in the Jacobi iteration

# %%
tol = 1e-4
err = tol + 1
x = np.ones(21)
lambda_h = [] # empty list
k = 0
while err >= tol:
    y = T@x
    tmp1 = y.dot(x)
    tmp2 = x.dot(x)
    y = y/np.linalg.norm(y)
    err = np.linalg.norm(x - y)
    x = y
    lambda_h.append(tmp1/tmp2)
    k += 1
    if k > 10000:
        break

print(lambda_h[len(lambda_h)-1])
