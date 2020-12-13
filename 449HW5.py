import numpy as np
def gradient_f(r,p,Q):#r 1*2 p 1*2 Q 2*2
    res1 = np.dot(r,r.T)
    res2 = np.dot(np.dot(p,Q),p.T)
    alpha = res1/res2;
    return alpha

#Q = np.array([[1, 2],[1,2]]) 
#r = np.array([[1, 2]])
#p = np.array([[1, 2]])
#gradient_f(r,p,Q)

def stepfour(x,alpha,p):# 1*2 1*1 1*2
    xnew = x+np.dot(alpha,p)
    return xnew

def stepfive(r,alpha,Q,p):#p 1*2 Q 2*2 r 1*2
    rnew = r - np.dot(alpha,np.dot(Q,p.T)).T
    return rnew

#Q = np.array([[1, 8],[1,2]]) 
#r = np.array([[1, 2]])
#x = np.array([[1, 2]]) 
#p = np.array([[1, 2]])
#alpha = 2
#stepfour(x,alpha,p)
#stepfive(r,alpha,Q,p)

def stepsix(rnew,r,p):#p 1*2 rnew 1*2
    ratio = (rnew[0,0]*rnew[0,0]+rnew[0,1]*rnew[0,1])/(r[0,0]*r[0,0]+r[0,1]*r[0,1])
    pnew = rnew+np.dot(ratio,p)
    return pnew

##p = np.array([[1, 2]])
##r = np.array([[1, 2]])
##rnew = np.array([[1, 2]])
##stepsix(rnew,r,p)

def Conjugate_gradient(Q,b,x,tol,N):
    r = b - np.dot(Q,x)#1 * 2
    k = 0
    p = r#1*2
    while np.sqrt(r[0,0]*r[0,0]+r[0,1]*r[0,1]) > tol and k < 100:
        a = gradient_f(r,p,Q)#one dimension
        x = stepfour(x,a,p) # 1*2
        r_pre = r # 1*2
        r = stepfive(r,a,Q,p)#2*1
        p = stepsix(r,r_pre,p)#2*1
        k = k + 1
    res = 0.5*np.dot(x.T,np.dot(Q,x)) - np.dot(b.T,x)
    return x, res
