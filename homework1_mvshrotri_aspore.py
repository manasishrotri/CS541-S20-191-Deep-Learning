#@author: Manasi Shrotri Akshta Pore
#HW 1_group14
#"--------------------------------------------------------------------------------------

import numpy as np

#Problem 1

def problem_a (A, B):
    ans=A+B
    print("\nproblem a:\n",ans)
    return ans

def problem_b (A, B, C):
    ans=(np.dot(A,B)-C)
    print("\nproblem b:\n",ans)
    return ans

def problem_c (A, B, C):
    ans=(A*B+(C.T))
    print("\nproblem c:\n",ans)
    return ans

def problem_d (x, y):
    ans=(np.dot(x.T,y))
    print("\nproblem d:\n",ans)
    return ans

def problem_e (A):
    shapeofA=A.shape
    ans=np.zeros(shapeofA)
    print("\nproblem e:\n",ans)
    return ans

def problem_f (A, x):
    ans=np.linalg.solve(A,x)
    print("\nproblem f:\n",ans)
    return ans

def problem_g (A, x1):
    ans=(np.linalg.solve(A.T,x1.T)).T
    print("\nproblem g:\n",ans)
    return ans

def problem_h (A, alpha):
    shapeofA=A.shape
    ans=A+(alpha*np.eye(shapeofA[0]))
    print("\nproblem h:\n",ans)
    
    return ans

def problem_i (A, i, j):
    ans=A[i,j]
    print("\nproblem i:\n",ans)
    return ans

def problem_j (A, i):
    ans=np.sum(A[i,::2])
    print("\nproblem j:\n",ans)
    #"Considering elements 0,2,4,... 
    return ans

def problem_k (A, c, d):
    
    select_A=A[(A>=c)&(A<=d)]
    #selecting non zero values for mean calculation
    select_A=select_A[np.nonzero(select_A)]
    ans=np.mean(select_A)
    print("\nproblem k:\n",ans)

    return ans

def problem_l (A, k):
    
    val,vec=np.linalg.eig(A)
    LargeEigenVal=val.argsort()[::-1]
    LargeEigenVal=LargeEigenVal[:k] #"Select K values
    ans=vec[:,LargeEigenVal]
    print("\nproblem l:\n",ans)
    return ans

def problem_m (x, k, m, s):
    
    n=len(x)
    z=np.ones(n)
    I=np.eye(n)
    mean_=m*z #"mean
    cov_=s*I  #"covariance
    ans=np.random.multivariate_normal(x+mean_, cov_, (k)).T
    print("\nproblem m:\n",ans)
    return ans

def problem_n (A):
    
    rows=A.shape[0]
    rows=np.random.permutation(rows)#permute only rows
    ans=A[rows,:]
    print("\nproblem n:\n",ans)
    return ans 

#"-----------------------------------------------------------------------------
#"Problem 2

def linear_regression (X_tr, y_tr):
    w=np.linalg.solve(np.dot(X_tr.T,X_tr),np.dot(X_tr.T,y_tr))
    return w

def fmse(X,w,Y):
    Y_hat=np.dot(X,w)
    fmse=(1/(2*Y_hat.shape[0]))*np.sum((Y_hat-Y)**2)
    return fmse

def train_age_regressor ():
    # Load data
    X_tr =np.reshape(np.load("age_regression_Xtr.npy"), (-1, 48*48))    
    ytr = np.load("age_regression_ytr.npy")
    X_te = np.reshape(np.load("age_regression_Xte.npy"), (-1, 48*48))
    yte = np.load("age_regression_yte.npy")

    w = linear_regression(X_tr, ytr)

    # Report fMSE cost on the training and testing data (separately)
    # fmse for training
    print("\n\n Problem 2:\n")
    fmse_training=fmse(X_tr,w,ytr)
    print("Fmse for train data is :"+"{:.2f}".format(fmse_training))
    fmse_test=fmse(X_te,w,yte)
    print("Fmse for test data is :"+"{:.2f}".format(fmse_test))
    
#"-----------------------------------------------------------------------------

#"Assigning values to matrices and variables
#A=np.random.rand(3,2)
#B=np.random.rand(3,2)
#C=np.random.rand(3,2)
A=np.array([[1,2,3],[4,5,6],[7,8,10]], float)
B=np.array([[1,2,3],[4,5,6],[7,8,10]], float)
C=np.array([[1,2,3],[4,5,6],[7,8,10]], float)
x=np.array([1,2,3]).T
#x=np.array([[1],[2],[3]],float)
y=np.array([[1],[2],[3]],float)

x1=x.T

i=1
j=2

alpha=1.2

c=2
d=6
k=2
m=2
s=0.5
#"---------------------------------------------------------------------
#"Call functions
problem_a(A,B)
problem_b(A,B,C)
problem_c(A,B,C)
problem_d(x,y)
problem_e(A)
problem_f (A, x)
problem_g (A, x.T)
problem_h (A, alpha)
problem_i (A, i, j)
problem_j (A, i)
problem_k (A, c, d)
problem_l (A, k)
problem_m (x, k, m, s)
problem_n (A)

train_age_regressor()
