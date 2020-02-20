import numpy as np
import math
import matplotlib.pyplot as plt
from numpy.linalg import inv
from numpy.linalg import eig
from numpy import dot
from numpy import diag

# Initial Values

dimx = 4
dimy = 1
T=100
N=4
r=0.5
q=0.1
mu=np.zeros((dimx,1))
muv = np.zeros((dimy,N))
A = np.array([[0.7,0.1,0,0.2],[0.2,0.6,0,0.3],[0.1,0.2,0.7,0.5],[0,0.2,0,0.7]])
C = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
Q = np.array([[q,0,0,0],[0,q,0,0],[0,0,q,0],[0,0,0,q]])
R = np.array([[r,0,0,0],[0,r,0,0],[0,0,r,0],[0,0,0,r]]) 
X0 = np.random.normal(0,1,4).T
Xhat0 = np.zeros((dimx))
Sigma0 = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
time = np.arange(0,T)

# Functions

# Linear Process
def process(A,Q,mu,R,muv,dimx,dimy,X0,T,N):
    x = np.zeros((dimx, T+1))
    y = np.zeros((dimy*N, T+1))
    x[:,1] = X0
    for t in range(1,T):
        y[:,t] = dot(C,x[:,t]) + diag(np.random.normal(muv,np.sqrt(R)))
        x[:,t+1] = dot(A,x[:,t]) + diag(np.random.normal(mu,np.sqrt(Q)))
    return x, y

#Centralized kalman Filter with no delay in obs.
def CenKal(y,R,Q,C,dimx,Xhat0,Sigma0,T):          
    X = np.zeros((dimx,T+1))
    P = np.zeros((dimx*(T+1),dimx))
    nu = np.zeros(dimx)
    IS = Pt = K = np.zeros((dimx,dimx))
    X[:,0], P[0:dimx,:] = Xhat0, Sigma0
    for t in range(0,T):
        nu = y[:,t+1] - dot(C, dot(A, X[:,t]))
        Pt = dot(A, dot(P[dimx*t:dimx*(t+1),:], A.T)) + Q
        IS = dot(C, dot(Pt, C.T)) + R
        K = dot(Pt, dot(C.T, inv(IS)))
        X[:,t+1] =  dot(A, X[:,t]) + dot(K, nu)
        P[dimx*(t+1):dimx*(t+2),:] = dot(A, dot(P[dimx*t:dimx*(t+1),:], A.T)) - dot(K, dot(Pt.T, C)) + Q
    return X, P


# Test

[x, y]=process(A,Q,mu,R,muv,dimx,dimy,X0,T,N)
[XCen, PCen]=CenKal(y,R,Q,C,dimx,Xhat0,Sigma0,T)


# Plot

plt.plot(XCen[0,time], label = 'Kalman estimate')
plt.plot(x[0,time], label = 'State')
plt.plot(x[0,time] - XCen[0,time], label = 'Error')
plt.ylabel('value')
plt.ylabel('time')
plt.legend()
plt.show()
