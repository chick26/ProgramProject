
#%%
import numpy as np
import time
import math
import random
from numpy import linalg as la
from sparsesvd import sparsesvd
from scipy.sparse.linalg import norm
import scipy.sparse as ss
import scipy.io
import random
from __future__ import division
import matplotlib.pyplot as plt
import pandas as pd
import itertools
get_ipython().run_line_magic('matplotlib', 'inline')

#%% [markdown]
# ## 1. SVT Algorithm
# **The referenced paper is 'A Singular Value Thresholding Algorithm for Matrix Completion'.** [It can be acccessed here](https://arxiv.org/pdf/0810.3286.pdf)

#%%
"""
 This whole part is only for real number calculations. I didn't implement the complex number part.
 
 M is a random full matrix which we try to recover;
 n1,n2 are the shape of M;
 r is the rank of matrix M;
 df is the degree of freedom;
 m is number of samples;
 Omega is the sampling space;
 data is sampled data in space Omega;
 P_Omega_M is the sampled matrix;
 tau is the threshold; 
 delta is the learning rate;
 maxiter is the number of iteration;
 tol is tolerance;
 incre is the increment of number of singular values needed to calculate in each iteration.
 
"""

# parameter initialized
n1, n2, r = 150, 300, 10
M = np.random.random((n1,r)).dot(np.random.random((r,n2)))
df = r*(n1+n2-r);
oversampling = 5; 
m = min(5*df,round(.99*n1*n2)); 
p  = m/(n1*n2);
ind = random.sample(range(n1*n2),m)
Omega = np.unravel_index(ind, (n1,n2))

data = M[Omega]

tau = 5*math.sqrt(n1*n2); 
delta = 2
maxiter = 400
tol = 1e-4
incre = 5

"""
SVT
"""
start = time.clock()

b = data

r = 0
P_Omega_M = ss.csr_matrix((b,Omega),shape = (n1,n2))

normProjM = norm(P_Omega_M)

k0 = np.ceil(tau / (delta*normProjM))

Y = k0*delta*P_Omega_M
rmse = []

for k in range(maxiter):
    s = r + 1
    while True:
        u1,s1,v1 = sparsesvd(ss.csc_matrix(Y),s)
        if s1[s-1] <= tau : break
        s = min(s+incre,n1,n2)
        if s == min(n1,n2): break
    
    r = np.sum(s1>tau)
    U = u1.T[:,:r]
    V = v1[:r,:]
    S = s1[:r]-tau
    x = (U*S).dot(V)
    x_omega = ss.csr_matrix((x[Omega],Omega),shape = (n1,n2))

    if norm(x_omega-P_Omega_M)/norm(P_Omega_M) < tol:
        break
    
    Y += delta*(P_Omega_M-x_omega)
    diff = ss.csr_matrix(M-x)
    rmse.append(norm(x_omega-P_Omega_M) / np.sqrt(n1*n2))
    
print ('calculating time: '+ str(time.clock() - start))


#%%
print ('Recovered Matrix:')
print (x)


#%%
print ('Original Matrix:')
print (M)


#%%
x_coordinate = range(len(rmse))
plt.ylim(0,0.5)
plt.xlabel('Number of iterations')
plt.ylabel('RMSE')
plt.plot(x_coordinate,rmse,'-')

#%% [markdown]
# ## Summary: Three ways to do SVD in python  
# **The first is for normal matrix SVD using numpy.linalg;
# The second and the third is specially for sparse SVD.   
# There are 2 differences between the second and the third:   
# 1) the second's singular values are in increasing order while the third's are in descending order.  
# 2) The left singular vector of the seocnd is the transpose of the third one.**

#%%
# 1st
U,s,V = la.svd(Y.todense())
print (U.shape,s.shape,V.shape)
S = np.zeros(Y.shape)
index = s.shape[0]
S[:index, :index] = np.diag(s)
np.dot(U,np.dot(S,V))


#%%
# 2nd
u1,s1,v1 = ss.linalg.svds(Y,6)
print (u1.shape,s1.shape,v1.shape)
print (s1)
(u1*s1).dot(v1)


#%%
# 3rd
ut, s, vt = sparsesvd(ss.csc_matrix(Y),6)
print (ut.shape,s.shape,vt.shape)
print (s)
(ut.T*s).dot(vt)

#%% [markdown]
# ## Solving the Netflix Problem using SVT above
#%% [markdown]
# **I have done some preprocessing to the Netfilx files(e.g. mapping the users id ranging from [6,2649429] to their actual id number [1-480189]) which takes about an hour just for running the program.  
# Then I wrote the rate, rates' row index and rates' col index into the following 'rate.csv', 'row.csv', 'col.csv' files And I directly read from these files to save time and write them into a sparse matrix M_original.  
# The rows of M_original are the users' actual id numbers ranging from [1,480189] while the columns are the id of 17770 movies. The data in the matrix is the rates. 
# **

#%%
start = time.clock()

row = pd.read_csv('G:/nf_prize_dataset/download/row.csv',header = None).iloc[:,1]
col = pd.read_csv('G:/nf_prize_dataset/download/col.csv',header = None).iloc[:,1]
rate = pd.read_csv('G:/nf_prize_dataset/download/rate.csv',header = None).iloc[:,1]

M_original = ss.csr_matrix((rate,(row,col)),shape = (480189,17770))

time.clock() - start


#%%
def SVT(M1,iter_num):
    
    n1,n2 = M1.shape
    total_num = len(M1.nonzero()[0])
    proportion = 1.0
    idx = random.sample(range(total_num),int(total_num*proportion))
    Omega = (M1.nonzero()[0][idx],M1.nonzero()[1][idx])
    p = 0.5
    tau=20000
    delta = 2
    maxiter = iter_num
    tol = 0.001
    incre = 5

    # SVT
    r = 0
    b = M1[Omega]
    #P_Omega_M = ss.csr_matrix((np.ravel(M1[Omega]),Omega),shape = (n1,n2))
    P_Omega_M = ss.csr_matrix((np.ravel(b),Omega),shape = (n1,n2))
    #P_Omega_M = M1[Omega]
    normProjM = norm(P_Omega_M)
    k0 = np.ceil(tau / (delta*normProjM))
    Y = k0*delta*P_Omega_M
    rmse = []

    for k in xrange(maxiter):
        #print str(k+1) + ' iterative.'
        s = r + 1
        while True:
            u1,s1,v1 = sparsesvd(ss.csc_matrix(Y),s)
            if s1[s-1] <= tau : break
            s = min(s+incre,n1,n2)
            if s == min(n1,n2): break

        r = np.sum(s1>tau)
        U = u1.T[:,:r]
        V = v1[:r,:]
        S = s1[:r]-tau
        x = (U*S).dot(V)
        x_omega = ss.csr_matrix((x[Omega],Omega),shape = (n1,n2))                                      

        if norm(x_omega-P_Omega_M)/norm(P_Omega_M) < tol:
            break

        diff = P_Omega_M-x_omega
        Y += delta*diff
        #rmse.append(norm(diff)/np.sqrt(n1*n2))
        rmse.append(la.norm(M1[M1.nonzero()]-x[M1.nonzero()]) / np.sqrt(len(x[M1.nonzero()])))

    return rmse

#%% [markdown]
# ### 1. # of iteration increases (# of entries fixed = 1000) 
# ### 50,100,150,200 iterations are picked

#%%
start = time.clock()
num_of_entries = 1000
M1 = M_original[:num_of_entries]
iter_list = range(50,250,50)
rmse1 = []
for item in iter_list:
    rmse1.append(SVT(M1,item)[-1])
time.clock() - start


#%%
plt.plot(iter_list,rmse1,'*-')
plt.xlabel('Number of iterations')
plt.ylabel('RMSE')

#%% [markdown]
# ### 2. # of entries increases (# of iteration fixed = 50)  
# ### the first 1000,5000,10000, 20000 entries are picked

#%%
start = time.clock()
entry_list = [1000,5000,10000,20000]
num_of_iter = 50
rmse2 = []
for entry in entry_list:
    M2 = M_original[:entry]
    rmse2.append(SVT(M2,num_of_iter)[-1])
time.clock() - start


#%%
plt.plot(entry_list,rmse2,'*-')
plt.xlabel('Number of entries')
plt.ylabel('RMSE')


#%%
prize = SVT(M1,500)[-1]
print ('RMSE of the first 1000 entries after 500 iterations is ' + str(prize))

#%% [markdown]
# ### The first 30 nonzero positions in original matrix and corresponding positions in the recovered matrix

#%%
np.ravel(M1[M1.nonzero()])[:30]


#%%
x[M1.nonzero()][:30]


#%%
haha = np.round(x[M1.nonzero()])


#%%
np.sum(haha == np.ravel(M1[M1.nonzero()]))/len(haha)


#%%
# the last 10 rmse of the first 1000 entries after 500 iterations
rmse[-10:]


