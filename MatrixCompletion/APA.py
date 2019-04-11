from sparsesvd import sparsesvd
import matplotlib.pyplot as plt
import scipy.sparse as ss
import numpy as np
import random
import math

class APA(object):

    def __init__(self,n1,n2,r,mode):
        self.n1=n1
        self.n2=n2
        self.r=r
        self.mode=mode
        #M(n1*r)M(r*n2)-->M(n1*n2) with rank r
        self.M = np.dot(np.random.randn(n1,r),np.random.randn(r,n2))
        self.Morigin=self.M
        #Sample index for range(n1*n2)
        Omega_index=random.sample(range(n1*n2),round(n1*n2*0.45))
        #Set up n1*n2 Find index of Omega
        self.Omega=np.unravel_index(Omega_index,(n1,n2))
        #(array([3, 0, 4, 1, 2, 1, 4, 2, 4, 1, 0], dtype=int64), array([2, 3, 1, 4, 2, 3, 0, 1, 2, 1, 4], dtype=int64))
        #GMM & T_Cond
        self.GMM_noise=self.GMM(0,6)
        v=self.GMM_noise[self.Omega]
        self.T_Cond1=np.linalg.norm(v,1)
        self.T_Cond2=np.linalg.norm(v,2)
        #GMM_noise Remix
        #self.M=self.M+self.GMM_noise
        self.M_Omega=self.M[self.Omega]
        #Save as sparse
        self.P_Omega=ss.csr_matrix((self.M_Omega,self.Omega),shape=(n1,n2))

    def GMM(self,mu,snr):
        M_omega=ss.csr_matrix((self.Morigin[self.Omega],self.Omega),shape=(self.n1,self.n2)).toarray()
        M_f=np.linalg.norm(M_omega, ord='fro')
        sigma_v=math.sqrt(M_f**2/(len(self.Omega[0])*math.pow(10,snr/10)))
        sigma=1/math.sqrt(10.9)*sigma_v
        BIG_index=np.unravel_index(random.sample(range(self.n1*self.n2),round(self.n1*self.n2*0.1)),(self.n1,self.n2))
        OmegaT=list(map(list,zip(*BIG_index)))
        GMM_SMALL=np.random.normal(mu, sigma, (self.n1,self.n2))
        GMM_BIG=np.random.normal(mu, 10*sigma, (self.n1,self.n2))
        for var in OmegaT:
            GMM_SMALL[var[0],var[1]]=GMM_BIG[var[0],var[1]]
        return GMM_SMALL

    def Equality(self):
        self.RMSE=[]
        self.RMSE.append(np.linalg.norm((self.P_Omega.toarray()-self.Morigin), ord='fro') / np.linalg.norm(self.Morigin, ord='fro'))
        X=self.P_Omega.toarray()
        m=self.P_Omega.toarray()
        #index as (2,4)
        #OmegaT=list(map(list,zip(*self.Omega)))
        i=0
        while i<500:
            u,s,v=np.linalg.svd(X)
            Y=self.TruncatedSvd(s,u,v)
            Y_Omega=Y[self.Omega]
            Z=ss.csr_matrix((Y_Omega,self.Omega),shape=(self.n1,self.n2))
            X=m+Y-Z.toarray()
            self.RMSE.append(np.linalg.norm((X-self.Morigin), ord='fro') / np.linalg.norm(self.Morigin, ord='fro'))
            i+=1
        x_coordinate = range(len(self.RMSE))
        plt.xlabel('Number of iterations')
        plt.ylabel('RMSE')
        #log
        plt.yscale('log')
        plt.plot(x_coordinate,self.RMSE,'-')
        plt.show()
        return X
    
    def NORM_1(self):
        self.RMSE=[]
        self.RMSE.append(np.linalg.norm(self.P_Omega.toarray()-self.Morigin, ord='fro') / np.linalg.norm(self.Morigin, ord='fro'))
        X=self.P_Omega.toarray()
        m=self.P_Omega.toarray()
        OmegaT=list(map(list,zip(*self.Omega)))
        i=0
        while i<100:
            u,s,v=np.linalg.svd(X)
            Y=self.TruncatedSvd(s,u,v)
            Y_Omega=Y[self.Omega]
            Z=ss.csr_matrix((Y_Omega,self.Omega),shape=(self.n1,self.n2))
            lamb=self.bisection(np.sort(abs((Z.toarray()-m)[self.Omega])))
            X=np.sign(Z.toarray()-m)*np.maximum(abs(Z.toarray()-m)-lamb,0)+Y-Z.toarray()+m
            self.RMSE.append(np.linalg.norm((X-self.Morigin), ord='fro') / np.linalg.norm(self.Morigin, ord='fro'))
            i+=1
        x_coordinate = range(len(self.RMSE))
        plt.xlabel('Number of iterations')
        plt.ylabel('RMSE')
        #lg scale
        #plt.yscale('log')
        plt.plot(x_coordinate,self.RMSE,'-')
        plt.show()
        return X

    def NORM_2(self):
        self.RMSE=[]
        self.RMSE.append(np.linalg.norm((self.P_Omega.toarray()-self.Morigin), ord='fro') / np.linalg.norm(self.Morigin, ord='fro'))
        X=self.P_Omega.toarray()
        m=self.P_Omega.toarray()
        i=0
        while i<100:
            u,s,v=np.linalg.svd(X)
            Y=self.TruncatedSvd(s,u,v)
            Y_Omega=Y[self.Omega]
            Z=ss.csr_matrix((Y_Omega,self.Omega),shape=(self.n1,self.n2))
            X=m+(self.T_Cond2/np.linalg.norm((Z.toarray()-m),ord=2))*(Z.toarray()-m)+Y-Z.toarray()
            #X=m+(Z.toarray()-m)/np.maximum(np.linalg.norm((Z.toarray()-m),ord=2)**2,self.T_Cond2)+Y-Z.toarray()
            self.RMSE.append(np.linalg.norm((X-self.Morigin), ord='fro') / np.linalg.norm(self.Morigin, ord='fro'))
            i+=1
        x_coordinate = range(len(self.RMSE))
        plt.xlabel('Number of iterations')
        plt.ylabel('RMSE')
        #plt.yscale('log')
        plt.plot(x_coordinate,self.RMSE,'-')
        plt.show()
        return X

    def TruncatedSvd(self,sigma, u, v):
        m = len(u)
        n = len(v[0])
        a = np.zeros((m, n))
        for k in range(self.r):
            uk = u[:, k].reshape(m, 1)
            vk = v[k].reshape(1, n)
            a += sigma[k] * np.dot(uk, vk)
        return a

    def Print2TXT(self):
    # The output part 2
        list=self.RMSE
        output = open('MatrixCompletion\data.txt','w',encoding='gbk')
        for row in list:
	        output.write(str(row))
	        output.write('\n')
        output.close()
        return 0
    
    def KKT_NORM1(self,x,array):
        sum=0.0
        for var in array:
            sum+=np.maximum(var-x,0.0)
        return sum-self.T_Cond1

    def bisection(self,array):
        #a=max(array[-1]-1,array[0]-1/len(array))
        #b=array[-1]-1/len(array)
        a=array[0]
        b=array[-1]
        while(1):
            x = (a + b)/2.0
            if self.KKT_NORM1(x,array)==0.0:
                break
            elif ( self.KKT_NORM1(x,array)*self.KKT_NORM1(a,array)<0.0 ):
                b = x
            elif ( self.KKT_NORM1(x,array)*self.KKT_NORM1(a,array)>0.0 ):
                a = x
            elif (a > b):
                return a
            if abs(a - b)< 1e-5:
                break
        return x

if __name__ == "__main__":
    obj=APA(150,300,10,0)
    if obj.mode==0:
        obj.Equality()
    elif obj.mode==1:
        obj.NORM_1()
    elif obj.mode==2:
        obj.NORM_2()
