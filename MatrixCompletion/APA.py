from sparsesvd import sparsesvd
import matplotlib.pyplot as plt
import numpy as np
import random
import scipy.sparse as ss

class APA(object):

    def __init__(self,n1,n2,r,mode,):
        self.n1=n1
        self.n2=n2
        self.r=r
        self.mode=mode
        mu=0
        sigma=1
        #M(n1*r)M(r*n2)-->M(n1*n2) with rank r
        self.M = np.dot(np.random.randn(n1,r),np.random.randn(r,n2))
        self.Morigin=self.M
        #GMM
        self.GMM_noise=0.9*np.random.normal(mu, sigma, (n1,n2))+0.1*np.random.normal(mu, 10*sigma, (n1,n2))
        #GMM_noise Remix
        self.M=self.M+self.GMM_noise
        #Sample index for range(n1*n2)
        Omega_index=random.sample(range(n1*n2),round(n1*n2*0.45))
        #Set up n1*n2 Find index of Omega
        self.Omega=np.unravel_index(Omega_index,(n1,n2))
        #(array([3, 0, 4, 1, 2, 1, 4, 2, 4, 1, 0], dtype=int64), array([2, 3, 1, 4, 2, 3, 0, 1, 2, 1, 4], dtype=int64))
        self.M_Omega=self.M[self.Omega]
        #Save as sparse
        self.P_Omega=ss.csr_matrix((self.M_Omega,self.Omega),shape=(n1,n2))



    def Noise_free(self):
        self.RMSE=[]
        self.RMSE.append(np.linalg.norm(np.subtract(self.Morigin, self.P_Omega.toarray()), ord='fro') / np.linalg.norm(self.Morigin, ord='fro'))
        #np.linalg.norm()
        X=self.P_Omega
        OmegaT=list(map(list,zip(*self.Omega)))
        i=0
        while i<100:
            ur,sr,vr=sparsesvd(ss.csc_matrix(X),self.r)
            U=ur.T[:,:self.r]
            V=vr[:self.r,:]
            S=sr[:self.r]
            Y=(U*S).dot(V)
            X=ss.csr_matrix(Y)
            for var in OmegaT:
                X[var[0],var[1]]=self.P_Omega[var[0],var[1]]
            self.RMSE.append(np.linalg.norm(np.subtract(self.Morigin, X.toarray()), ord='fro') / np.linalg.norm(self.Morigin, ord='fro'))
            i+=1
        x_coordinate = range(len(self.RMSE))
        plt.xlabel('Number of iterations')
        plt.ylabel('RMSE')
        plt.plot(x_coordinate,self.RMSE/self.RMSE[0],'-')
        #lg
        #plt.plot(x_coordinate,np.log10(self.RMSE/self.RMSE[0]),'-')
        plt.show()
        return X
    
    def NORM_1(self):
        self.RMSE=[]
        #np.linalg.norm()
        X=self.P_Omega
        OmegaT=list(map(list,zip(*self.Omega)))
        i=0
        while i<10:
            ur,sr,vr=sparsesvd(ss.csc_matrix(X),self.r)
            U=ur.T[:,:self.r-1]
            V=vr[:self.r-1,:]
            S=sr[:self.r-1]
            Y=(U*S).dot(V)
            X=ss.csr_matrix(Y)
            for var in OmegaT:
                X[var[0],var[1]]=self.P_Omega[var[0],var[1]]
            self.RMSE.append(np.linalg.norm(np.subtract(M, self.X), ord='fro') / np.linalg.norm(
              self.M, ord='fro'))
            i+=1
        print(X.toarray())
        print(self.M)
        x_coordinate = range(len(self.RMSE))
        plt.xlabel('Number of iterations')
        plt.ylabel('RMSE')
        plt.plot(x_coordinate,self.RMSE,'-')
        plt.show()
        return X

    def Print2TXT(self):
    # The output part 2
        list=self.RMSE
        output = open('MatrixCompletion\data.txt','w',encoding='gbk')
        for row in list:
	        output.write(str(row))
	        output.write('\n')
        output.close()

if __name__ == "__main__":
    obj=APA(150,300,10,0)
    if(obj.mode==0):
        obj.Noise_free()