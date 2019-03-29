import numpy as np
import matplotlib.pyplot as plt

def factorial(num):
    Temp = 1
    if num < 0:return False
    elif num == 0:return 1
    else:
        for i in range(1,num + 1):
            Temp = Temp*i
        return Temp
"""
def ErlangB(A,k):
    if k==0:return 1
    elif k<=0:return False
    else:
        Up=A**k/factorial(k)
        Low=0
        for i in range(0,k):
            Low+=A**i/factorial(i)
        return Up/Low
"""

def ErlangC(A,k):
    if k==0:return 1
    elif k<=0:return False
    else:
        Up=((A**k)*k)/(factorial(k)*(k-A))
        Low=0
        for i in range(0,k-1):
            Low+=A**i/factorial(i)
        return Up/(Low+Up)

def MeanDelay(Ck,miu,Lamda,k):
    return Ck/(miu*k-Lamda)+1/miu

if __name__=="__main__":
    ED=[]
    U=[]
    for k in range(1,100):
        Lamda=1
        miu=10
        A=Lamda/miu
        Ck=ErlangC(A,k)
        ED.append(MeanDelay(Ck,miu,Lamda,k))
        U.append(A/k)
    s='Result Analysis λ='+str(Lamda)+' μ='+str(miu)+' A='+str(A)
    plt.title("%s"%s)
    plt.plot(range(1,100,1), ED, color='red', label='MeanDelay')
    plt.plot(range(1,100,1), U, color='blue', label='Utilization')
    plt.legend()
    plt.xlabel('Num of Sever')
    plt.ylabel('Utilization & MeanDelay')
    plt.show()
