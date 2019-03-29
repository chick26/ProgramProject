import numpy as np
import random
import math

def Markov(Lamda,Miu,k):
    MaxQueue=100000
    Queue=0
    Customer_Arrival=0
    Customer_Block=0
    while Customer_Arrival<MaxQueue:
        Arrive=np.random.uniform(0,1)
        if Arrive <=Lamda/(Queue*Miu+Lamda):
            Customer_Arrival+=1
            if Queue==k:Customer_Block+=1
            else:Queue+=1
        else:Queue-=1
    return Customer_Block/Customer_Arrival

print(Markov(39.0,2.0,10))
