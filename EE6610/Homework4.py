import numpy as np
a,b=input().split(',')                   #Get a,b
a=(float(a))
b=(float(b))
t=2.23                                   #Search from table
sample_size=[10,100,1000,10000,100000,1000000]
print("\tMax\t\t\tMin\t\t\tLength")
for size in sample_size:                 #Try different sample size
    obs_mean=[]                          #initialization
    for i in range(11):                  #Get 11 random estimations
        unif=np.random.uniform(0,1,size)
        par=a*((1-unif)**(-1/b))
        mean=np.mean(par)               #Calculate each mean
        obs_mean.append(mean)            #Add in list
    True_Mean=np.mean(obs_mean)          #Observed Mean
    Dev=np.std(obs_mean,ddof=1)          #Observed Standard Deviation
    Ur=t*Dev/(np.sqrt(11))               #Confidence interval
    max=True_Mean+Ur
    min=True_Mean-Ur
    print(min,'\t',max,'\t',Ur*2)