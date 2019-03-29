import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from sklearn import mixture


def demo1():
    mu ,sigma = 0, 0.01
    s = 0.1*np.random.normal(mu, sigma, (5,5))+0.9*np.random.normal(mu, 100*sigma, (5,5))
    print(s)

demo1()
