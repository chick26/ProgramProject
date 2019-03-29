import scipy.misc
import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

org_img=cv2.imread("pic.png")                           #Load original image
gray_image = cv2.cvtColor(org_img, cv2.COLOR_BGR2GRAY)
gray_image=gray_image/255                                #Set[0,255] into [0,1]

def findedge(gray_img):    
    T=0.5                       
    edge_img = np.zeros(gray_img.shape)                  #Define a Edgemap
    for x in range(2,gray_img.shape[0]-1):               #Count gx and gy
        for y in range(2,gray_img.shape[1]-1):
            gx=gray_img[x+1,y-1]+gray_img[x+1,y]+gray_img[x+1,y+1]-gray_img[x-1,y-1]-gray_img[x-1,y]-gray_img[x-1,y+1]
            gy=gray_img[x-1,y+1]+gray_img[x,y+1]+gray_img[x+1,y+1]-gray_img[x-1,y-1]-gray_img[x,y-1]-gray_img[x+1,y-1]
            A=math.sqrt(gx*gx+gy*gy)
            if(A>=T):                                     #Threshold
                edge_img[x,y]=255                         #Set the edge with 255
    scipy.misc.imsave('outfile.jpg', edge_img)            #Save Edge image
    return edge_img

findedge(gray_image)

img = mpimg.imread('outfile.jpg')
plt.imshow(img,cmap='Greys_r')                
plt.title('Original train image')   
plt.show()