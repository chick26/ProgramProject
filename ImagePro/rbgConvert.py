import scipy.misc
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
from scipy.misc import imread
rgb_img=imread("pic1.jpg")


offset = np.array([16, 128, 128])                    #Transformational Matrix For Ycbcr
offset1=np.array([1,1,1])                            #Transformational Matrix For CMY

def rgb2ycbcr(img):                                  #Convert Func of RBG to Ycrcb
    ycbcr_img = np.zeros(img.shape)                  #Define a Zero Matrix of Ycrcb
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            r=img[x,y,0]*1.0
            g=img[x,y,1]*1.0
            b=img[x,y,2]*1.0
            ycbcr_img[x, y, 0] = .299*r + .587*g + .114*b
            ycbcr_img[x, y, 1] = 128+(b-ycbcr_img[x, y, 0])*0.564
            ycbcr_img[x, y, 2] = 128+(r-ycbcr_img[x, y, 0])*0.713
    scipy.misc.imsave('outfile.jpg', ycbcr_img)      #Save Ycbcr image
    return ycbcr_img

def rgb2cmy(img):                                    #Convert Func of RBG to CMY
    cmy_img = np.zeros(img.shape)                    #Define a Zero Matrix of CMY
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            cmy_img[x, y, :] = np.round(offset1*255-img[x, y, :]) #Transform Process
    scipy.misc.imsave('outfile1.jpg', cmy_img)       #Save CMY image
    return cmy_img

def rgb2hsv(img):
    hsv_img = np.zeros(img.shape)                    #Define a Zero Matrix of Ycrcb
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            r=img[x,y,0]/255.0                       #Make into [0~1]&float
            g=img[x,y,1]/255.0
            b=img[x,y,2]/255.0
            cmax=max(r,g,b)
            cmin=min(r,g,b)                          #Get Max&Min
            diff=cmax-cmin
            #Calculate S        
            if cmax!=0:hsv_img[x,y,1]=diff/cmax
            else: hsv_img[x,y,1]=0
            #CalcuHlate H
            if diff==0: hsv_img[x,y,0]=0
            elif cmax==r:hsv_img[x,y,0]=60*((g-b)/diff)
            elif cmax==g:hsv_img[x,y,0]=60*((b-r)/diff)+120
            elif cmax==b:hsv_img[x,y,0]=60*((r-g)/diff)+240
            if hsv_img[x,y,0]<0:hsv_img[x,y,0]+=360  #Transform Process
            hsv_img[x,y,0]=hsv_img[x,y,0]/2
            hsv_img[x,y,1]=hsv_img[x,y,1]*255
            hsv_img[x,y,2]=cmax*255        #Make into [0~255]
    scipy.misc.imsave('outfile2.jpg', hsv_img)       #Save HsvJpg
    return hsv_img

def printplot(img):
    plt.figure(num='',figsize=(8,8))   

    plt.subplot(2,2,1)                                   #The window is divided into two rows and two columns
    plt.title('origin image')                            #and four sub-pictures, which can display four pictures.
    plt.imshow(img)
    plt.axis('off')                                      

    plt.subplot(2,2,2)                                   
    plt.title('Y channel')                           
    plt.imshow(img[:,:,0],plt.cm.gray)                   
    plt.axis('off')                                      

    plt.subplot(2,2,3)                                   #Third subgraph
    plt.title('Cb channel')                               #Title
    plt.imshow(img[:,:,1],plt.cm.gray)                   #Draw the third picture and it is grayscale
    plt.axis('off')                                      #Do not display coordinate size

    plt.subplot(2,2,4)                                   #Fourth subgraph
    plt.title('Cr channel')                               #Title
    plt.imshow(img[:,:,2],plt.cm.gray)                   #Draw the fourth picture and it is grayscale
    plt.axis('off')                                      #Do not display coordinate size
    plt.show()                                           #Show  
    return img

rgb2ycbcr(rgb_img)
#rgb2cmy(rgb_img)
#rgb2hsv(rgb_img)

img = mpimg.imread('outfile.jpg')
printplot(img)