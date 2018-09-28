# -*- coding: utf-8 -*-
"""
Created on Wed Sep 19 12:27:59 2018

@author: nadha
"""
# F18 EECS 504 HW1p1 Homography Estimation
import numpy as np
import matplotlib.pyplot as plt
import os

import eta.core.image as etai

def get_correspondences(img1, img2, n):
    '''
    Function to pick corresponding points from two images and save as .npy file
    Args:
	img1: Input image 1
	img1: Input image 2
	n   : Number of corresponding points 
   '''
    
    correspondence_pts = []
    fig = plt.figure()
    ax = fig.add_subplot(111)
    coords = []
    def onclick(event):
        global ix, iy
        ix, iy = event.xdata, event.ydata
        print("The current point is: ")
        print (ix, iy)
        
        coords.append((ix, iy))

        if len(coords) == n:
            fig.canvas.mpl_disconnect(cid)
            plt.close()
        return coords

    ax.imshow(img1)
    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()
    correspondence_pts.append(coords)
    coords = []
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(img2)
    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()
    correspondence_pts.append(coords)
    
    np.save('football_pts_'+str(n)+'.npy',correspondence_pts)

def main(n):
    '''
    This function will find the homography matrix and then use it to find corresponding marker in football image 2
    '''
    # reading the images
    img1 = etai.read('football1.jpg')
    img2 = etai.read('football2.jpg')

    filepath = 'football_pts_'+str(n)+'.npy'
    # get n corresponding points
    if not os.path.exists(filepath):
        get_correspondences(img1, img2,n)
    
    correspondence_pts = np.load(filepath)
    
    XY1 = correspondence_pts[0]
    XY2 = correspondence_pts[1]
    # plotting the Fooball image 1 with marker 33
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(img1)
    u=[1210,1701]
    v=[126,939]
    ax.plot(u, v,color='yellow')
    ax.set(title='Football image 1')
    plt.show()
    
    #------------------------------------------------
    # FILL YOUR CODE HERE
    # Your code should estimate the homogrphy and draw the  
    # corresponding yellow line in the second image.
    
    


if __name__ == "__main__": 
    
    #------------------------------------------------
    # FILL BLANK HERE
    # Specify the number of pairs of points you need.
    n = 
    #------------------------------------------------
    main(n)
