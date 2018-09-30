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

def calc_homography(n):
    pts = np.load('football_pts_'+str(n)+'.npy')
    print('POINTS:', pts, '\n')
    print(('Points.shape', pts.shape, '\n'))
    print('Points.shape[0]', pts.shape[0], '\n')
    print('Points[0,0]', pts[0,0], '\n')

    X = pts[0,0,0]
    print ('X1:', X, "\n")


    A = np.zeros((2*pts.shape[1], 9))
    for m in range(pts.shape[1]):
        print('m: ', m)
        (x,y) = pts[0,m]
        (xs,ys) = pts[1,m]
        A[2*m,:] = [x,y,1,0,0,0,-x*xs,-y*xs,-xs]
        A[2*m+1,:] = [0,0,0,x,y,1,-x*ys,-y*ys,-ys]

    np.set_printoptions(precision=3)
    print(A)

    U,S,Vh = np.linalg.svd(A, compute_uv=True)

    print('Vh.shape:', Vh.shape)

    x = Vh[-1,:]/ np.linalg.norm(Vh[-1,:])
    print('x:', x)

    H = np.reshape(x,(3,3))
    print(H)
    return H


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
    print('XY1: ', XY1, '\n')
    XY2 = correspondence_pts[1]
    print('XY2: ', XY2, '\n')
    # plotting the Fooball image 1 with marker 33
    fig = plt.figure()
    ax = fig.add_subplot(111)#
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

    H = calc_homography(n)

    fig = plt.figure()
    ax = fig.add_subplot(111)#
    ax.imshow(img2)
    u=[1210,1701]
    v=[126,939]
    #transform the points
    u_t = np.zeros_like(u)
    v_t = np.zeros_like(u)

    test = np.matmul(H ,np.array([u[0], v[0], 1]))
    print('test:', test/test[-1])

    [temp_u,temp_v,w] = np.matmul(H,np.array([u[0], v[0], 1]))
    (u_t[0],v_t[0]) = (temp_u/w, temp_v/w) 

    [temp_u,temp_v,w] = np.matmul(H,np.array([u[1], v[1], 1]))
    (u_t[1],v_t[1]) = (temp_u/w, temp_v/w) 

    ax.plot(u_t, v_t, color='yellow')
    ax.set(title='Football image 2')
    plt.show()
    


if __name__ == "__main__": 
    
    #------------------------------------------------
    # FILL BLANK HERE
    # Specify the number of pairs of points you need.
    n = 4
    #------------------------------------------------
    main(n)
