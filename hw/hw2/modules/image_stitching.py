#!/usr/bin/env python
'''
A module for stitching two parts of an image into one whole image.

Info:
    type: eta.core.types.Module
    version: 0.1.0
'''
# pragma pylint: disable=redefined-builtin
# pragma pylint: disable=unused-wildcard-import
# pragma pylint: disable=wildcard-import
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from builtins import *
from collections import defaultdict
# pragma pylint: enable=redefined-builtin
# pragma pylint: enable=unused-wildcard-import
# pragma pylint: enable=wildcard-import

import sys

import numpy as np
import cv2

from eta.core.config import Config
import eta.core.image as etai
import eta.core.module as etam
import matplotlib.pyplot as plt


class ImageStitchingConfig(etam.BaseModuleConfig):
    '''Image stitching configuration settings.

    Attributes:
        data (DataConfig)
        parameters (ParametersConfig)
    '''

    def __init__(self, d):
        super(ImageStitchingConfig, self).__init__(d)
        self.data = self.parse_object_array(d, "data", DataConfig)
        self.parameters = self.parse_object(d, "parameters", ParametersConfig)


class DataConfig(Config):
    '''Data configuration settings.

    Inputs:
        corner_locs_1 (eta.core.types.NpzFile): An Nx2 matrix
            containing (x,y) locations of all corners in image 1,
            detected by the Harris Corner algorithm
        corner_locs_2 (eta.core.types.NpzFile): A Mx2 matrix
            containing (x,y) locations of all corners in image 2,
            detected by the Harris Corner algorithm
        image_1 (eta.core.types.Image): the first input image
        image_2 (eta.core.types.Image): the second input image
    Outputs:
        stitched_image (eta.core.types.ImageFile): The final stitched image
    '''

    def __init__(self, d):
        self.corner_locs_1 = self.parse_string(d, "corner_locs_1")
        self.corner_locs_2 = self.parse_string(d, "corner_locs_2")
        self.image_1 = self.parse_string(d, "image_1")
        self.image_2 = self.parse_string(d, "image_2")
        self.stitched_image = self.parse_string(d, "stitched_image")


class ParametersConfig(Config):
    '''Parameter configuration settings.

    Parameters:
        no_correspondence (eta.core.types.Number): [4] the number of
            points to use when computing the homography
    '''

    def __init__(self, d):
        self.no_correspondence = self.parse_number(
            d, "no_correspondence", default=4)


def _get_HOG_descriptors(corner_locs, in_img):
    '''Return a MxN matrix that contains the M-dimensional HOG feature vectors
        for all N corners.

    Args:
        corner_locs: the location of Harris corners, given as a
            Nx2 2-dimensional matrix
        in_img: the input image

    Returns:
        hog_features: a N x 3780 matrix containing HOG feature vectors for every
            detected corner
    '''
    # Setting parameters
    win_size = (64,128)
    blockSize = (16,16)
    blockStride = (8,8)
    cellSize = (8,8)
    nbins = 9
    derivAperture = 1
    winSigma = 4.
    histogramNormType = 0
    L2HysThreshold = 2.0000000000000001e-01
    gammaCorrection = 0
    nlevels = 64

    # Initializing descriptor
    hog = cv2.HOGDescriptor(win_size,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,
                            histogramNormType,L2HysThreshold,gammaCorrection,nlevels)

    # Setting compute parameters
    win_stride = (8,8)
    padding = (8,8)
    new_locations = []

    # Gathering all corner locations
    # NOTE: This will not work until you successfully implement the Harris
    #       Corner Detector in 'modules/harris.py'.
    for i in range(corner_locs.shape[0]):
        new_locations.append((int(corner_locs[i][0]), int(corner_locs[i][1])))
    N = len(new_locations)

    # Computing HOG feature vectors for all corners and concatenating them
    # together
    hog_descrp = hog.compute(in_img, win_stride, padding, new_locations)
    feat_size = int((((win_size[0] / 8) - 1) * ((win_size[1] / 8) - 1)) * 36)
    hog_features = np.asarray(hog_descrp)

    # Reshaping as a N x 3780 array
    hog_features = np.reshape(hog_descrp,(N,feat_size))

    return hog_features


def _match_keypoints(hog_features_1, hog_features_2, img1_corners, img2_corners, img1, img2):
    '''Match the HOG features of the two images and return a list of matched
    keypoints.

    Args:
        hog_features_1: the HOG features for the first image
        hog_features_2: the HOG features for the second image
        img1_corners: the corners detected in the first image, from which the
            HOG features were computed
        img2_corners: the corners detected in the second image, from which the
            HOG feautures were computed
        img1: the first image, in case you want to visualize the matches.
        img2: the second image, in case you want to visualize the matches

    Returns:
       img1_matched_points: a list of corner locations in the first image that
            match with those in the second image
       img2_matched_points: a list of corner locations in the second image that
            match with those in the first image
    '''
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

    matches = bf.match(hog_features_1,hog_features_2)

    # Sort them in the order of their distance.
    matches = sorted(matches, key = lambda x:x.distance)
    img1_matched_pts = []
    img2_matched_pts = []
    for match in matches:
        img1_matched_pts.append(img1_corners[match.queryIdx])
        img2_matched_pts.append(img2_corners[match.trainIdx])

    '''
    # Draw the first 20 matches (the blue dots are the matches)
    out_img = img1.copy()
    out_img_2 = img2.copy()
    for i in range(20):
        cv2.circle(out_img,
                   (img1_matched_pts[i][0], img1_matched_pts[i][1]),
                   4,
                   (0, 0, 255),
                   -1)
    for i in range(20):
        cv2.circle(out_img_2,
                   (img2_matched_pts[i][0], img2_matched_pts[i][1]),
                   4,
                   (0, 0, 255),
                   -1)
    # The images below will be stored in your current working directory
    etai.write(out_img, "out1.png")
    etai.write(out_img_2, "out2.png")
    '''

    return img1_matched_pts, img2_matched_pts


def _get_homography(img1_keypoints, img2_keypoints):
    '''Calculate the homography matrix that relates the first image with the
    second image, using the matched keypoints.

    Args:
        img1_keypoints: a list of matched keypoints in image 1. The number
            of keypoints is indicated by the parameter, 'no_correspondence'.
        img2_keypoints: a list of matched keypoints in image 2. The number
            of keypoints is indicated by the parameter, 'no_correspondence'.

    Returns:
        homog_matrix: the homography matrix that relates image 1 and image 2
    '''
    print(img1_keypoints)
    print(img2_keypoints)

    homog_matrix = np.zeros((3,3))

    img1_keypoints = np.asarray(img1_keypoints)
    img2_keypoints = np.asarray(img2_keypoints)

    A = np.zeros((2*img1_keypoints.shape[0], 9))
    for m in range(img1_keypoints.shape[0]):
        
        (y,x) = img1_keypoints[m,:]
        (ys,xs) = img2_keypoints[m,:]

        A[2*m,:] = [x,y,1,0,0,0,-x*xs,-y*xs,-xs]
        A[2*m+1,:] = [0,0,0,x,y,1,-x*ys,-y*ys,-ys]

    U,S,Vh = np.linalg.svd(A, compute_uv=True)

    #grab right singular vector corresponding to smallest SingValue, normalize because numpy
    x = Vh[-1,:]/ np.linalg.norm(Vh[-1,:])

    homog_matrix = np.reshape(x,(3,3))

    return homog_matrix


def _overlap(img1, img2, homog_matrix):
    '''Applies a homography transformation to img2 to stitch img1 and img2
    togther.

    Args:
        img1: the first image
        img2: the second image
        homog_matrix: the homography matrix

    Returns:
        stitched_image: the final stitched image
    '''

    #make sure H is normalized to 1 for h33
    np.set_printoptions(precision=4)
    print(homog_matrix)
    H = np.linalg.inv(homog_matrix)
    H = H/H[2,2]
    print(H)

    row = img2.shape[0]
    col = img2.shape[1]

    #find corners of the image set up as y,x standard
    img2_corners = np.zeros((4,3))
    img2_corners[0,:] = np.transpose(np.matmul(H, [[  [0], [0],    [1]]])).flatten()
    img2_corners[1,:] = np.transpose(np.matmul(H, [[  [0], [col],    [1]]])).flatten()
    img2_corners[2,:] = np.transpose(np.matmul(H, [[[row], [col],  [1]]])).flatten()
    img2_corners[3,:] = np.transpose(np.matmul(H, [[[row], [0],  [1]]])).flatten()
    #renormalize
    for i in range(4):
        img2_corners[i,:] = img2_corners[i,:]/ img2_corners[i,2]
    print(img2_corners)

    #Determine offsets due to negative indices, offset will be zero w/ no negative idices
    offsety = int(min(min(img2_corners[:,0]),0))
    offsetx = int(min(min(img2_corners[:,1]),0))
    print("Offsetx, y = ",offsetx,offsety)

    #find sizes of output image y,x  = row, col = height,width
    width  = int(max(img2_corners[:,1]) - offsetx)
    height = int(max(img2_corners[:,0]) - offsety)
    print("w,h: ", width, height)

    #create empty output image
    img = np.zeros((height+1, width+1, 3))
    interp = np.zeros(3)

    #insert img1
    img[abs(offsety):int(row+abs(offsety)),offsetx:int(col+offsetx),:] = img1

    #insert img2
    #for index in np.ndindex(img.shape[0], img.shape[1]):
    for i in range(offsety, img.shape[0]):
        for j in range(offsetx, img.shape[1]):
            
            ind2 = np.asarray([i,j,1])
            ind_warp = np.matmul(homog_matrix,ind2)# transform 2 to 1
            ind_warp = (ind_warp/ind_warp[2]).astype(int)

            #ind(2,502) should map to indwarp(0,0)
            if ind_warp[1]<img2.shape[1] and ind_warp[1]>-1 and ind_warp[0]<img2.shape[0] and ind_warp[0]>-1:
                # print("ind_warp: ",ind_warp[0], ind_warp[1])
                # print("img index = ", i+abs(offsety),j+offsetx)
                #bilinear interpolation
                ind_floor = np.floor(ind_warp[0:2]).astype(int)
                ind_ceil  = np.floor(ind_warp[0:2]).astype(int)
                
                interpy   = np.asarray([1-(ind_warp[0]%1), (ind_warp[0]%1)])
                interpx   = np.asarray([1-(ind_warp[1]%1),(ind_warp[1]%1)]).T
            
                for c in range(3):
                    im_array = np.asarray([[img2[ind_floor[0],ind_floor[1],c], img2[ind_floor[0], ind_ceil[1],c]],
                                        [img2[ind_ceil[0], ind_floor[1],c], img2[ind_ceil[0] , ind_ceil[1],c]]])
                    interp[c] = np.matmul(np.matmul(interpy,im_array), interpx)
                img[i+abs(offsety),j+offsetx,:] = interp  

            if (i%100== 0 and j%100== 0) :
                print('Shmearing at : ',i,j)
    
    stitched_image = img/255
    plt.imshow(stitched_image)
    plt.show()

    return stitched_image*255


def _stitch_images(image_stitching_config):
    for data in image_stitching_config.data:
        # Load the corner locations
        img1_corners = np.load(data.corner_locs_1)["corner_locations"]
        img2_corners = np.load(data.corner_locs_2)["corner_locations"]

        # Read in the input images
        img1 = etai.read(data.image_1)
        img2 = etai.read(data.image_2)

        # Compute HOG feature vectors for every detected corner
        hog_features_1 = _get_HOG_descriptors(img1_corners, img1)
        hog_features_2 = _get_HOG_descriptors(img2_corners, img2)

        # Match the feature vectors
        img_1_pts, img_2_pts = _match_keypoints(hog_features_1, hog_features_2,
                                    img1_corners, img2_corners, img1, img2)

        # Tune this parameter in "requests/image_stitching_request.json"
        # to specify the number of corresponding points to use when computing
        # the homography matrix
        no_correspondence = image_stitching_config.parameters.no_correspondence

        # Compute the homography matrix that relates image 1 and image 2
        H = _get_homography(img_1_pts[1:no_correspondence+1], img_2_pts[1:no_correspondence+1])

        # Stitching the images by applying the homography matrix to image 2
        final_img = _overlap(img1, img2, H)

        # Write the final stitched image
        etai.write(final_img, data.stitched_image)


def run(config_path, pipeline_config_path=None):
    '''Run the Image Stitching module.

    Args:
        config_path: path to a ImageStitchingConfig file
        pipeline_config_path: optional path to a PipelineConfig file
    '''
    image_stitching_config= ImageStitchingConfig.from_json(config_path)
    etam.setup(image_stitching_config,
               pipeline_config_path=pipeline_config_path)
    _stitch_images(image_stitching_config)


if __name__ == "__main__":
    run(*sys.argv[1:])
