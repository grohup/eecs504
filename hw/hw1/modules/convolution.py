#!/usr/bin/env python
'''
A module for convolving two 2d images.

This module convolves an RGB or grayscale image with a kernel specified
by the user.

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
# pragma pylint: enable=redefined-builtin
# pragma pylint: enable=unused-wildcard-import
# pragma pylint: enable=wildcard-import

import sys

import numpy as np

from eta.core.config import Config, ConfigError
import eta.core.utils as etau
import eta.core.image as etai
import eta.core.module as etam
import matplotlib.pyplot as plt


class ConvolutionConfig(etam.BaseModuleConfig):
    '''Convolution configuration settings.

    Attributes:
        data (DataConfig)
        parameters (ParametersConfig)
    '''

    def __init__(self, d):
        super(ConvolutionConfig, self).__init__(d)
        self.data = self.parse_object_array(d, "data", DataConfig)
        self.parameters = self.parse_object(d, "parameters", ParametersConfig)


class DataConfig(Config):
    '''Data configuration settings.

    Inputs:
        input_image (eta.core.types.Image): The input image

    Outputs:
        filtered_image (eta.core.types.ImageFile): [None] The result of
            convolution, stored in an image (all values are changed to fit
            between 0 and 255)
        filtered_matrix (eta.core.types.NpzFile): [None] The result of
            convolution, stored in a matrix (all values, including their
            signs, are maintained)
    '''

    def __init__(self, d):
        self.input_image = self.parse_string(d, "input_image")
        self.filtered_image = self.parse_string(
            d, "filtered_image", default=None)
        self.filtered_matrix = self.parse_string(
            d, "filtered_matrix", default=None)


class ParametersConfig(Config):
    '''Parameter configuration settings.

    Parameters:
        kernel_type (eta.core.types.String): ["x_derivative"] The type of
            kernel used for convolution.
        image_type (eta.core.types.String): ["color"] The format required
            for convolution with the specified kernel. The image type
            must be grayscale when applying the Sobel kernel.
        gaussian_sigma (eta.core.types.Number): [1] The sigma used when
            creating the gaussian kernel
    '''

    def __init__(self, d):
        self.kernel_type = self.parse_string(
            d, "kernel_type", default="Potts")
        self.image_type = self.parse_string(
            d, "image_type", default="color")
        self.gaussian_sigma = self.parse_number(
            d, "gaussian_sigma", default=1)
        self._validate()

    def _validate(self):
        possible_kernels = ["x_derivative", "y_derivative",
                            "sobel_horizontal", "sobel_vertical",
                            "gaussian"]
        possible_types = ["color", "grayscale"]
        if self.kernel_type not in possible_kernels:
            raise ConfigError("kernel_type %s is not supported",
                              self.kernel_type)
        if self.image_type not in possible_types:
            raise ConfigError("image_type %s is not supported",
                              self.image_type)


def _create_x_derivative_kernel():
    '''Creates a kernel that calculates the discrete x-derivative
    of a 2-d image.

    Returns:
        kernel: the x-derivative kernel
    '''
    return np.array([[-1,1]])


def _create_y_derivative_kernel():
    '''Creates a kernel that calculates the discrete y-derivative
    of a 2-d image.

    Returns:
        kernel: the y-derivative kernel
    '''
    return np.transpose(np.array([[-1,1]]))

def _create_sobel_horizontal_kernel():
    return np.array([[1,0,-1],[2,0,-2],[1,0,-1]])


def _create_sobel_vertical_kernel():
    '''Creates the 3x3 vertical sobel kernel.

    Returns:
        kernel: the sobel vertical kernel   
    '''
    return np.array([[1,2,1],[0,0,0],[-1,-2,-1]])


def _create_gaussian_kernel(sigma):
    '''Creates a Gaussian kernel.

    Returns:
        func_2d: a 2-D Gaussian kernel
    '''

    x_space = np.linspace(-1, 1, num=15)
    y_space = np.linspace(-1, 1, num=15)
    x_func = np.exp(-(x_space ** 2) / (2 * sigma ** 2))
    y_func = np.exp(-(y_space ** 2) / (2 * sigma ** 2))
    func_2d = y_func[:, np.newaxis] * x_func[np.newaxis, :]
    func_2d = (1 / func_2d.sum()) * func_2d
    return func_2d


def _convolve(kernel, in_img):
    '''Convolve the input image "in_img" with the kernel "kernel".
    Assume the kernel has already been flipped.

    Args:
        kernel: a 2d kernel that is assumed to be flipped appropriately
        in_img: the input image

    Returns:
        out_img: the result of convolving the input image with the specified
            kernel
    '''
    # Get some sizes
    m1, n1 = in_img.shape
    m2 = kernel.shape[0]
    n2 = kernel.shape[1]

    # Pad Image
    pad = int(max(m2,n2))//2 #3x3//2 = 1, 5x5//2=2 width, 15x15//2 = 7
    in_img_pad = np.zeros((m1+2*pad,n1+2*pad))
    in_img_pad[pad:(m1+pad),pad:(n1+pad)] = in_img
    print("padded image shape", in_img_pad.shape)
    out_img = np.zeros((m1-pad,n1-pad))


    # Perform Convolution
    for (x,y) in np.ndindex(int(m1-pad),int(n1-pad)):
        #out_img[x,y]=(kernel*in_img_pad[x:x+m2,y:y+n2]).sum()
        out_img[x,y]= np.dot(np.reshape(kernel,(m2*n2)),np.reshape((in_img_pad[x:x+m2,y:y+n2]),(m2*n2)))
    
    """for ind in np.ndindex(m1,n):
        for ind2 in np.ndindex(m2,n2):
            in_ind = tuple(np.subtract(ind,ind2) + ([pad,pad]))
            #print("index1: ",  ind, "index2: ", ind2, "index - ind2 + pad", in_ind)
            out_img[ind] = out_img[ind] + (in_img_pad[in_ind] * kernel[ind2])"""
            
    '''
    # Pad Kernel to output matrix size
    kern = np.zeros([m,n])
    kern[0:kernel.shape[0], 0:kernel.shape[1]] = kernel

    # Perform Convolution not the way to do it
    for ind,x in np.ndenumerate(out_img):
        for ind2,x in np.ndenumerate(kernel):
            out_img[ind] = out_img[ind] + in_img[ind2] * kern[tuple(np.subtract(ind,ind2))]
            print("index1: ",  ind, "index2: ", ind2, "index - ind2", np.subtract(ind,ind2)) '''
    return out_img


def _perform_convolution(convolution_config):
    '''Performs convolution of an input image with a kernel specified
    by the configuration parameters, and writes the result to the
    path specified by "filtered_image".

    Args:
        convolution_config: the configuration file for the module
    '''
    kernel_type = convolution_config.parameters.kernel_type
    if kernel_type == "x_derivative":
        kernel = _create_x_derivative_kernel()
    elif kernel_type == "y_derivative":
        kernel = _create_y_derivative_kernel()
    elif kernel_type == "sobel_vertical":
        kernel = _create_sobel_vertical_kernel()
    elif kernel_type == "sobel_horizontal":
        kernel = _create_sobel_horizontal_kernel()
    else:
        # this will be the Gaussian kernel
        kernel = _create_gaussian_kernel(
                    convolution_config.parameters.gaussian_sigma)

    for data in convolution_config.data:
        in_img = etai.read(data.input_image)
        if convolution_config.parameters.image_type == "grayscale":
            in_img = etai.rgb_to_gray(in_img)
        else:
            # if the image should be a color image, convert the grayscale
            # image to color (simply converts the image into a 3-channel
            # image)
            if etai.is_gray(in_img):
                in_img = etai.gray_to_rgb(in_img)

        filtered_image = _convolve(kernel, in_img)
        #plt.imshow(filtered_image, cmap='gray')
        #plt.show()
        if data.filtered_matrix:
            etau.ensure_basedir(data.filtered_matrix)
            np.savez(data.filtered_matrix, filtered_matrix=filtered_image)
        if data.filtered_image:
            etau.ensure_basedir(data.filtered_image)
            etai.write(filtered_image, data.filtered_image)


def run(config_path, pipeline_config_path=None):
    '''Run the convolution module.

    Args:
        config_path: path to a ConvolutionConfig file
        pipeline_config_path: optional path to a PipelineConfig file
    '''
    convolution_config = ConvolutionConfig.from_json(config_path)
    etam.setup(convolution_config, pipeline_config_path=pipeline_config_path)
    _perform_convolution(convolution_config)


if __name__ == "__main__":
    run(*sys.argv[1:])
