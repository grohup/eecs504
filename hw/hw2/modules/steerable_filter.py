#!/usr/bin/env python
'''
A module for applying a steerable filter on an image.

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

from eta.core.config import Config
import eta.core.image as etai
import eta.core.module as etam
import matplotlib.pyplot as plt



class SteerableFilterConfig(etam.BaseModuleConfig):
    '''Steerable filter configuration settings.

    Attributes:
        data (DataConfig)
    '''

    def __init__(self, d):
        super(SteerableFilterConfig, self).__init__(d)
        self.data = self.parse_object_array(d, "data", DataConfig)


class DataConfig(Config):
    '''Data configuration settings.

    Inputs:
        sobel_horizontal_result (eta.core.types.NpzFile): The result of
            convolving the original image with the "sobel_horizontal" kernel.
            This will give the value of Gx (the gradient in the x
            direction).
        sobel_vertical_result (eta.core.types.NpzFile): The result of
            convolving the original image with the "sobel_vertical" kernel.
            This will give the value of Gy (the gradient in the y
            direction).
    Outputs:
        filtered_image (eta.core.types.ImageFile): The output image after
            applying the steerable filter.
    '''

    def __init__(self, d):
        self.sobel_horizontal_result = self.parse_string(
            d, "sobel_horizontal_result")
        self.sobel_vertical_result = self.parse_string(
            d, "sobel_vertical_result")
        self.filtered_image = self.parse_string(
            d, "filtered_image")


def _apply_steerable_filter(Gx, Gy):
    '''Applies the steerable filter on the given image, using the results
    from sobel kernel convolution.

    Args:
        G_x: the x derivative of the image, given as the result of convolving
            the input image with the horizontal sobel kernel
        G_y: the y derivative of the image, given as the result of convolving
            the input image with the vertical sobel kernel

    Returns:
        g_intensity: the intensity of the input image, defined as
            sqrt(Gx^2 + Gy^2), at every line that does not lie on the
            x-axis or y-axis
    '''
    # TODO
    # REPLACE THE CODE BELOW WITH YOUR IMPLEMENTATION
    g_int = np.zeros_like(Gx)
    g_orient = np.zeros_like(Gx)

    #Create orientation matrix
    g_int = np.sqrt(Gx**2 + Gy**2)
    g_orient = np.arctan2(Gy,Gx)

    #Find pixels where orienation is up, down, left, right
    g_orient  = g_orient*360/(2*np.pi)
    
    #normalize range of angles to [0,pi] from [-pi,pi]
    g_orient[g_orient<0] = g_orient[g_orient<0] + 180
    g_orient = np.round(g_orient)
    #g_orient = (2* np.round(g_orient/2)).astype(int)

    print(g_orient[385,613])
    plt.subplot(211)
    plt.imshow(g_int, cmap='gray')
    plt.subplot(212)
    g_int[g_orient == 0] = 0
    g_int[g_orient == 90] = 0
    g_int[g_orient == 180] = 0
    plt.imshow(g_int, cmap='gray')
    plt.show()
    return g_int


def _filter_image(steerable_filter_config):
    for data in steerable_filter_config.data:
        sobel_horiz = np.load(data.sobel_horizontal_result)["filtered_matrix"]
        sobel_vert = np.load(data.sobel_vertical_result)["filtered_matrix"]
        filtered_image = _apply_steerable_filter(sobel_horiz, sobel_vert)
        etai.write(filtered_image, data.filtered_image)


def run(config_path, pipeline_config_path=None):
    '''Run the Steerable Filter module.

    Args:
        config_path: path to a SteerableFilterConfig file
        pipeline_config_path: optional path to a PipelineConfig file
    '''
    steerable_filter_config = SteerableFilterConfig.from_json(config_path)
    etam.setup(steerable_filter_config,
               pipeline_config_path=pipeline_config_path)
    _filter_image(steerable_filter_config)


if __name__ == "__main__":
    run(*sys.argv[1:])
