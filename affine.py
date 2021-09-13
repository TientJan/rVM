# -*- coding: utf-8 -*-
"""
Created on Sat Aug  4 22:54:09 2018
@author: Fakrul-IslamTUSHAR
"""

##################import Libraries##################################
import SimpleITK as sitk
import matplotlib.pyplot as plt
from glob import glob

from datetime import timedelta

"""------------------------------"""
import time
import os
import sys

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--option", type=str, dest="option", default='affine')
args = parser.parse_args()

class Logger(object):

    def __init__(self, stream = sys.stdout):
        output_dir = "log"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        log_name = '{}.log'.format(time.strftime('%Y-%m-%d-%H-%M'))
        filename = os.path.join(output_dir, log_name)

        self.terminal = stream
        self.log = open(filename, 'a+')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

sys.stdout = Logger(sys.stdout)  #  将输出记录到log
sys.stderr = Logger(sys.stderr)  # 将错误信息记录到log

"""--------------------------------"""

#########################Import Libaries#############################
start_time = time.time()

# =============================================================================
# Function Definitions
# =============================================================================

# Callback invoked by the interact IPython method for scrolling through the image stacks of
# the two images (moving and fixed).

#############################Functions Done####################################

def read(path):
    return sitk.ReadImage(path, sitk.sitkInt16)

paths = glob("../../Dataset/RESECT/resize/train/*")


for path in paths:
    t1path = glob(path+"/*T1.nii")[0]
    print(t1path)
    flairpath = glob(path + "/*FLAIR.nii")[0]
    nt1path = t1path.replace("resize","resize_affine")
    nflairpath = flairpath.replace("resize", "resize_affine")
    t1dir = os.path.split(nt1path)[0]
    if not os.path.exists(t1dir):
        os.makedirs(t1dir)
    moving_image = read(t1path)
    fixed_image = read(flairpath)
    print(fixed_image.GetSize())
    print(t1dir)

    """  --预处理 --"""
    start_time = time.time()
    registration_method = sitk.ImageRegistrationMethod()

    if args.option == "affine":
    # pre_method = sitk.ImageRegistrationMethod()
        ## 1 pre_transform = sitk.CenteredTransformInitializer(fixed_image, moving_image, sitk.Similarity3DTransform())
        # pre_method.SetInitialTransform(sitk.CenteredTransformInitializer(fixed_image, moving_image, sitk.Similarity3DTransform()))
        # pre_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins = 100)
        # pre_method.SetMetricSamplingStrategy(pre_method.RANDOM)
        # pre_method.SetMetricSamplingPercentage(0.3)

        # pre_method.SetInterpolator(sitk.sitkLinear)

        # pre_method.SetOptimizerAsGradientDescent(learningRate = 1.5, numberOfIterations = 10,
        #                                                   convergenceMinimumValue = 1e-5,
        #                                                   convergenceWindowSize = 6)
        # pre_method.SetOptimizerScalesFromPhysicalShift()

        # pre_transform = pre_method.Execute(sitk.Cast(fixed_image, sitk.sitkFloat32),
        #                                               sitk.Cast(moving_image, sitk.sitkFloat32))

        ## 2 moving_pre = sitk.Resample(moving_image, fixed_image, pre_transform, sitk.sitkLinear, 0.2,
                                        # moving_image.GetPixelID())

        ##Name the Image
        # =============================================================================
        # Registartion Start
        # =============================================================================
        # registration Method.
        
        initial_transform = sitk.AffineTransform(fixed_image.GetDimension())
        registration_method.SetInitialTransform(initial_transform)

        moving_pre = moving_image ## 如果把 1、2 恢复，这里应该注释掉

    if args.option == 'bspline':
        # Determine the number of BSpline control points using the physical spacing we want for the control grid.
        #############################Initializing Initial Transformation##################################
        grid_physical_spacing = [100.0, 100.0, 100.0]  # A control point every 50mm
        image_physical_size = [size * spacing for size, spacing in zip(fixed_image.GetSize(), fixed_image.GetSpacing())]
        mesh_size = [int(image_size / grid_spacing + 0.5) \
                    for image_size, grid_spacing in zip(image_physical_size, grid_physical_spacing)]
        initial_transform = sitk.BSplineTransformInitializer(image1 = fixed_image, transformDomainMeshSize = mesh_size, order=2)
        
        registration_method.SetInitialTransform(initial_transform)

        # moving_pre = sitk.Resample(moving_image, fixed_image, pre_transform, sitk.sitkLinear, 0.2,
        #                                 moving_image.GetPixelID())

        ##################Multi-resolution framework############3
        registration_method.SetShrinkFactorsPerLevel(shrinkFactors = [4, 2, 1])
        registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas = [2, 1, 0])
        registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

        moving_pre = moving_image


    #######################Matrix###################################################3
    # registration_method.SetMetricAsMeanSquares()
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins = 100)
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.3)

    ##############Interpolation#################################
    registration_method.SetInterpolator(sitk.sitkLinear)

    ##################Optimizer############################
    # registration_method.SetOptimizerAsLBFGSB(gradientConvergenceTolerance=1e-5, numberOfIterations=10)
    registration_method.SetOptimizerAsGradientDescent(learningRate = 1.5, numberOfIterations = 10,
                                                    convergenceMinimumValue = 1e-5,
                                                    convergenceWindowSize = 6)
    registration_method.SetOptimizerScalesFromPhysicalShift()


    #######################################Print Comment#############################################
    # # Connect all of the observers so that we can perform plotting during registration.
    # registration_method.AddCommand(sitk.sitkStartEvent, start_plot)
    # registration_method.AddCommand(sitk.sitkEndEvent, end_plot)
    # registration_method.AddCommand(sitk.sitkMultiResolutionIterationEvent, update_multires_iterations)
    # registration_method.AddCommand(sitk.sitkIterationEvent, lambda: plot_values(registration_method))

    #################Transformation###################################################################
    final_transform = registration_method.Execute(sitk.Cast(fixed_image, sitk.sitkFloat32),
                                                  sitk.Cast(moving_pre, sitk.sitkFloat32))

    # =============================================================================
    # post processing Analysis
    # =============================================================================
    print('Final metric value: {0}'.format(registration_method.GetMetricValue()))
    print('Optimizer\'s stopping condition, {0}'.format(registration_method.GetOptimizerStopConditionDescription()))
    # Visualize Expected Results
    moving_resampled = sitk.Resample(moving_pre, fixed_image, final_transform, sitk.sitkLinear, 0.2,
                                     moving_pre.GetPixelID())

    ################Saving Transformed images###################################33
    sitk.WriteImage(moving_resampled, nt1path)
    sitk.WriteImage(fixed_image, nflairpath)
    # sitk.WriteImage(moving_resampled2,Registered_imageName+'_two' +'.nii.gz')
    # sitk.WriteImage(moving_resampled3,Registered_imageName +'_three'+'.nii.gz')

    elapsed_time_secs = time.time() - start_time
 
    msg = "Execution took: %s secs (Wall clock time)" % timedelta(seconds=round(elapsed_time_secs))

    print(msg)
 
