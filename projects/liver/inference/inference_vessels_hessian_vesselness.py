# Run with:
# python projects/liver/inference/inference_vessels_hessian_vesselness.py

import argparse
import sys
import itk
import os
from distutils.version import StrictVersion as VS
if VS(itk.Version.GetITKVersion()) < VS("5.0.0"):
    print("ITK 5.0.0 or newer is required.")
    sys.exit(1)

vessels_scardapane_dir = 'H:\\Datasets\\Liver\\LiverScardapaneNew\\nii'
input_image_name = os.path.join(vessels_scardapane_dir, 'ct_scan_00.nii.gz')
output_image_name = os.path.join(vessels_scardapane_dir, 'ct_pred_00.nii.gz')

sigma=1.0
alpha1=0.5
alpha2=2.0

input_image = itk.imread(input_image_name, itk.ctype('float'))

hessian_image = itk.hessian_recursive_gaussian_image_filter(input_image, sigma=sigma)

vesselness_filter = itk.Hessian3DToVesselnessMeasureImageFilter[itk.ctype('float')].New()
vesselness_filter.SetInput(hessian_image)
vesselness_filter.SetAlpha1(alpha1)
vesselness_filter.SetAlpha2(alpha2)

itk.imwrite(vesselness_filter, output_image_name)