'''
This script performs resolution normalization and histogram matching for image data.
Author: Heng Li
Email: lihengbit@foxmail.com
Date: 06/20/2017
You are free to modify and redistribute the script.
'''
import os
import scipy.ndimage.interpolation
import numpy as np
import scipy.io as sio
import nibabel as nib
import pandas as pd
import scipy.ndimage

Table = pd.ExcelFile('ProstateX-2-Findings-Test.xls').parse('ProstateX')
Patients=Table.iloc[:,0]
Findings=Table.iloc[:,1]
# Labels=Table.iloc[:,4]
# Data loading and preprocessing
DataPath='./NormData/'


def resample(image, scan, new_spacing=[0.5,0.5,0.5],order=3):
    # Determine current pixel spacing
    spacing = np.array((scan[0],scan[1],scan[2]), dtype=np.float32)
    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor
    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, order=order, mode='nearest', prefilter=False)
    return image, new_spacing

def hist_match(sourcePath, templatePath, DataPath):
    source = nib.load(os.path.join(DataPath,sourcePath)).get_data()
    template = nib.load(os.path.join(DataPath,templatePath)).get_data()
    oldshape = source.shape
    source = source.ravel()
    template = template.ravel()
    # get the set of unique pixel values and their corresponding indices and
    # counts
    s_values, bin_idx, s_counts = np.unique(source, return_inverse=True,
                                            return_counts=True)
    t_values, t_counts = np.unique(template, return_counts=True)
    # take the cumsum of the counts and normalize by the number of pixels to
    # get the empirical cumulative distribution functions for the source and
    # template images (maps pixel value --> quantile)
    s_quantiles = np.cumsum(s_counts).astype(np.float64)
    s_quantiles /= s_quantiles[-1]
    t_quantiles = np.cumsum(t_counts).astype(np.float64)
    t_quantiles /= t_quantiles[-1]
    # interpolate linearly to find the pixel values in the template image
    # that correspond most closely to the quantiles in the source image
    interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)
    nib.save(nib.Nifti1Image(interp_t_values[bin_idx].reshape(oldshape), nib.load(os.path.join(DataPath,sourcePath)).affine),os.path.join(DataPath, sourcePath))
    # return interp_t_values[bin_idx].reshape(oldshape)	
	

def Preprocess(ImagePath,MaskPath, DataPath):
	ImgInfo=nib.load(os.path.join(DataPath,ImagePath))
	Img=ImgInfo.get_data()
	Mask=nib.load(os.path.join(DataPath,MaskPath)).get_data()	
	Shape=np.where(Mask==20)
	input=Img[Shape[0].min():Shape[0].max()+1,Shape[1].min():Shape[1].max()+1,Shape[2].min():Shape[2].max()+1]
	NormalImage,_=resample(input,ImgInfo.affine[:,3][:3])
	inputMask=Mask[Shape[0].min():Shape[0].max()+1,Shape[1].min():Shape[1].max()+1,Shape[2].min():Shape[2].max()+1]
	NormalMask,_=resample(inputMask,ImgInfo.affine[:,3][:3],order=0)
	ImgInfo.affine[0:3,3]=0.5
	ImgInfo.affine[0,0]=0.5
	ImgInfo.affine[1,1]=0.5
	ImgInfo.affine[2,2]=0.5
	NormalImage[NormalImage<0]=0
	nib.save(nib.Nifti1Image(np.int_(NormalImage), ImgInfo.affine),os.path.join('./NormTABK', ImagePath[:-7]+MaskPath[-9:]))
	nib.save(nib.Nifti1Image(np.int_(NormalMask), ImgInfo.affine),os.path.join('./NormTABK', MaskPath))
	
#####Normal Intensity
# for Patient in Patients:
	# TraPath=Patient+'_Tra.nii.gz'
	# hist_match(TraPath,'ProstateX-0000_Tra.nii.gz',DataPath)


	
	
######ROI
for Patient,Finding in zip(Patients,Findings):
	ADCPath=Patient+'_ADC.nii.gz'
	ADCMaskPath=Patient+'_Tra_Mask_'+str(Finding)+'.nii.gz'
	Preprocess(ADCPath,ADCMaskPath,'NormData')
	TraPath=Patient+'_Tra.nii.gz'
	TraMaskPath=Patient+'_Tra_Mask_'+str(Finding)+'.nii.gz'
	Preprocess(TraPath,TraMaskPath,'NormData')
	BVALPath=Patient+'_BVAL.nii.gz'
	BVALMaskPath=Patient+'_Tra_Mask_'+str(Finding)+'.nii.gz'
	Preprocess(BVALPath,BVALMaskPath,'NormData')		
	KtransPath=Patient+'_Ktrans.nii.gz'
	KtransMaskPath=Patient+'_Tra_Mask_'+str(Finding)+'.nii.gz'	
	Preprocess(KtransPath,KtransMaskPath,'NormData')

