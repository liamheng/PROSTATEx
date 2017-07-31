####This code is to prepare the input data for Vgg model
import os
import scipy.ndimage.interpolation
import numpy as np
import scipy.io as sio
import nibabel as nib
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from scipy.misc import toimage

Table = pd.ExcelFile('ProstateX-2-Findings-Test.xls').parse('ProstateX')
Patients=Table.iloc[:,0]
Findings=Table.iloc[:,1]
# Data loading and preprocessing
DataPath='./AugmentData/'


def Rotate(Img, Mask, axes, angle):
	RoatateImg=scipy.ndimage.interpolation.rotate(Img,angle,axes,reshape=False,mode='reflect', prefilter=False)
	RoatateMask=scipy.ndimage.interpolation.rotate(Mask,angle,axes,reshape=False,order=0, prefilter=False)
	Location=np.where((RoatateMask>0.9)&(RoatateMask<1.1))
	counts = np.bincount(Location[2])
	Zslice=np.argmax(counts)
	# slices=np.argmax(counts)+[-1,0,1]
	# slices=slices+(slices.min()<0)*1-(1+slices.min()>RoatateMask.shape[2])*1
	# input=[]
	# for Zslice in slices: 
	Shape=np.where((RoatateMask[:,:,Zslice]>19.9)&(RoatateMask[:,:,Zslice]<20.1))
	#print(Shape)
	ROI = RoatateImg[Shape[0].min():Shape[0].max()+1,Shape[1].min():Shape[1].max()+1,Zslice]
	scale=np.divide(np.array([128,128], dtype=float), np.array(ROI.shape, dtype=float))#scale 1
	input=scipy.ndimage.interpolation.zoom(ROI,scale)	
	return np.array(input)


def AugmentData(ImagePath,MaskPath):
	Img=nib.load(ImagePath).get_data()
	Mask=nib.load(MaskPath).get_data()
	real_resize_factor=np.array(Mask.shape)/np.array(Img.shape)
	Img = scipy.ndimage.interpolation.zoom(Img, real_resize_factor, order=3, mode='nearest', prefilter=False)
	Augment=[]
	Augment.append(Rotate(Img, Mask, axes=(0,1), angle=0))
	for axes in [(0,1),(0,2),(1,2)]:
		for angle in [-5,-4,-3,-2,-1,1,2,3,4,5]:
			Augment.append(Rotate(Img, Mask, axes, angle))
	# Augment.append(Rotate(Img, Mask, axes=(0,1), angle=5))[-5,-4,-3,-2,-1,1,2,3,4,5]
	# Augment.append(Rotate(Img, Mask, axes=(0,2), angle=-5))
	# Augment.append(Rotate(Img, Mask, axes=(0,2), angle=5))
	# Augment.append(Rotate(Img, Mask, axes=(1,2), angle=-5))
	# Augment.append(Rotate(Img, Mask, axes=(1,2), angle=5))	
	return np.array(Augment)#.reshape(57,224,224)
	
Data=[]
for ii in range(5):
	for Patient,Finding in zip(Patients,Findings):
		ADCPath=os.path.join(DataPath,Patient+'_ADC_'+str(Finding)+'_Aug'+str(ii)+'.nii.gz')
		ADCMaskPath=os.path.join(DataPath,Patient+'_Mask_'+str(Finding)+'.nii.gz')
		ADC=AugmentData(ADCPath,ADCMaskPath)
		TraPath=os.path.join(DataPath,Patient+'_Tra_'+str(Finding)+'_Aug'+str(ii)+'.nii.gz')
		TraMaskPath=os.path.join(DataPath,Patient+'_Mask_'+str(Finding)+'.nii.gz')
		Tra=AugmentData(TraPath,TraMaskPath)	
		BVALPath=os.path.join(DataPath,Patient+'_BVAL_'+str(Finding)+'_Aug'+str(ii)+'.nii.gz')
		BVALMaskPath=os.path.join(DataPath,Patient+'_Mask_'+str(Finding)+'.nii.gz')
		BVAL=AugmentData(BVALPath,BVALMaskPath)		
		KtransPath=os.path.join(DataPath,Patient+'_Ktrans_'+str(Finding)+'_Aug'+str(ii)+'.nii.gz')
		KtransMaskPath=os.path.join(DataPath,Patient+'_Mask_'+str(Finding)+'.nii.gz')	
		Ktrans=AugmentData(KtransPath,KtransMaskPath)
		ADC=ADC.reshape(ADC.shape+(1,))
		Ktrans=Ktrans.reshape(Ktrans.shape+(1,))
		Tra=Tra.reshape(Tra.shape+(1,))
		BVAL=BVAL.reshape(BVAL.shape+(1,))
		X=np.concatenate((Tra,ADC,BVAL,Ktrans),axis=3).astype('int32')#combine 3 channels
		for nn in range(X.shape[0]):
			toimage(X[nn], high=np.max(X[nn]), low=np.min(X[nn])).save(os.path.join('AugPNG',Patient+'_'+str(Finding)+'_Aug'+str(ii)+'_'+str(nn)+'.png'))
	# Read=scipy.misc.imread('./RegionGrow/ProstateX-0001_1_1.png')
	# Data.append(X)
	
# np.savez('./Region2DData', Data=Data)	
	# img = Image.fromarray(X[0], 'RGB')
	# plt.subplot(221)
	# plt.imshow(X[0,:,:,0], cmap='gray')
	# plt.subplot(222)
	# plt.imshow(X[0,:,:,1], cmap='gray')
	# plt.subplot(223)
	# plt.imshow(X[0,:,:,2], cmap='gray')
	# plt.subplot(224)
	# plt.imshow(np.int_(X[0]))
	# plt.show()
# plt.imsave('test.png', X[0])