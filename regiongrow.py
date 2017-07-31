'''
This is code for modify the ROI mask. Search the lightest points round the provided tumor center in DTI-BVAL image, then perform watershed to modify ROI mask. Finally get the new tumor center.
Author: Heng Li
Email: lihengbit@foxmail.com
Date: 06/20/2017
You are free to modify and redistribute the script.
'''
import os
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage.morphology import watershed
from skimage.segmentation import random_walker,active_contour
from skimage.feature import peak_local_max
import morphsnakes

Table = pd.ExcelFile('ProstateX-2-Findings-Test.xls').parse('ProstateX')
Patients=Table.iloc[:,0]
Findings=Table.iloc[:,1]
# Labels=Table.iloc[:,4]
DataPath='./NormTABK'

def PatchIntensity(Img,Mask):
	center=np.median(np.array(np.where(Mask==1)),axis=1)
	Upbounder=center+14-np.array(Img.shape)
	if np.max(Upbounder)>0:
		center[Upbounder>0]=np.array(Img.shape)[Upbounder>0]-14
	Downbounder=center-13
	if np.min(Downbounder)<0:
		center[Downbounder<0]=13
	sumpatch=np.zeros((21,21,21))
	for i in range(-3,4):
		for j in range(-3,4):
			for k in range(-3,4):
				patch=Img[center[0]-10+i:center[0]+11+i,center[1]-10+j:center[1]+11+j,center[2]-10+k:center[2]+11+k]
				sumpatch=sumpatch+patch
		# nib.save(nib.Nifti1Image(sumpatch, np.eye(4)),'Patch.nii.gz')		
		move=np.array(np.where(sumpatch==sumpatch.max())).T-[11,11,11]
	return center+move[0]


for Patient,Finding in zip(Patients,Findings):
	ImgPath=os.path.join(DataPath,Patient+'_BVAL_'+str(Finding)+'.nii.gz')
	MaskPath=os.path.join(DataPath,Patient+'_Tra_Mask_'+str(Finding)+'.nii.gz')
	ImgInfo=nib.load(ImgPath)
	Img=ImgInfo.get_data()
	Img= (Img - Img.min()) / (Img.max() - Img.min()) * 255
	Mask=nib.load(MaskPath).get_data()
	center=np.int_(PatchIntensity(Img,Mask))
	Markers=np.ones_like(Img)*-1
	Markers[center[0]-20:center[0]+20,center[1]-20:center[1]+20,center[2]-20:center[2]+20]=0
	threshold=Img[center[0],center[1],center[2]]*0.85
	Markers[(Img>threshold)&(Markers==0)]=1
	Markers[center[0]-2:center[0]+3,center[1]-2:center[1]+3,center[2]-2:center[2]+3]=1
	labels = watershed(Img, Markers,compactness=0.1)
	nib.save(nib.Nifti1Image(labels, ImgInfo.affine),'labels.nii.gz')	
	where=np.where(labels==1)
	counts = np.bincount(where[2])
	Zslice=np.argmax(counts)
	centroid=np.int_(np.median(np.array(np.where(labels[:,:,Zslice]==1)),axis=1))
	centroid=np.array((centroid[0],centroid[1],Zslice))
	distance=np.absolute(centroid-center)
	if distance[0]>15:
		centroid[0]=int((centroid[0]+center[0])/2)
	if distance[1]>15:
		centroid[1]=int((centroid[1]+center[1])/2)
	if distance[2]>7:
		centroid[2]=int((centroid[2]+center[2])/2)			
	NewMask=np.ones_like(Mask)
	NewMask[centroid[0],centroid[1],centroid[2]]=0
	NewMask=ndi.distance_transform_edt(NewMask)
	nib.save(nib.Nifti1Image(NewMask, ImgInfo.affine),os.path.join('./RegionGrow/',Patient+'_Mask_'+str(Finding)+'.nii.gz'))

# plt.imshow(Mask[:,:,center[2]])	
# plt.show()