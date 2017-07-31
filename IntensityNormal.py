'''
This script performs intensity normalization.
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

DataPath='./NormTABK'
Files=os.listdir(DataPath)
Files=[ii for ii in Files if 'Mask' not in ii]
for File in Files:
	ImgPath=os.path.join(DataPath,File)
	ImgInfo=nib.load(ImgPath)
	Img=ImgInfo.get_data()
	Img[Img<0]=0
	Img= (Img - Img.min()) / (Img.max() - Img.min()) * 255
	nib.save(nib.Nifti1Image(np.int_(Img), ImgInfo.affine),os.path.join('./RegionGrowIN',File))	


