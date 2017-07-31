'''
This script performs modals registration inter subjects. Software FLS is necassary.
Author: Heng Li
Email: lihengbit@foxmail.com
Date: 06/20/2017
You are free to modify and redistribute the script.
'''
import os
import sys
import scipy.ndimage.interpolation
import numpy as np
import scipy.io as sio
import nibabel as nib
import pandas as pd
import scipy.ndimage

Table = pd.ExcelFile('ProstateX-Findings-Train.xls').parse('ProstateX')
Patients=Table.iloc[:,0]
Findings=Table.iloc[:,1]
# Data loading and preprocessing
DataPath='./NormData/'


		
def registADC(sourcePath, sourcePathadd, templatePath, DataPath):
	sourceInfo = nib.load(os.path.join(DataPath,sourcePath))
	source=sourceInfo.get_data()
	hist,bin_edges=np.histogram(source,bins=128)
	Index=np.where(hist==hist[1:].max())[0][0]
	refImg=np.absolute(source-bin_edges[Index-8])#+source*0.5
	refImg[:,:15,:]=0
	refImg[:,-10:,:]=0
	# refImg[:4,:,:]=0
	# refImg[-20:,:,:]=0
	refImg[:,:,:6]=0
	# refImg[:,:,:3]=0
	#refImg[refImg<1007]=0
	# source[source<1651]=0
	# refImg=source#[:,:-15,:]
	nib.save(nib.Nifti1Image(refImg, sourceInfo.affine),os.path.join(DataPath, 'Regist'+sourcePath))
	templateInfo = nib.load(os.path.join(DataPath,templatePath))
	template=templateInfo.get_data()#[:,:-15,:]
	#template[template<104]=0
	template[:,:,:7]=0
	template[:15,:,:]=0
	template[-15:,:,:]=0
	nib.save(nib.Nifti1Image(template, templateInfo.affine),os.path.join(DataPath, 'Regist'+templatePath))
	os.system('setenv Home '+DataPath)
	os.system('flirt -in ${Home}/Regist'+sourcePath+' -ref ${Home}/Regist'+templatePath
		+' -out ADC2Tra_FSL.nii.gz -omat ADC2Tra.mat -bins 256 '
		+'-cost mutualinfo -searchrx -5 5 -searchry -5 5 -searchrz -5 5 '
		+'-dof 6  -interp sinc -sincwidth 7 -sincwindow hanning')
	os.system('flirt -in ${Home}/'+sourcePath+' -ref ${Home}/'+templatePath
		+' -out ./Out/'+sourcePath+'  -applyxfm -init ADC2Tra.mat '
		+'-interp sinc -sincwidth 7 -sincwindow hanning')
	os.system('flirt -in ${Home}/'+sourcePathadd+' -ref ${Home}/'+templatePath
	+' -out ./Out/'+sourcePathadd+'  -applyxfm -init ADC2Tra.mat '
	+'-interp sinc -sincwidth 7 -sincwindow hanning')	

def registKtrans(sourcePath, templatePath, DataPath):
	os.system('export Home='+DataPath)
	#os.system('fslmaths ${Home}/'+sourcePath+' -mul -5 Ktrans1.nii.gz')
	sourceInfo = nib.load(os.path.join(DataPath,sourcePath))
	source=sourceInfo.get_data()
	source=np.log(source+1e-4)*(-1)
	# source[:,:,:-9]=source.min()
	# source[:,:5,:]=source.min()
	# source[:,-10:,:]=source.min()
	# source[:7,:,:]=source.min()
	# source[-7:,:,:]=source.min()
	nib.save(nib.Nifti1Image(source, sourceInfo.affine),'Ktrans1.nii.gz')		
	templateInfo = nib.load(os.path.join(DataPath,templatePath))
	template=templateInfo.get_data()#[:,:-15,:]
	# template[:,:,:-10]=template.min()
	nib.save(nib.Nifti1Image(template, templateInfo.affine),'Ktrans2.nii.gz')	
	#os.system('fslmaths ${Home}/'+templatePath+' -mul -1 Ktrans1.nii.gz')
	#os.system('flirt -in Ktrans1.nii.gz -ref ${Home}/'+templatePath
	os.system('flirt -in  Ktrans1.nii.gz -ref Ktrans2.nii.gz'
		+' -out Kt2Tra_FSL.nii.gz -omat Kt2Tra.mat -bins 256 '
		+'-cost mutualinfo -searchrx -3 3 -searchry -3 3 -searchrz -2 2 '
		+'-dof 6  -interp sinc -sincwidth 7 -sincwindow hanning')
	os.system('flirt -in ${Home}/'+sourcePath+' -ref ${Home}/'+templatePath
		+' -out ./Out/'+sourcePath+'  -applyxfm -init Kt2Tra.mat '
		+'-interp sinc -sincwidth 7 -sincwindow hanning')
#####Normal Intensity
# for Patient in Patients:
def run(Patient):
	TraPath=Patient+'_Tra.nii.gz'
	KtransPath=Patient+'_Ktrans.nii.gz'
	ADCPath=Patient+'_ADC.nii.gz'
	BVALPath=Patient+'_BVAL.nii.gz'
	# if not os.path.isfile('./Out/'+KtransPath):
		# registKtrans(KtransPath,TraPath,DataPath)
	# if not os.path.isfile('./Out/'+ADCPath):	
	registADC(ADCPath,BVALPath,TraPath,DataPath)	

#run(Patient='ProstateX-0257')
if __name__ == '__main__':
	Patient0 = sys.argv[1]
	run(Patient0)

