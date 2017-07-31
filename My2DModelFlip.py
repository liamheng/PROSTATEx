####nvidia-docker run -it --name=henglikeras -v /projects-priv10/hli17/:/projects-priv10/hli17/ gcr.io/tensorflow/tensorflow:latest-gpu /bin/bash
### docker start henglikeras
### nvidia-docker attach henglikeras
### pip install h5py
### pip install keras
from __future__ import print_function
import os
import numpy as np
import pandas as pd
from scipy import misc
from sklearn.metrics import roc_auc_score
import keras
from keras import optimizers
from keras.models import Model,Sequential,model_from_json
from keras.layers import Flatten,Dense,Input,Dropout
from keras.layers import Conv2D,MaxPooling2D,GlobalAveragePooling2D,GlobalMaxPooling2D,ZeroPadding2D
from keras.utils import plot_model
from keras.models import load_model
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
##############
# for round in range(0,1):
round=0
np.random.seed(round)
##Load data
Table = pd.ExcelFile('ProstateX-2-Findings-Train.xls').parse('ProstateX')
Patients=Table.iloc[:,0]
Findings=Table.iloc[:,1]
Labels=np.array(Table.iloc[:,4])-1
Label0Index=np.random.permutation(np.where(Labels==0)[0])
Label1Index=np.random.permutation(np.where(Labels==1)[0])
Label2Index=np.random.permutation(np.where(Labels==2)[0])
Label3Index=np.random.permutation(np.where(Labels==3)[0])
Label4Index=np.random.permutation(np.where(Labels==4)[0])
Path='./PNG_TABK/'
	
def readdata(LabelIndex,number=0,outall=False):
	file_names=[str(Patient)+'_'+str(Finding) for Patient,Finding in zip(Patients[LabelIndex],Findings[LabelIndex])]	
	TrainData=[]
	if outall:
		for names in file_names:
			for ii in range(31):
				TrainData.append(misc.imread(Path+names+'_'+str(ii)+'.png'))
		OutData=np.array(TrainData)
	else:	
		OutData=np.array([misc.imread(Path+names+'_0.png') for names in file_names])
		for names in file_names:
			for ii in range(1,31):
				TrainData.append(misc.imread(Path+names+'_'+str(ii)+'.png'))
		TrainData=np.array(TrainData)		
		indices=np.random.choice(TrainData.shape[0], number, replace=False)
		OutData=np.concatenate((OutData,TrainData[indices]),axis=0)
	TrainData=[]
	return OutData
	
def generatemore(data,filp=4):
	n,w,h=data.shape[:3]
	Output=np.repeat(data,filp,axis=0)
	Output[:n,:43,:,:]=data[:,43:86,:,:]
	Output[n:n*2,-42:,:,:]=data[:,-84:-42,:,:]
	if filp>2:
		Output[n*2:n*3,:,:43,:]=data[:,:,43:86,:]
		Output[n*3:n*4,:,-42:,:]=data[:,:,-84:-42,:]
	return Output	
	
# TrainData=np.concatenate((readdata(Label0Index[:29],outall=True),readdata(Label1Index[:33],outall=True),generatemore(readdata(Label2Index[:15],outall=True),2),generatemore(readdata(Label3Index[:5],outall=True)),generatemore(readdata(Label4Index[:5],outall=True))))	
# TrainLabel=np.concatenate((np.ones((Label0Index[:29].shape[0]*31,1))*0,np.ones((Label1Index[:33].shape[0]*31,1))*1,np.ones((Label2Index[:15].shape[0]*2*31,1))*2,np.ones((Label3Index[:5].shape[0]*4*31,1))*3,np.ones((Label4Index[:5].shape[0]*4*31,1))*4))
TrainData=np.concatenate((generatemore(readdata(Label0Index[:29],126)),generatemore(readdata(Label1Index[:33],122)),generatemore(readdata(Label2Index[:15],140)),generatemore(readdata(Label3Index[:5],outall=True)),generatemore(readdata(Label4Index[:5],outall=True))))
TrainLabel=np.concatenate((np.ones((620,1))*0,np.ones((620,1))*1,np.ones((620,1))*2,np.ones((620,1))*3,np.ones((620,1))*4))
TrainLabel=keras.utils.to_categorical(TrainLabel)


TestData=np.concatenate((readdata(Label0Index[29:],outall=True),readdata(Label1Index[33:],outall=True),readdata(Label2Index[15:],outall=True),readdata(Label3Index[5:],outall=True),readdata(Label4Index[5:],outall=True)))
TestLabel=np.concatenate((np.ones((Label0Index[29:].shape[0]*31,1))*0,np.ones((Label1Index[33:].shape[0]*31,1))*1,np.ones((Label2Index[15:].shape[0]*31,1))*2,np.ones((Label3Index[5:].shape[0]*31,1))*3,np.ones((Label4Index[5:].shape[0]*31,1))*4))
TestLabel=keras.utils.to_categorical(TestLabel)	

#####Build model
def get_model(summary=False):
	""" Return the Keras model of the network
	"""
	model = Sequential()
	input_shape=(128,128,4) # c, l, h, w
	# 1st layer group
	model.add(Conv2D(64, kernel_size=(3, 3), activation='relu',
							border_mode='same', name='conv1_1',
							input_shape=input_shape))
	# model.add(Conv2D(64, kernel_size=(3, 3), activation='relu',
							# border_mode='same', name='conv1_2'))							
	model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2),
						   border_mode='valid', name='pool1'))
	# 2nd layer group
	model.add(Conv2D(128, kernel_size=(3, 3), activation='relu',
							border_mode='same', name='conv2_1'))
	# model.add(Conv2D(128, kernel_size=(3, 3), activation='relu',
							# border_mode='same', name='conv2_2'))
	# model.add(Conv3D(128, 3, 3, 3, activation='relu',
							# border_mode='same', name='conv2_3'))
	model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2),
						   border_mode='valid', name='pool2'))
	# 3rd layer group
	model.add(Conv2D(256, kernel_size=(3, 3), activation='relu',
							border_mode='same', name='conv3_1'))
	# model.add(Conv3D(256, 3, 3, 3, activation='relu',
							# border_mode='same', name='conv3_2'))
	model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2),
						   border_mode='valid', name='pool3'))
	# 4rd layer group
	model.add(Conv2D(512, kernel_size=(3, 3), activation='relu',
							border_mode='same', name='conv4_1'))
	# model.add(Conv3D(256, 3, 3, 3, activation='relu',
							# border_mode='same', name='conv3_2'))
	model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2),
						   border_mode='valid', name='pool4'))	
	# 5rd layer group
	model.add(Conv2D(1024, kernel_size=(3, 3), activation='relu',
							border_mode='same', name='conv5_1'))
	# model.add(Conv3D(256, 3, 3, 3, activation='relu',
							# border_mode='same', name='conv3_2'))
	model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2),
						   border_mode='valid', name='pool5'))	
	model.add(Flatten())
	# FC layers group
	model.add(Dense(4096, activation='relu', name='fc1'))
	model.add(Dropout(.5))
	model.add(Dense(2048, activation='relu', name='fc2'))
	model.add(Dropout(.5))
	model.add(Dense(5, activation='softmax', name='fc3'))
	return model
	
model = get_model()	
# sgd = optimizers.SGD(lr=5e-7)		
# model.compile(loss='categorical_crossentropy', optimizer=sgd,metrics=['accuracy'])
# model=load_model('./VGG19Round'+str(round)+'_weights.hdf5')
sgd = optimizers.SGD(lr=1e-4)		
model.compile(loss='categorical_crossentropy', optimizer=sgd,metrics=['accuracy'])
earlyStopping=keras.callbacks.EarlyStopping(monitor='val_acc', patience=50, verbose=0, mode='auto')
saveBestModel=keras.callbacks.ModelCheckpoint('./My2DFlip_TABK_Round'+str(round)+'best_weights.hdf5', monitor='val_acc', verbose=1, save_best_only=True, mode='auto')	
history=model.fit(TrainData, TrainLabel,epochs=5000,batch_size=45,verbose=1,callbacks=[saveBestModel,earlyStopping],validation_data=(TestData, TestLabel))
model.save('./My2DFlip_TABK_Round'+str(round)+'_weights.hdf5')	
loss_history = np.array(history.history["loss"])
val_loss_history = np.array(history.history["val_loss"])
acc_history = np.array(history.history["acc"])
val_acc_history = np.array(history.history["val_acc"])
np.savetxt('./My2DFlip_TABK_Round'+str(round)+'_history.txt', [loss_history,val_loss_history,acc_history,val_acc_history], delimiter=",")
##Test predict label
model = load_model('./My2DFlip_TABK_Round'+str(round)+'best_weights.hdf5')
Pred = model.predict(TestData)	
PredLabel=np.argmax(Pred,axis=1)
RealLabel=np.argmax(TestLabel,axis=1)
Difference=PredLabel-RealLabel
Accuracy=Difference[Difference==0].shape[0]/float(Difference.shape[0])
print('Test Accuracy:',Accuracy)
