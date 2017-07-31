####nvidia-docker run -it --name=henglikeras3 -v /projects-priv10/hli17/:/projects-priv10/hli17/ gcr.io/tensorflow/tensorflow:latest-gpu /bin/bash
### docker start henglikeras2
### nvidia-docker attach henglikeras2 
### pip install h5py
### pip install keras
from __future__ import print_function
import os
import numpy as np
import c3d_model
import pandas as pd
from sklearn.metrics import roc_auc_score
import keras
from keras import optimizers
from keras.models import Model,Sequential,model_from_json
from keras.layers import Flatten,Dense,Input,Dropout
from keras.layers import Conv3D,MaxPooling3D,GlobalAveragePooling3D,GlobalMaxPooling3D,ZeroPadding3D
from keras.utils import plot_model
from keras.models import load_model
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
######
for round in range(0,20):
	np.random.seed(round)
	Table = pd.ExcelFile('ProstateX-2-Findings-Train.xls').parse('ProstateX')
	Patients=Table.iloc[:,0]
	Findings=Table.iloc[:,1]
	Labels=np.array(Table.iloc[:,4])-1
	Label0Index=np.random.permutation(np.where(Labels==0)[0])
	Label1Index=np.random.permutation(np.where(Labels==1)[0])
	Label2Index=np.random.permutation(np.where(Labels==2)[0])
	Label3Index=np.random.permutation(np.where(Labels==3)[0])
	Label4Index=np.random.permutation(np.where(Labels==4)[0])

	def readdata(LabelIndex,number=0,outall=False):
		TrainLabelName=['./BAK/'+Patient+'_Finding_'+str(Finding) for Patient, Finding in zip(Patients[LabelIndex],Findings[LabelIndex])]
		TrainData=[]
		for name in TrainLabelName:
			TrainData.append(np.load(name+'.npz')['Data'])
		TrainData=np.array(TrainData)
		if outall:
			TrainData=TrainData.reshape(TrainData.shape[0]*TrainData.shape[1],112,112,16,3)
			OutData=TrainData		
		else:
			OutData=TrainData[:,0]
			AugmentData=TrainData[:,1:]	
			AugmentData=AugmentData.reshape(AugmentData.shape[0]*AugmentData.shape[1],112,112,16,3)
			indices=np.random.choice(AugmentData.shape[0], number, replace=False)
			OutData=np.concatenate((OutData,AugmentData[indices]),axis=0)
		return OutData

	TrainData=np.concatenate((readdata(Label0Index[:29],91),readdata(Label1Index[:33],87),readdata(Label2Index[:15],105),readdata(Label3Index[:5],115),readdata(Label4Index[:5],115)))
	TrainData=np.swapaxes(TrainData,1,3)
	TrainLabel=np.concatenate((np.ones((120,1))*0,np.ones((120,1))*1,np.ones((120,1))*2,np.ones((120,1))*3,np.ones((120,1))*4))
	TrainLabel=keras.utils.to_categorical(TrainLabel)


	TestData=np.concatenate((readdata(Label0Index[29:],outall=True),readdata(Label1Index[33:],outall=True),readdata(Label2Index[15:],outall=True),readdata(Label3Index[5:],outall=True),readdata(Label4Index[5:],outall=True)))
	TestData=np.swapaxes(TestData,1,3)
	TestLabel=np.concatenate((np.ones((Label0Index[29:].shape[0]*24,1))*0,np.ones((Label1Index[33:].shape[0]*24,1))*1,np.ones((Label2Index[15:].shape[0]*24,1))*2,np.ones((Label3Index[5:].shape[0]*24,1))*3,np.ones((Label4Index[5:].shape[0]*24,1))*4))
	TestLabel=keras.utils.to_categorical(TestLabel)


	## Load pre-trained C3D model
	# if os.path.isfile('./BAKRound'+str(round)+'best_weights.hdf5'):
		# int_model=load_model('./BAKRound'+str(round)+'best_weights.hdf5')
		# # for layer in int_model.layers[:-6]:
			# # layer.trainable = False		
	# else:
	model_weight_filename = 'sports1M_weights_tf.h5'
	model_json_filename = 'sports1M_weights_tf.json'
	base_model = model_from_json(open(model_json_filename, 'r').read())
	base_model.load_weights(model_weight_filename)
	layer_dict = dict([(layer.name, layer) for layer in base_model.layers])#weights of each layer
	## Build new model
	int_model = c3d_model.get_int_model(model=base_model, layer='pool5')
	## Lock weights
	# for layer in int_model.layers:
		# layer.trainable = False
	##	
	int_model.add(Flatten())
	int_model.add(Dense(4096, activation='relu', name='fc6',
							weights=layer_dict['fc6'].get_weights()))
	int_model.add(Dropout(.05))
	int_model.add(Dense(4096, activation='relu', name='fc7',
							weights=layer_dict['fc7'].get_weights()))
	int_model.add(Dropout(.05))
	int_model.add(Dense(5, activation='softmax', name='fc8'))
	sgd = optimizers.SGD(lr=1e-5)
	# int_model.compile(loss='categorical_crossentropy', optimizer=sgd)
	int_model.compile(loss='binary_crossentropy', optimizer=sgd,metrics=['accuracy'])
	earlyStopping=keras.callbacks.EarlyStopping(monitor='val_acc', patience=50, verbose=0, mode='auto')
	saveBestModel=keras.callbacks.ModelCheckpoint('./BAKRound'+str(round)+'best_weights.hdf5', monitor='val_acc', verbose=1, save_best_only=True, mode='auto')	
	history=int_model.fit(TrainData, TrainLabel,epochs=5000,batch_size=30,verbose=1,callbacks=[earlyStopping,saveBestModel],validation_data=(TestData, TestLabel))
	int_model.save('./BAKRound'+str(round)+'_weights.hdf5')	
	loss_history = np.array(history.history["loss"])
	val_loss_history = np.array(history.history["val_loss"])		
	acc_history = np.array(history.history["acc"])
	val_acc_history = np.array(history.history["val_acc"])
	np.savetxt('./BAKRound'+str(round)+'_history.txt', [loss_history,val_loss_history,acc_history,val_acc_history], delimiter=",")

		  