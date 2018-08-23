from keras.models import Sequential
from keras.layers import Conv2D,Dense,MaxPooling2D,Flatten,Conv3D,MaxPooling3D,Dropout,core,Reshape,Input
from keras.callbacks import Callback,TensorBoard
from keras import backend as K
from keras.engine import Layer
from ent import *
import numpy as np


'''
What I am trying to do is
	1. First stage:
		- save data and the gradient when the accuracy is increasing 
	2. Second stage:
		when the accuracy is decreasing:
			- compare the coming data and the previous data to find the 
				most similar one D_s
			- From the first stage, we know how the gradient is changing
				when learning D_s. 
			- So we got current gradient G_c, and the gradient G_s when
				learning D_s 
			- Add the constrain to loss function |G_c - G_s|**2
'''




def Custom_loss(w_loss):

    def loss(y_true,y_pred):
        return K.categorical_crossentropy(y_true, y_pred) + w_loss

    return loss 
        

class cnn_model(object):

	def __init__(self,name):
		self.dim = 64
		self.num_classes = 10
		self.name = name

	def fit(self,X_train,y_train):
		input_shape = X_train[0].shape
		w_loss = 0
		
		self.model = Sequential()
		self.model.add(Conv2D(32, input_shape=input_shape,kernel_size=(3, 3),activation='relu',kernel_initializer='he_normal'))
		self.model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',kernel_initializer='he_normal'))
		self.model.add(MaxPooling2D((2, 2)))
		#self.model.add(Dropout(0.20))
		self.model.add(Conv2D(64, (3, 3), activation='relu',padding='same',kernel_initializer='he_normal'))
		self.model.add(Conv2D(64, (3, 3), activation='relu',padding='same',kernel_initializer='he_normal'))
		self.model.add(MaxPooling2D(pool_size=(2, 2)))
		#self.model.add(Dropout(0.25))
		self.model.add(Conv2D(self.dim, (3, 3), activation='relu',padding='same',kernel_initializer='he_normal'))
		#self.model.add(Dropout(0.25))
		self.model.add(Flatten())
		self.model.add(Dense(self.num_classes,activation='softmax'))
		
		
		tbCallBack = TensorBoard(log_dir='./logs/mnist_drift/final/5000_{}'.format(self.name),  
                 histogram_freq=0,  
                 write_graph=True,  
                 write_grads=True, 
                 write_images=True,
                 embeddings_freq=0, 
                 embeddings_layer_names=None, 
                 embeddings_metadata=None)

		paracallback = para_callback(X_train)
		epochcallback = epoch_callback(X_train)
		drawcallback = draw_callback(self.name)


		if self.name == 'nor':
			print('--------------normal cnn--------------')
			self.model.compile(optimizer='sgd',loss=Custom_loss(w_loss),metrics=['accuracy'])
			self.model.fit(X_train,y_train,epochs=10,batch_size=128,verbose=True,callbacks=[tbCallBack,drawcallback])
		elif self.name =='info_gain':
			print('--------------interference cnn--------------')
			self.model.compile(optimizer='sgd',loss=Custom_loss(w_loss),metrics=['accuracy'])
			self.model.fit(X_train,y_train,epochs=10,batch_size=128,verbose=True,callbacks=[tbCallBack,drawcallback,epochcallback])
		elif self.name =='para':
			print('--------------para cnn--------------')
			self.model.compile(optimizer='sgd',loss=Custom_loss(w_loss),metrics=['accuracy'])
			self.model.fit(X_train,y_train,epochs=10,batch_size=128,verbose=True,callbacks=[tbCallBack,drawcallback,paracallback])

	def evaluate(self,X_test,y_test):
		
		score=self.model.evaluate(X_test,y_test,batch_size=128)
		print("Convolutional neural network test loss:",score[0])
		print('Convolutional neural network test accuracy:',score[1])

		return score[1]



class para_callback(Callback):

	def __init__(self,X):
		self.X = X
		self.theta = []
		self.drift = []
		self.num = X.shape[0]
		self.epoch = 0
		self.pre_acc = 0
		self.pre_grad = None
		self.pre_data=None

		self.acc=[]
		self.acc_drift = []


	def weight_loss(self,cur_grad,pre_grad):
		nor = K.variable(7)
		return sum(list(map(lambda x: 
				(K.sum((x[0] - x[1])))**2,zip(cur_grad,pre_grad))))/nor


	def on_epoch_begin(self,epoch,logs={}):
		self.pre_data=[]
		self.theta = []
		self.epoch = epoch
	

	def on_batch_end(self,batch,logs={}):
		w_loss=0		
		self.batch_size = self.params['batch_size']	
		cur_data = self.X[self.batch_size*batch:self.batch_size*(batch+1)]
		cur_acc = logs.get('acc')

		if batch ==0:
			self.pre_acc = cur_acc
			self.pre_data.append(cur_data)
			self.theta.append(get_gradients(self.model))
			return 

		if 0<batch <=(self.num/self.batch_size * 2 / 3) :
			if cur_acc > self.pre_acc:
				self.pre_data.append(cur_data)
				self.theta.append(get_gradients(self.model))
		elif (self.num/self.batch_size * 2 / 3)<batch <(self.num/self.batch_size -1):	
			if cur_acc < self.pre_acc:
				indi = batch_loss(self.pre_data,cur_data)
				cur_w = get_gradients(self.model)
				w_loss = self.weight_loss(cur_w, self.theta[indi])
				

		self.pre_acc = cur_acc
		if (batch+1)/4 == 0:
			self.acc_drift.append(cur_acc)

		self.acc.append(cur_acc)
		self.model.compile(optimizer='sgd',loss=Custom_loss(w_loss),metrics=['accuracy'])


class epoch_callback(Callback):

	def __init__(self,X):
		self.X = X
		self.drift = []
		self.num = X.shape[0]
		self.epoch = 0
		self.pre_acc = 0
		self.pre_grad = None

		self.acc=[]
		self.acc_drift = []


		self.theta1 = []
		self.theta2 = []
		self.theta3 = []
		self.theta4 = []

		self.pre_data1 = [] 
		self.pre_data2 = []
		self.pre_data3 = []
		self.pre_data4 = [] 



	def weight_loss(self,cur_grad,pre_grad):
		nor = K.variable(7)
		return sum(list(map(lambda x: 
				(K.sum((x[0] - x[1])))**2,zip(cur_grad,pre_grad))))/nor


	def on_epoch_begin(self,epoch,logs={}):
		self.epoch = epoch
		self.count = 0
		if self.epoch % 2 ==0:
			self.theta1 = []
			self.theta2 = []
			self.theta3 = []
			self.theta4 = []

			self.pre_data1 = [] 
			self.pre_data2 = []
			self.pre_data3 = []
			self.pre_data4 = [] 
		
	

	def on_batch_end(self,batch,logs={}):

		w_loss=0		
		self.batch_size = self.params['batch_size']	
		self.batch_num = self.num/self.batch_size	
		'''
		print('---')
		print(round(self.batch_num/4+1))
		print(round(self.batch_num * 2/4 - 1),round(self.batch_num * 2/4+1))
		print(round(self.batch_num * 3/4 - 1),round(self.batch_num * 3/4+1))
		print(round(self.batch_num -3),round(self.batch_num-1))
		'''
		cur_data = self.X[self.batch_size*batch:self.batch_size*(batch+1)]
		cur_acc = logs.get('acc')

		if batch ==0:
			self.pre_acc = cur_acc
			self.pre_data1.append(cur_data)
			self.theta1.append(get_gradients(self.model))
			return 

		if self.epoch % 2 ==0:
			if 0<batch<(self.batch_num -1):
				if cur_acc > self.pre_acc:
					if batch<round(self.batch_num/3+1):
						self.theta1.append(get_gradients(self.model))
						self.pre_data1.append(cur_data)
					elif round(self.batch_num /3+1)<batch<round(self.batch_num * 2/3+1):
						self.theta2.append(get_gradients(self.model))
						self.pre_data2.append(cur_data)
					elif round(self.batch_num * 2/3 + 1)<batch<round(self.batch_num -1):
						self.theta3.append(get_gradients(self.model))
						self.pre_data3.append(cur_data)
					#elif round(self.batch_num *3/4+1)<batch<round(self.batch_num-1):
					#	self.theta4.append(get_gradients(self.model))
					#	self.pre_data4.append(cur_data)
		else:	
			if 0<batch<(self.num/self.batch_size -1):
				if cur_acc < self.pre_acc:
					if batch<round(self.batch_num/3+1):
						indi = batch_loss(self.pre_data1,cur_data)
						cur_w = get_gradients(self.model)
						w_loss = self.weight_loss(cur_w, self.theta1[indi])
					elif round(self.batch_num /3+1)<batch<round(self.batch_num * 2/3+1):
						indi = batch_loss(self.pre_data2,cur_data)
						cur_w = get_gradients(self.model)
						w_loss = self.weight_loss(cur_w, self.theta2[indi])
					elif round(self.batch_num * 2/3 + 1)<batch<round(self.batch_num - 1):
						indi = batch_loss(self.pre_data3,cur_data)
						cur_w = get_gradients(self.model)
						w_loss = self.weight_loss(cur_w, self.theta3[indi])
					#elif round(self.batch_num *3/4+1)<batch<round(self.batch_num-1):
					#	indi = batch_loss(self.pre_data4,cur_data)
					#	cur_w = get_gradients(self.model)
					#	w_loss = self.weight_loss(cur_w, self.theta4[indi])
				

		self.pre_acc = cur_acc
		if (batch+1)/4 == 0:
			self.acc_drift.append(cur_acc)

		self.acc.append(cur_acc)
		self.model.compile(optimizer='sgd',loss=Custom_loss(w_loss),metrics=['accuracy'])



		

class draw_callback(Callback):
	"""docstring for draw"""
	def __init__(self,name):
		super(draw_callback,self).__init__()
		self.acc=[]
		self.acc_drift = []
		self.name = name
		
	def on_batch_end(self,batch,logs={}):
		cur_acc = logs.get('acc')
		self.acc.append(cur_acc)
		
		if (batch+1)%4 == 0:
			self.acc_drift.append(cur_acc)

	def on_train_end(self,batch,logs={}):

		np.set_printoptions(precision=4)


		np.savetxt('./data/{}.txt'.format(self.name),np.array(self.acc))
		np.savetxt('./data/{}_drift.txt'.format(self.name),np.array(self.acc_drift))

		import matplotlib.pyplot as plt
		plt.subplot(211)
		plt.plot(self.acc,label='{} variance is {}'.format(self.name,str(np.cov(self.acc))))
		if self.name=='nor':
			for x,y in enumerate(self.acc):
				if x% 4==0:
					plt.vlines(x,0,1,color='r',linestyles='dashed')

		plt.title('Accuracy per batch')
		plt.subplot(212)
		plt.plot(self.acc_drift,label='{} variance is {}'.format(self.name,str(np.cov(self.acc_drift))))
		plt.title('Accuracy per drift')
		plt.subplots_adjust(wspace=1,hspace=1)
		plt.legend()
		plt.savefig('epoch_1500.png'.format(self.name))
		self.model.save('{}.h5'.format(self.name))
		





















































