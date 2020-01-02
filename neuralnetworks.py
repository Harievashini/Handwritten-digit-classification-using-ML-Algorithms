#!/usr/bin/python
import numpy as np 
import scipy.special
import sklearn
from sklearn.metrics import *
import matplotlib.pyplot as plot
import random
#Number of training examples are 60000
def exp(trainsize=60000):
	s=220
	print "Train data size:",trainsize
	for hu in hiddenunits:
		for m in momentum:
			print "hiddenunits=",hu
			print "momentum=",m
			w_i_h=np.random.uniform(-0.05,0.05,[hu,inputs]) #Weights between input and hidden nodes
			w_h_o=np.random.uniform(-0.05,0.05,[outputs,hu+1]) #weights between hidden and output nodes
			test_accuracies=[] #All the test data accuracies for plotting
			epochs=[] #Epoch list for plotting
			train_accuracies=[] #All the train data accuracies for plotting
    		#Prediction and target lists
			prediction_list=[] #The digits,model predicted
			target_list=[] #Actual target digits
			p_w_i_h= np.zeros(w_i_h.shape)
			p_w_h_o = np.zeros(w_h_o.shape)
			for e in range(epoch+1):
				print "Epoch=",e
				if e==0:
            		#Accuracies for test and train data for epoch 0
					correctly_classified=0
					train_accuracies=training_accuracy(correctly_classified,w_i_h,w_h_o,train_accuracies,trainsize) #Computing training accuracy
					correctly_classified=0
					test_accuracies=testing_accuracy(correctly_classified,w_i_h,w_h_o,test_accuracies) #Computing testing accuracy
				else:
					for count,ldata in enumerate(traindata):
						if count <= trainsize:
							#Accessing the test data
							xi=(ldata.strip()).split(",") # Accessing each pixels of a test data 
							# First column in a test example is actual target 
							target=int(xi[0])
							xi=xi[1:]
							#preprocessing
							inp = ((np.asfarray(xi))/255)
							targets=[]
							#Target 
							for i in range(outputs):
								if(i==target):
									targets.append(0.9)
								else:
									targets.append(0.1)
							# Training the data
							p_w_h_o,p_w_i_h,w_h_o,w_i_h=training(inp,w_i_h,hu,w_h_o,targets,m,p_w_h_o,p_w_i_h)
					correctly_classified=0
					train_accuracies=training_accuracy(correctly_classified,w_i_h,w_h_o,train_accuracies,trainsize) #Computing training accuracy
					correctly_classified=0
					test_accuracies=testing_accuracy(correctly_classified,w_i_h,w_h_o,test_accuracies) #Computing testing accuracy
				epochs.append(e)
			print "train",train_accuracies
			print "test",test_accuracies
			print "epochs",epochs
			#Confusion matrix for test data
			compute_confusion_matrix(w_i_h,w_h_o,prediction_list,target_list)
			#Graph of train data,test data - epoch vs accuracy
			s+=1
			plot.figure(1,figsize=(10,8))
			plot.subplot(s)
			plot.title("Hidden Units: %s, Momentum: %s"%(hu,m))
			plot.plot(epochs,test_accuracies,label='Test Data')
			plot.plot(epochs,train_accuracies,label='Train Data')
			plot.legend(loc='lower right')
			plot.ylabel("Accuracy")
			plot.yticks(range(0,110,10))
			plot.ylim(0,110)
			plot.xlabel("Epoch")
			plot.tight_layout()
	plot.show()


def training_accuracy(correctly_classified,w_i_h,w_h_o,train_accuracies,trainsize):
	for count,ldata in enumerate(traindata):
		if count <= trainsize:#Accessing the test data
			xi=(ldata.strip()).split(",") # Accessing each pixels of a train data 
			# First column in a test example is actual target 
			target=int(xi[0])
			xi=xi[1:]
			#preprocessing
			inp = ((np.asfarray(xi))/255)
			#appending +1 for bias and taking transpose of inputs for dot product
			inp = (np.append([1],inp)).T
			#Finding hj hidden node value - summation of input times the weight
			h_input=np.dot(w_i_h,inp) 
			h_act=1/(1+np.exp(-h_input)) #Activation function
			h_act = (np.append([1],h_act)).T #appending the bias
			#Finding Ok Output node values - summation of hidden value times the weight
			o_input=np.dot(w_h_o,h_act)
			o_act=1/(1+np.exp(-o_input)) #Activation function
			highest_o=np.argmax(o_act) #Finding the predicted value
			if(highest_o==target): #Comparing with the target
				correctly_classified+=1 #correctly classified train examples
	accuracy=(float(correctly_classified)/float(trainsize))*100 #accuracy for the train data is computed
	print "accuracy of train data",accuracy 
	#List of train Accuracies
	train_accuracies.append(accuracy)
	return train_accuracies

def testing_accuracy(correctly_classified,w_i_h,w_h_o,test_accuracies):
	for ldata in testdata: #Accessing the test data
		xi=(ldata.strip()).split(",") # Accessing each pixels of a test data 
		# First column in a test example is actual target 
		target=int(xi[0])
		xi=xi[1:]
		#preprocessing
		inp = ((np.asfarray(xi))/255)
		#appending +1 for bias and taking transpose of inputs for dot product
		inp = (np.append([1],inp)).T
		#Finding hj hidden node value - summation of input times the weight
		h_input=np.dot(w_i_h,inp) 
		h_act=1/(1+np.exp(-h_input)) #Activation function
		h_act = (np.append([1],h_act)).T #appending the bias
		#Finding Ok Output node values - summation of hidden value times the weight
		o_input=np.dot(w_h_o,h_act)
		o_act=1/(1+np.exp(-o_input)) #Activation function
		highest_o=np.argmax(o_act)#Finding the predicted value
		if(highest_o==target): #Comparing with the target
			correctly_classified+=1 #correctly classified test examples
	accuracy=(float(correctly_classified)/float(len(testdata)))*100 #accuracy for the test data is computed
	print "accuracy of test data",accuracy 
	#List of test Accuracies
	test_accuracies.append(accuracy)
	return test_accuracies

def compute_confusion_matrix(w_i_h,w_h_o,prediction_list,target_list):
	for ldata in testdata: #Accessing the test data
		xi=(ldata.strip()).split(",") # Accessing each pixels of a test data 
		# First column in a test example is actual target 
		target=int(xi[0])
		xi=xi[1:]
		#preprocessing
		inp = ((np.asfarray(xi))/255)
		#appending +1 for bias and taking transpose of inputs for dot product
		inp = (np.append([1],inp)).T
		#Finding hj hidden node value - summation of input times the weight
		h_input=np.dot(w_i_h,inp) 
		h_act=1/(1+np.exp(-h_input))#Activation function
		h_act = (np.append([1],h_act)).T #appending the bias
		#Finding Ok Output node values - summation of hidden value times the weight
		o_input=np.dot(w_h_o,h_act)
		o_act=1/(1+np.exp(-o_input))#Activation function
		highest_o=np.argmax(o_act) #Finding the predicted value
		prediction_list.append(highest_o) #appending the predicted digits to the predicted list
		target_list.append(target) #appending the target digits to the target list
	print "\n Confusion Matrix"
    #Using target and prediction list confusion matrix is drawn
	print(confusion_matrix(target_list, prediction_list))  

def training(inp,w_i_h,hiddenunits,w_h_o,targets,momentum,p_w_h_o,p_w_i_h):
	#appending +1 for bias and taking transpose of inputs for dot product
	inp = (np.append([1],inp)).T
	#Finding hj hidden node value - summation of input times the weight
	h_input=np.dot(w_i_h,inp) 
	h_act=1/(1+np.exp(-h_input)) #Activation function
	h_act = (np.append([1],h_act)).T #appending the bias
	#Finding Ok Output node values - summation of hidden value times the weight
	o_input=np.dot(w_h_o,h_act)
	o_act=1/(1+np.exp(-o_input))#Activation function
	target_array=np.asfarray(targets) #Converting the targets list to an array
	e_output=o_act*(1-o_act)*(target_array-o_act) #Computing the error in the output layer
	e_hidden=h_act*(1-h_act)*np.dot(e_output,w_h_o) #computing the error in hidden layer
	#converting into a 2 dimensional matrix for dot product
	h_act=np.array(h_act,ndmin=2) 
	e_output=np.array(e_output,ndmin=2)
	d_w_h_o=(learningrate*np.dot(e_output.T,h_act))+(momentum*p_w_h_o) #Change in the weight ,delta weight between the hidden and output layer
	#converting into a 2 dimensional matrix for dot product
	e_hidden=np.array(e_hidden,ndmin=2)
	inp=np.array(inp,ndmin=2)
	d_w_i_h=(learningrate*np.dot(e_hidden[:,1:].T,inp))+(momentum*p_w_i_h) #Change in the weight ,delta weight between the input and hidden layer
	w_h_o+=d_w_h_o #New weight between hidden and output layer
	w_i_h+=d_w_i_h #New weight between input and hidden layer
	#delta weight of previous iteration
	p_w_h_o=d_w_h_o 
	p_w_i_h=d_w_i_h
	return p_w_h_o,p_w_i_h,w_h_o,w_i_h



#Reading the train dataset
train_file=open("mnist_train.csv" ,"r")
traindata=train_file.readlines()
train_file.close()

#Reading the test dataset
test_file=open("mnist_test.csv" ,"r")
testdata=test_file.readlines()
test_file.close()

#Initializing inputs,outputs,learning rate and epochs
inputs=785
outputs=10
epoch=50
learningrate=0.1

#Experiment 1 - Number of hidden units are changed to 10,20,100 and Momentum is kept constant 0.9
print "Experiment 1"
print "Vary number of hidden units"
hiddenunits=[10,20,100]
momentum=[0.9]
exp()
#Experiment 2 - Momentum value is changed to 0.0, 0.5, 1.0, 0.9 and Hidden unit is kept constant 100
print "Experiment 2"
print "Vary the momentum value"
hiddenunits=[100]             
momentum=[0.0,0.5,1.0,0.9] 
exp()
#Experiment 3 - Momentum value is kept 0.9 , Hidden unit is kept 100 and Number of Training examples are changed
print "Experiment 3"
print "Vary the number of training examples"
hiddenunits=[100]             
momentum=[0.9] 
exp(15000)
exp(30000)
