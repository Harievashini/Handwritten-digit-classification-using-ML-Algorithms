#!/usr/bin/python
import numpy as np 
import scipy.special
import sklearn
from sklearn.metrics import *
import matplotlib.pyplot as plot
#Training the dataset
def training(inputs,weights,target):
    #appending +1 for bias and taking transpose of inputs for dot product
    inputs = (np.append([1],inputs)).T
    y=[] #output list
    for i in range(no_of_perceptrons):
        b = weights[i:i+1, 0:]
        #Finding summation of inputs and weights for each perceptron
        xiwi=(b.dot(inputs))
        #Determining the output y
        if(xiwi>0):
            y.append(1)
        else:
            y.append(0)
    #weight updation due to difference in output and target
    for j in range(len(y)):
        if(j==target):
            if(y[target]!=1):
                oerror=1-0  # Determining output error(t-y)
                updated_weights=update_weights(weights,oerror,inputs,target) #updating the weights if output is not 1
            else:
                updated_weights=weights
    
        else:
            if(y[j]!=0):
                oerror=0-1 # Determining output error(t-y)
                updated_weights=update_weights(weights,oerror,inputs,j) #updating the weights if output is not 0
            else:
                updated_weights=weights

    return updated_weights #Weights are updated

def training_accuracy(correctly_classified,weights,totaltrainset):

    for ldata in traindata: #Accessing the training examples
        xi=(ldata.strip()).split(",") # Accessing each pixels of a training data      
         # First column in a training example is actual target       
        target=int(xi[0])
        xi=xi[1:]
         #Preprocessing       
        in1 = ((np.asfarray(xi))/255)
        inputs = (np.append([1],in1)).T # Inputs to the peceptron
        xiwi_array=np.dot(weights,inputs) #summation of weights and inputs
        # Highest value of weight and input summation
        highest_xiwi=np.argmax(xiwi_array)
        if(highest_xiwi==target):
            correctly_classified+=1 #correctly classified training examples
    
    accuracy=(float(correctly_classified)/float(totaltrainset))*100 #accuracy for the training data is computed
    print "accuracy of train data",accuracy 
    #List of train Accuracies
    train_accuracies.append(accuracy)
    
    
def testing_accuracy(correctly_classified,weights,totaltestset):
    
    for ldata in testdata: #Accessing the test data
        xi=(ldata.strip()).split(",")  # Accessing each pixels of a test data      
        # First column in a test example is actual target      
        target=int(xi[0])
        xi=xi[1:]
        #Preprocessing        
        in1 = ((np.asfarray(xi))/255)
        inputs = (np.append([1],in1)).T #Inputs to the perceptron
        xiwi_array=np.dot(weights,inputs) #Summation of weights and inputs
        # Highest value of weight and input summation
        highest_xiwi=np.argmax(xiwi_array)
        if(highest_xiwi==target):
            correctly_classified+=1 #correctly classified test examples
    
    accuracy=(float(correctly_classified)/float(totaltestset))*100 #accuracy for the test data is computed
    print "accuracy of test data",accuracy 
    #List of test Accuracies
    test_accuracies.append(accuracy)
 
def compute_confusion_matrix(weights):
    
    for ldata in testdata: #Accessing the test data
        xi=(ldata.strip()).split(",") # Accessing each pixels of a test data      
        # First column in a test example is actual target      
        target=int(xi[0])
        xi=xi[1:]
        #Preprocessing        
        in1 = ((np.asfarray(xi))/255)
        inputs = (np.append([1],in1)).T #Inputs to the perceptron
        xiwi_array=np.dot(weights,inputs) #Summation of weights and inputs
        # Highest value of weight and input summation
        highest_xiwi=np.argmax(xiwi_array)
        # Adding the predicted values and target values to the list for computing confusion matrix
        prediction_list.append(highest_xiwi)
        target_list.append(target)
    print "\n Confusion Matrix"
    #Using target and prediction list confusion matrix is drawn
    print(confusion_matrix(target_list, prediction_list))        
    

def update_weights(weights,oerror,inputs,pos):

    # updating weights using perceptron learning algorithm
    weights[pos:pos+1,0:]+=lr*np.dot(oerror,inputs)
    return weights
        
#Reading the train dataset
train_file=open("mnist_train.csv" ,"r")
traindata=train_file.readlines()
train_file.close()

#Reading the test dataset
test_file=open("mnist_test.csv" ,"r")
testdata=test_file.readlines()
test_file.close()

#Initializing epochs,learningrates and number of perceptrons
s=220
no_of_perceptrons=10
learningrates=[0.01,0.1,1.0] 
epoch=50
inputs=785


# Computing the data for different learning rates  
for lr in learningrates:
    print "Learning rate: ", lr
    weights=np.random.uniform(-0.05,0.05,[10,785]) #weights
    #Epoch and accuracy lists
    test_accuracies=[]
    epochs=[]
    train_accuracies=[]
    #Prediction and target lists
    prediction_list=[]
    target_list=[]
    # Training the data for 50 epochs
    for e in range(epoch+1):
        print "End of Epoch ",e
        if e==0:
            #Accuracies for test and train data for epoch 0
            correctly_classified=0
            training_accuracy(correctly_classified,weights,len(traindata))
            correctly_classified=0
            testing_accuracy(correctly_classified,weights,len(testdata))
        else:
            
            correctly_classified=0
            for ldata in traindata: #Accessing the test data
                xi=(ldata.strip()).split(",") # Accessing each pixels of a test data 
               # First column in a test example is actual target 
                target=int(xi[0])
                xi=xi[1:]
                #preprocessing
                in1 = ((np.asfarray(xi))/255)
                # Training the data
                updated_weights = training(in1,weights,target)
                # updated_weights
                weights=updated_weights
            #computing accuracy of both test data and train data after each epoch
            training_accuracy(correctly_classified,weights,len(traindata))
            correctly_classified=0
            testing_accuracy(correctly_classified,weights,len(testdata))
        epochs.append(e)
    compute_confusion_matrix(weights)
    
    
    # Graph of train data and test data with accuracy and epoch in y and x axis for various learning rates is plotted
    s+=1
    plot.figure(1,figsize=(10,8))
    plot.subplot(s)
    plot.title("Learning Rate %s"%lr)
    plot.plot(epochs,test_accuracies,label='Test Data')
    plot.plot(epochs,train_accuracies,label='Train Data')
    plot.legend(loc='lower right')
    plot.ylabel("Accuracy")
    plot.yticks([20,40,60,80,100])
    plot.ylim(0,100)
    plot.xlabel("Epoch")
    plot.tight_layout()
    
plot.show()
            

    


