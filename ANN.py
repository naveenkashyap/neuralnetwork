import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import numpy as np
import sys
import math
from TensorGenerator import TensorGenerator




#First command line argument is the path to the circle CSV, second command line argument is the path to the torus CSV.



#Configured for No Overlap (TorusRadius=60). Change this if needed here:
TorusRadius=60


#############################################################################################################################################
#STEP ONE: READ IN DATA


#Gives me two numpy arrays. One of them is NumDataPoints by 2 array (each row has an x and y coordinate) and the other is NumDataPoints by 1 array that just contains the labels.
ToGenerateData=TensorGenerator()

#train_x is the training coords, train_y is the training labels, test_x is the testing coords, test_y is the testing labels. 
train_x,train_y,test_x,test_y,NumOfDataPoints=ToGenerateData.OutputData(sys.argv[1],sys.argv[2])


#############################################################################################################################################






#############################################################################################################################################
#STEP TWO: SET UP THE CONSTANTS OF THE NN


#One hidden layer only
num_Nodes=10

#Number of output nodes
n_classes=2

#Batch Size
batch_size=100

#x is input (the x-coordinate and y-coordinate). y is labels. These are placeholders I will feed things through later on (Will feed the data to be analyzed through x and the labels through y)
x=tf.placeholder('float')
y=tf.placeholder('float')


#############################################################################################################################################






#############################################################################################################################################
#STEP THREE: SETTING UP THE ACTUAL LAYER(S) OF THE ANN.


#Setting up the computation graph of the ANN. The ANN has two inputs (the x-coordinate and the y-coordinate of the data point), one layer with the user defined number of nodes, and one output layer. "data" is what I feed to the neural network. 
def neural_network_model(data):

	#The weights are randomized for the first run through. The weights will be calibrated to minimize the cost function. 
	hidden_1_layer={'weights':tf.Variable(tf.random_normal([2,num_Nodes])),'biases':tf.Variable(tf.random_normal([num_Nodes]))}
	
	#These are the weights that are leading into the two output nodes.
	output_layer={'weights':tf.Variable(tf.random_normal([num_Nodes,n_classes])),'biases':tf.Variable(tf.random_normal([n_classes]))}
	
	#This preforms the actual manipulation of the data. The data is multiplied by the weights, and then the biases are added. This gives the nodes something to work with (for their activation function)
	l1=tf.add(tf.matmul(data,hidden_1_layer['weights']),hidden_1_layer['biases'])
	
	#Takes the output of the (input*weights) + biases of layer one and puts it through a sigmoid function (to get the input to the output nodes). This is the output of the first layer that is to be fed into the output layer. The explicit function is: 1/(1+exp(-x))
	l1=tf.nn.sigmoid(l1)

	#Gives the two final values of the ANN. The first output value corresponds to the input point being part of the circle set and the second output value corresponds to the input point being part of the torus set. Since I am using the softmax activation function, the outputs of the ANN correspond to the probability the input point is in a given group. The first output is the predicted probability the input point is a part of the circle group, and the second output point is the predicted probability the input point is a part of the torus group.
	output=tf.add(tf.matmul(l1,output_layer['weights']),output_layer['biases'])

	output=tf.nn.softmax(logits=output)

	
	return output


#############################################################################################################################################






#############################################################################################################################################
#STEP THREE: TRAINING THE NEURAL NETWORK


#x is the data that is to be fed through. 	
def train_neural_network(x):


	#SETTING UP OPERATIONS ON THE NEURAL NETWORK
	

	#Getting the output of the ANN
	prediction=neural_network_model(x)
	
	#Setting up the cost function (which is later minimized to increase accuracy).
	cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y))
	
	#Optimizing function that acts to minimize the cost function (and therefore minimize the difference between the output values and the expected values)
	optimizer = tf.train.AdamOptimizer().minimize(cost)

	#One epoch is a complete cycle through the data as well as the backpropogation (adjusting the weights to increase accuracy)
	num_epochs=50

	#Setting up the accuracy array to hold the ANN's accuracy per epoch.
        AccuracyArray=np.zeros(num_epochs)
		
	
	#Now actually running the ANN.
	with tf.Session() as sess:

		#Initializes the variables (the weights and biases of the first hidden layer and the output layer) that were set up earlier. 
		sess.run(tf.global_variables_initializer())


		#THIS IS THE TRAINING

		#Setting up the correct and accuracy operations to be used later on.
                correct=tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
                accuracy=tf.reduce_mean(tf.cast(correct,'float'))

		for epoch in range(num_epochs):
			epoch_loss = 0
			i=0
			batch_num=0

	
			#Takes batches of the training data.
			while i < len(train_x):
				start=i
				end=i+batch_size
				batch_x=np.array(train_x[start:end])
				batch_y=np.array(train_y[start:end])



				#This is where the ANN is actually ran. The weights and biases are modified with this line of code (since the optimizer operation is ran).
				_,c,ToPrint=sess.run([optimizer,cost, prediction], feed_dict={x: batch_x, y: batch_y})
				epoch_loss+=c
				correct=tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
                		accuracy=tf.reduce_mean(tf.cast(correct,'float'))
				i+=batch_size
			
			#Getting the accuracy per epoch
			AccuracyArray[epoch]=accuracy.eval({x:test_x, y:test_y})*100

			epochNum=epoch+1
		
		#Getting the max accuracy achieved as well as the epoch it achieved that accuracy.
		MaxAccuracyAchieved=np.amax(AccuracyArray)
		EpochMaxAccuracy=np.argmax(AccuracyArray)+1
			
			

#############################################################################################################################################






#############################################################################################################################################
#STEP FOUR: TESTING THE ACCURACY OF THE NEURAL NETWORK 


		#NOTE: Since the optimizer operation is not being ran, the weights are not being changed. 
					
		
		#Setting up the correct and accuracy operations to allow for the accuracy calculation (see ppt. for more detail).
		correct=tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
		accuracy=tf.reduce_mean(tf.cast(correct,'float'))
		
		#Printing a message indicating the accuracy the ANN achieved as well as how many epochs it required.
		print "%f,%d" %(MaxAccuracyAchieved,EpochMaxAccuracy)

		
#############################################################################################################################################






#############################################################################################################################################
#STEP FIVE: RUNNING THE CODE


train_neural_network(x)

 
#############################################################################################################################################
