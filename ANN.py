import tensorflow as tf
import numpy as np
import sys
import math
import generator



#Command Line Arguments:
#1) Dimensionality
#2) Num Points
#3) OS_mu
#4) OS_sigma
#5) IS_mu
#6) IS_sigma
#7) num_Nodes
#8) num_epochs


#############################################################################################################################################
#STEP ONE: READ IN DATA

g=generator.Generator()
Dimensionality = int(sys.argv[1])
Num_Points = int(sys.argv[2])


OS_mu = int(sys.argv[3])
OS_sigma = int(sys.argv[4])


IS_mu=int(sys.argv[5])
IS_sigma=int(sys.argv[6])

#Generating the outer sphere points
OuterSphereArray=g.generate(OS_mu,OS_sigma,Dimensionality,Num_Points,1)


#Generating the inner sphere points
InnerSphereArray=g.generate(IS_mu,IS_sigma,Dimensionality,Num_Points,0)


#Shuffling the data and splitting it up into the different arrays required.

AllData=np.append(OuterSphereArray,InnerSphereArray,axis=0)
np.random.shuffle(AllData)



TrainingRows=int(math.floor((Num_Points*2)*0.8))

TrainingDataPoints=np.zeros((TrainingRows,Dimensionality))
TrainingLabels=np.zeros((TrainingRows,2))


TestingDataPoints=np.zeros(((Num_Points*2-TrainingRows),Dimensionality))
TestingLabels=np.zeros((len(TestingDataPoints),2))


j=0
for i in range(len(AllData)):
	if i < TrainingRows:
		Row=AllData[i]
		TrainingDataPoints[i]=Row[0:Dimensionality]
		TrainingLabels[i]=Row[-2:]
	else:
		Row=AllData[i]
		TestingDataPoints[j]=Row[0:Dimensionality]
		TestingLabels[j]=Row[-2:]
		j=j+1




#############################################################################################################################################






#############################################################################################################################################
#STEP TWO: SET UP THE CONSTANTS OF THE NN


#One hidden layer only
num_Nodes=int(sys.argv[7])

#Number of output nodes
n_classes=2

#Batch Size
batch_size=100

#Placeholders for the input and labels
Input_Placeholder=tf.placeholder('float')
Labels_Placeholder=tf.placeholder('float')


#############################################################################################################################################






#############################################################################################################################################
#STEP THREE: SETTING UP THE ACTUAL LAYER(S) OF THE ANN.


#Setting up the computation graph of the ANN. The ANN has two inputs (the x-coordinate and the y-coordinate of the data point), one layer with the user defined number of nodes, and one output layer. 
def neural_network_model(data):

	#The weights are randomized for the first run through. The weights will be calibrated to minimize the cost function. 
	hidden_1_layer={'weights':tf.Variable(tf.random_normal([Dimensionality,num_Nodes])),'biases':tf.Variable(tf.random_normal([num_Nodes]))}
	
	#These are the weights that are leading into the two output nodes.
	output_layer={'weights':tf.Variable(tf.random_normal([num_Nodes,n_classes])),'biases':tf.Variable(tf.random_normal([n_classes]))}
	
	#This preforms the actual manipulation of the data. The data is multiplied by the weights, and then the biases are added. This gives the nodes something to work with (for their activation function)
	l1=tf.add(tf.matmul(data,hidden_1_layer['weights']),hidden_1_layer['biases'])
	
	#Takes the output of the (input*weights) + biases of layer one and puts it through a sigmoid function (to get the input to the output nodes). This is the output of the first layer that is to be fed into the output layer. The explicit function is: 1/(1+exp(-x))
	l1=tf.nn.sigmoid(l1)

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
	cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=Labels_Placeholder))
	
	#Optimizing function that acts to minimize the cost function (and therefore minimize the difference between the output values and the expected values)
	optimizer = tf.train.AdamOptimizer().minimize(cost)

	#One epoch is a complete cycle through the data as well as the backpropogation (adjusting the weights to increase accuracy)
	num_epochs=int(sys.argv[8])

	#Setting up the accuracy array to hold the ANN's accuracy per epoch.
        AccuracyArray=np.zeros(num_epochs)
		

	with tf.device('/device:GPU:0'):	
		#Now actually running the ANN.
		with tf.Session() as sess:

			#Initializes the variables (the weights and biases of the first hidden layer and the output layer) that were set up earlier. 
			sess.run(tf.global_variables_initializer())


			#THIS IS THE TRAINING

			#Setting up the correct and accuracy operations to be used later on.
			correct=tf.equal(tf.argmax(prediction,1), tf.argmax(Labels_Placeholder,1))
			accuracy=tf.reduce_mean(tf.cast(correct,'float'))

			for epoch in range(num_epochs):
				i=0

	
				#Takes batches of the training data.
				while i < len(TrainingDataPoints):
					start=i
					end=i+batch_size
					batch_x=np.array(TrainingDataPoints[start:end])
					batch_y=np.array(TrainingLabels[start:end])



					#This is where the ANN is actually ran. The weights and biases are modified with this line of code (since the optimizer operation is ran).
					_,ANNOutput=sess.run([optimizer,prediction], feed_dict={Input_Placeholder: batch_x, Labels_Placeholder: batch_y})
				
					correct=tf.equal(tf.argmax(prediction,1), tf.argmax(Labels_Placeholder,1))
					accuracy=tf.reduce_mean(tf.cast(correct,'float'))
					i+=batch_size
			
				#To-do: Figure out how we want to work with the results. AccuracyArray contains our results per epoch.
				#Getting the accuracy per epoch. 
				AccuracyArray[epoch]=accuracy.eval({Input_Placeholder:TestingDataPoints, Labels_Placeholder:TestingLabels})*100
		
			#Getting the max accuracy achieved as well as the epoch it achieved that accuracy.
			#MaxAccuracyAchieved=np.amax(AccuracyArray)
			#EpochMaxAccuracy=np.argmax(AccuracyArray)+1
			
			

#############################################################################################################################################






#############################################################################################################################################
#STEP FOUR: TESTING THE ACCURACY OF THE NEURAL NETWORK (Commented this out for now, but can put back in to test the accuracy after training has completed. 


		#NOTE: Since the optimizer operation is not being ran, the weights are not being changed. 
					
		
			#Setting up the correct and accuracy operations to allow for the accuracy calculation (see ppt. for more detail).
			#correct=tf.equal(tf.argmax(prediction,1), tf.argmax(Labels_Placeholder,1))
			#accuracy=tf.reduce_mean(tf.cast(correct,'float'))
		
			#Printing a message indicating the accuracy the ANN achieved as well as how many epochs it required.
			#print "%f,%d" %(MaxAccuracyAchieved,EpochMaxAccuracy)

		
#############################################################################################################################################






#############################################################################################################################################
#STEP FIVE: RUNNING THE CODE


train_neural_network(Input_Placeholder)

 
#############################################################################################################################################
