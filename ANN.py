import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
import numpy as np
import sys
import math
import generator
import time

#Command Line Arguments:
#1) Dimensionality
#2) Num Points
#3) OS_mu
#4) OS_sigma
#5) IS_mu
#6) IS_sigma
#7) num_Nodes
#8) num_epochs
#9) num_layers

def print_accuracy(accuracies, end_time):

	filename = "results/" + "_".join([str(num_layers), str(Dimensionality), str(Num_Points), str(OS_mu), str(OS_sigma), str(IS_mu), str(IS_sigma), str(num_Nodes), str(num_epochs)]) + ".csv"
	f = open(filename, "w")
	print("Writing to " + filename)

	f.write("layers = " + str(num_layers) + "\n")
	f.write("n = " + str(Dimensionality) + "\n")
	f.write("points = " + str(Num_Points) + "\n")
	f.write("Inner Sphere: Mu = " + str(IS_mu) + "\n")
	f.write("Inner Sphere: Sigma = " + str(IS_sigma) + "\n")
	f.write("Outer Sphere: Mu = " + str(OS_mu) + "\n")
	f.write("Outer Sphere: Sigma = " + str(OS_sigma) + "\n")
	f.write("nodes per layer = " + str(num_Nodes) + "\n")
	f.write("epochs = " + str(num_epochs) + "\n")
	f.write("Computation time: " + str(end_time - start_time) + "\n")
	f.write("-----------------------------------\n")

	for i in range(len(accuracies)):
		line = str(i) + ", " + str(accuracies[i]) + "\n"
		f.write(line)
	
	f.close()


#############################################################################################################################################
#STEP ONE: READ IN DATA

g=generator.Generator()
Dimensionality = int(sys.argv[1])
Num_Points = int(sys.argv[2])

OS_mu = int(sys.argv[3])
OS_sigma = int(sys.argv[4])

IS_mu=int(sys.argv[5])
IS_sigma=int(sys.argv[6])

num_Nodes=int(sys.argv[7])
num_epochs=int(sys.argv[8])
num_layers = int(sys.argv[9])

print("Generating outer sphere")
#Generating the outer sphere points
OuterSphereArray=g.generate(OS_mu,OS_sigma,Dimensionality,Num_Points,1)

print("Generating inner sphere")
#Generating the inner sphere points
InnerSphereArray=g.generate(IS_mu,IS_sigma,Dimensionality,Num_Points,0)


#Shuffling the data and splitting it up into the different arrays required.

print("Shuffling data")
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

	layers = list()

	#The weights are randomized for the first run through. The weights will be calibrated to minimize the cost function. 
	hidden_1_layer={'weights':tf.Variable(tf.random_normal([Dimensionality,num_Nodes])),'biases':tf.Variable(tf.random_normal([num_Nodes]))}

	#This preforms the actual manipulation of the data. The data is multiplied by the weights, and then the biases are added. This gives the nodes something to work with (for their activation function)
	l1=tf.add(tf.matmul(data,hidden_1_layer['weights']),hidden_1_layer['biases'])

	#Takes the output of the (input*weights) + biases of layer one and puts it through a sigmoid function (to get the input to the output nodes). This is the output of the first layer that is to be fed into the output layer. The explicit function is: 1/(1+exp(-x))
	l1=tf.nn.sigmoid(l1)

	layers.append(l1)

	for i in range(1, num_layers):

		#The weights are randomized for the first run through. The weights will be calibrated to minimize the cost function. 
		layer = {'weights':tf.Variable(tf.random_normal([num_Nodes,num_Nodes])),'biases':tf.Variable(tf.random_normal([num_Nodes]))}

		#This preforms the actual manipulation of the data. The data is multiplied by the weights, and then the biases are added. This gives the nodes something to work with (for their activation function)
		l = tf.add(tf.matmul(layers[i-1], layer['weights']),layer['biases'])

		l=tf.nn.sigmoid(l)
	
		layers.append(l)

	#These are the weights that are leading into the two output nodes.
	output_layer={'weights':tf.Variable(tf.random_normal([num_Nodes,n_classes])),'biases':tf.Variable(tf.random_normal([n_classes]))}

	output=tf.add(tf.matmul(layers[-1],output_layer['weights']),output_layer['biases'])

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

	#Setting up the accuracy array to hold the ANN's accuracy per epoch.
	AccuracyArray=np.zeros(num_epochs)
		

	with tf.device('/device:GPU:0'):	
		#Now actually running the ANN.
		with tf.Session(config=config) as sess:

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

				f = open("current_progress.txt", 'w')
				line = str(epoch) + ", " + str(AccuracyArray[epoch]) + "\n"
				f.write(line)
				f.close()
		
			#Getting the max accuracy achieved as well as the epoch it achieved that accuracy.
			#MaxAccuracyAchieved=np.amax(AccuracyArray)
			#EpochMaxAccuracy=np.argmax(AccuracyArray)+1
	end_time = time.time()
	print_accuracy(AccuracyArray, end_time)
			

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

start_time = time.time()
train_neural_network(Input_Placeholder)

 
#############################################################################################################################################

