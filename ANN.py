import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import numpy as np
import sys
import math
from TensorGenerator import TensorGenerator



#First command line argument is the path to the file that stores the desired data.
#Second command line argument is the number of points for one of the shapes (Total number of points/2)
#Third command line argument is the dimensionality of the problem under consideration (i.e. N=?)


*****CHANGE HERE BY USING TF DATASET API IN ORDER TO PROCESS DATA. ASSIGN THE PREPROCCESSING TO THE CPU TO FREE UP THE GPU(S) FOR TRAINING.THEREFORE, SPECIFY THAT THE CPU IS TO BE USED TO FORMAT THE DATA IN A TF.TENSOR OBJECT SO THAT THE ANN CAN WORK WITH IT*****
#############################################################################################################################################
#STEP ONE: READ IN DATA. Using tf.device to specify that the CPU is used to preprocess the data.



#STEPS:
#1) Assign the directions to the cpu via the tf.device statement
#2) Set up a queue of file names (can put more files in here if desired later on)
#3) Initialize a tf.TextLineReader to read the data from the CSV
#4) Tell the reader what file to read via a read statement (the file is automatically dequeued once it has hit the end of file) 
#5) Use tf.decode_csv to go through one line and assign each comma seperated value to a tensor object.  
#6) Customizing the configuration of the data in the list and stacking all of the tensor objects in the list into one big tensor ([Coord1,Coord2,...,CoordN,[Label]]) to get it ready for addition to the Dataset. 
#7) If it is the first iteration (that is, it is the first line being read) then create the Dataset structure (named InputAndLabels) with the tensor that contains all the coordinates and the label as the first entry. If it is not the first iteration, just concatenate the tensorwith all the CSV information to the pre-existing Dataset object (which is named InputAndLabels)
#8) Running a session to format all of the data.

FirstRun=true;
Dimensionality=sys.argv[3]


#Step 1)
with tf.device('/cpu:0'):
	
	#Step 2)
	#Can put in more files if desired here.
	QueueOfFiles=tf.train.string_input_producer([sys.argv[1]])

	#Step 3)
	#Setting up a reader operation.	
	Reader=tf.TextLineReader()
	
	#Num Data Points variable to keep track of how many data points to expect.
	NumDataPoints=int(sys.argv[2])


	#Step 4)
	#Outputs used to identify file
	_,Row=Reader.read(QueueOfFiles)

	


	#Setting up default array so the reader knows what to expect. Using a negative number much larger than what we use for troubleshooting later.
	GenDataPoint=tf.constant(-9999,dtype=tf.float32)
	GenLabel=tf.constant(-1,dtype=tf.int32)
	Default=[]
	for x in range(0,NumDataPoints):
		Default.append[[GenDataPoint]]
	Default.append([[GenLabel]])
	

	#Step 5)
	#Setting up an operation that reads a single line from the CSV. This returns a list of length N+1 that contains all of the elements in one line of the CSV. That is: Coord1,Coord2,Coord3,...Label.
	CSV_Values=tf.decode_csv(Row,record_defaults=Default)

	#Step 6)
	#Configuring the lists
	EndPoint=length(InputData)-2
	DataPointList=CSV_Values[:EndPoint]
	Label=CSV_Values[EndPoint:]
	
	#Format of NewList is [Coord1,Coord2,Coord3,...CoordN,[Label]]
	NewList=DataPointList.append([Label])


	#Step 7)
	#Stacking all of the returned data into one tensor object. This tensor object will be put into a Dataset object that contains all of the coordinates we are training the ANN on and the Label in its own category as a seperate list (see NewList comment).
	OneCompleteCoordAndLabel=tf.stack(NewList)
	

	if FirstRun:
		InputAndLabels=Dataset.from_tensors(OneCompleteCoordAndLabel)
		FirstRun=false
	else:
		InputAndLabels.concatenate(OneCompleteCoordAndLabel)


		
	#Step 8)
	with tf.Session() as sess:
		#Starting all the threads on the graph so that QueueOfFiles is filled.
		coord=tf.train.Coordinator()
		threads=tf.train.start_queue_runners(coord=coord)

		
		for i in range(0,NumDataPoints*2): 
			#Getting the whole list of tensors.			

			DataGen=sess.run([OneCompleteCoordAndLabel])
			if i==0:
                		TrainingInputAndLabels=Dataset.from_tensors(DataGen)
                		FirstRun=false
        		else:
				if i<=(NumDataPoints*2)*0.8:
                			TrainingInputAndLabels.concatenate(DataGen)
				else:
					if i==((NumDataPoints*2)*0.8):
						ValidationInputAndLabels.Dataset.from_tensors(DataGen)
					else:
						ValidationInputAndLabels.concatenate(DataGen)
				


		

#############################################################################################################################################





*****BASIC SETUP OF THE ANN. LEAVE AS IS*****
#############################################################################################################################################
#STEP TWO: SET UP THE CONSTANTS OF THE NN


#One hidden layer only
num_Nodes=10

#Number of output nodes
n_classes=2


#x is input (the N dimensional coordinate). y is labels.
x=tf.placeholder('float')
y=tf.placeholder('float')


#############################################################################################################################################





*****BASIC SETUP OF THE ANN. LEAVE AS IS*****
#############################################################################################################################################
#STEP THREE: SETTING UP THE ACTUAL LAYER(S) OF THE ANN.


#Setting up the computation graph of the ANN. 
def neural_network_model(data):

	#The weights are randomized for the first run through. The weights will be calibrated to minimize the cost function. 
	hidden_1_layer={'weights':tf.Variable(tf.random_normal([Dimensionality,num_Nodes])),'biases':tf.Variable(tf.random_normal([num_Nodes]))}
	
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




#Assuming the dataset is made from here on out.
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


	#Setting up the accuracy array to hold the ANN's accuracy per epoch.
        AccuracyArray=np.zeros(num_epochs)
		

	#Making the iterator
	iterator=InputAndLabels.make_initializable_iterator()
	next_element=iterator.get_next()


	#Making the constants to feed the iterator.
	TrainingMaxVal=tf.constant	


	#Now actually running the ANN.
	with tf.Session() as sess:

		#Initializes the variables (the weights and biases of the first hidden layer and the output layer) that were set up earlier. 
		sess.run(tf.global_variables_initializer())


		#THIS IS THE TRAINING

		#Setting up the correct and accuracy operations to be used later on.
                correct=tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
                accuracy=tf.reduce_mean(tf.cast(correct,'float'))

	
			




		#Need iterating/training code here




	
		#Getting the max accuracy achieved as well as the epoch it achieved that accuracy.
		MaxAccuracyAchieved=np.amax(AccuracyArray)
		EpochMaxAccuracy=np.argmax(AccuracyArray)+1
			
			

#############################################################################################################################################



#Slight modification of how accuracy is determined needed
#############################################################################################################################################
#STEP FOUR: TESTING THE ACCURACY OF THE NEURAL NETWORK 


		#NOTE: Since the optimizer operation is not being ran, the weights are not being changed. 
					
		
		#Setting up the correct and accuracy operations to allow for the accuracy calculation (see ppt. for more detail).
		correct=tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
		accuracy=tf.reduce_mean(tf.cast(correct,'float'))
		
		#Writing results to a file
		FileName=str(sys.argv[1])
		with open(FileName,"a") as f:
			f.write(MaxAccuracyAchieved,EpochMaxAccuracy)
		 
		
#############################################################################################################################################




#Good to go 
#############################################################################################################################################
#STEP FIVE: RUNNING THE CODE


train_neural_network(x)

 
#############################################################################################################################################
