import tensorflow as tf
import numpy as np
import sys



#Need the labels to be a 2-D tensor for the cost function to work

class Trainer():


	def __init__(self,numHL1Nodes,numOutputNodes):
	
			
		#Placeholders to feed CSV data through later on
		Input_Placeholder=tf.placeholder('float',name="Input Placeholder")
		KnownLabels_Placeholder=tf.placeholder('float',name="Label Placeholder")


		#Establishing the activation functions of HL1 and the output layer.
		l1=tf.add(tf.matmul(Input_Placeholder,HiddenLayer1['weights'],name="Layer One Multiplication"),HiddenLayer1['biases'],name="Layer One Addition")
		l1=tf.nn.sigmoid(l1,name="Layer One Activation Function")
		output=tf.add(tf.matmul(l1,OutputLayer['weights'],name="Output Layer Multiplication"),OutputLayer['biases'],name="Output Layer Addition")
		output=tf.nn.softmax(logits=output,name="Output Layer Activation Function")
		


		#Setting up the CostFunction operation and the Optimizer operation.
		CostFunction=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output,labels=KnownLabels_Placeholder),name="Cost Function")
		Optimizer = tf.train.AdamOptimizer().minimize(CostFunction,name="Optimizer")
	
	
		 #Assigning the default graph to the CompGraph object, then assigning CompGraph to the variable ComputationGraph to the instance of Tester.
                CompGraph=tf.get_default_graph()
                self.ComputationGraph=CompGraph	
	

	def Train(self,Dimensionality,HL1,OutputLayer,TrainingDataset,QualityAssuranceFileName):
	
		iterator=TrainingDataset.make_one_shot_iterator()
		next_element=iterator.get_next()


		with tf.device('/device:GPU:0'):
		
			with tf.Session(graph=self.ComputationGraph) as sess:
		
				sess.run(tf.global_variables_initializer())
	

				NumDataPoints=(tf.size(TrainingDataset)).eval()
				for i in range(NumDataPoints):
					DataPointAndLabel=sess.run(next_element)
				
					#Splits DataPointAndLabel into two tensors: the DataPoint (a rank 1 tensor wth all of the coordinates in it) and Label (a rank zero tensor with either a 1 or a 0, depending on what group the input point belongs to)		
					
					DataPoint,Label=tf.split(DataPointAndLabel,Dimensionality,0)
				
					#Run one data point through the ANN with its accompanying label and change the weights (weights being changed b/c optimizer operation is being ran).	
					_,ANNOutput=sess.run([Optimizer,output],feed_dict={Input_Placeholder: DataPoint, KnownLabels_Placeholder:Label})
					#Write the result of the ANN (the predicted probabilities of group membership) to a designated file. This is temporary, doing this to make sure the ANN works the way it should. For final draft will remove this. 	
						with open (QualityAssuranceFileName, "a") as OutputFile:
							OutputFile.write(DataPoint)
							OutputFile.write(ANNOutput)
							OutputFile.write('\n')
			
	#Returns the altered HL1 and OutputLayer dictionaries. 				
	return (HL1, OutputLayer)
