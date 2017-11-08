import tensorflow as tf
import numpy as np



Class Tester():	



	def __init__(self,numHL1Nodes,numOutputNodes):

		#Placeholders to feed CSV data through later on
		Input_Placeholder=tf.placeholder('float',name="Input Placeholder")
		KnownLabels_Placeholder=tf.placeholder('float',name="Label Placeholder")


		#Establishing the activation functions of HL1 and the output layer.
		l1=tf.add(tf.matmul(Input_Placeholder,HiddenLayer1['weights'],name="Layer One Multiplication"),HiddenLayer1['biases'],name="Layer One Addition")
		l1=tf.nn.sigmoid(l1,name="Layer One Activation Function")
		output=tf.add(tf.matmul(l1,OutputLayer['weights'],name="Output Layer Multiplication"),OutputLayer['biases'],name="Output Layer Addition")
		output=tf.nn.softmax(logits=output,name="Output Layer Activation Function")

		TestingCorrectness=tf.cast(tf.equal(tf.argmax(output,name="Index of Output Max"),tf.argmax(KnownLabels_Placeholder,name="Index of Label Max"),name="Checking Correctness"),'float')

		#Assigning the default graph to the specific instance of Tester. 
		self.ComputationGraph=tf.get_default_graph()


	def test(Hidden1Layer,OutputLayer,TestingDataset,AccuracyFileName):
		
		iterator=TestingDataset.make_one_shot_iterator()
		next_element=iterator.get_next()

		
		#Initializing Results (this will keep a record of what points the ANN got right vs what points it got wrong later on.
		Results=[]


		with tf.device('/device:GPU:0'):
			
			with tf.Session(self.ComputationGraph) as sess:

				SizeTestingDataset=(tf.size(TestingDataset)).eval()

				for i in range(SizeTestingDataset):
					DataPointAndLabel=sess.run(next_element)
                                        
					#Splits DataPointAndLabel into two tensors: the DataPoint (a rank 1 tensor wth all of the coordinates in it) and Label (a rank zero tensor with either a 1 or a 0, depending on what group the input point belongs to)
					DataPoint,Label=tf.split(DataPointAndLabel,Dimensionality,0)
				

					#This will return a tensor that indicates if the ANN correctly guessed or not. 1=Correct Guess, 0=Incorrect Guess	
					CorrectOrNot=sess.run(TestingCorrectness,feed_dict={InputPlaceholder:DataPoint,KnownLabels_Placeholder:Label)
	
					#Gets the value from within the tensor and appends it to the Results list.
					Results.append(CorrectOrNot.eval())



			#Calculating the final accuracy by summing over all of the results and dividing by the total number of results to see how many the ANN was able to correctly determine.
			TotalCorrect=sum(Results)
			TotalPointsConsidered=len(Results)
			Accuracy=TotalCorrect/TotalPointsConsidered	

			with (AccuracyFileName,"a") as AccuracyFile:
				AccuracyFile.write(Accuracy)
				AccuracyFile.write('\n')

					
