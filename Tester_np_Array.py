import tensorflow as tf
import numpy as np



class Tester_np_Array():	



	def __init__(self):

		#Placeholders to feed data through later on
		self.Input_Placeholder=tf.placeholder('float',name="Input Placeholder")
		self.KnownLabels_Placeholder=tf.placeholder('float',name="Label Placeholder")
		
		#Placeholders to feed the trained dictionaries through.
		self.HiddenLayerOneWeights=tf.placeholder('float',name="HL1 Weights Placeholder")
	 	self.HiddenLayerOneBiases=tf.placeholder('float',name="HL1 Biases Placeholder")	

		self.OutputLayerWeights=tf.placeholder('float',name="Output Layer Weights Placeholder")
		self.OutputLayerBiases=tf.placeholder('float',name="Output Layer Biases Placeholder")   		

		#Establishing the activation functions of HL1 and the output layer.
		l1=tf.add(tf.matmul(self.Input_Placeholder,self.HiddenLayerOneWeights,name="Layer One Multiplication"),self.HiddenLayerOneBiases,name="Layer One Addition")
		l1=tf.nn.sigmoid(l1,name="Layer One Activation Function")
		output=tf.add(tf.matmul(l1,self.OutputLayerWeights,name="Output Layer Multiplication"),self.OutputLayerBiases,name="Output Layer Addition")
		output=tf.nn.softmax(logits=output,name="Output Layer Activation Function")


		TestingCorrectness=tf.equal(tf.argmax(output,1), tf.argmax(self.KnownLabels_Placeholder,1))
		self.Accuracy=tf.reduce_mean(tf.cast(TestingCorrectness,'float'))


		#Assigning the default graph to the specific instance of Tester. 
		self.ComputationGraph=tf.get_default_graph()



	#TestingDataPoints and TestingLables are np arrays.
	def test(HiddenLayer1Dict,OutputLayerDict,TestingDataPoints,TestingLabels,AccuracyFileName):

		with tf.device('/device:GPU:0'):
			
			with tf.Session(self.ComputationGraph) as sess:
				
				#Not batching b/c assuming sufficiently small data sizes 				
				AccuracyValue=self.Accuracy.eval({self.Input_Placeholder: TestingDataPoints, self.KnownLabels_Placeholder:TestingLabels,self.HiddenLayerOneWeights: HiddenLayer1Dict["Weights"],self.HiddenLayerOneBiases: HiddenLayer1Dict["Biases"],self.OutputLayerWeights:OutputLayerDict["Weights"],self.OutputLayerBiases: OutputLayerDict["Biases"])




			#Writing the results (the accuracy) to file
			with (AccuracyFileName,"a") as AccuracyFile:
				AccuracyFile.write(AccuracyValue)
				AccuracyFile.write(\n)

					
