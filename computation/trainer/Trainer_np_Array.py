import tensorflow as tf
import numpy as np


class Trainer_np_Array():


	def __init__(self):
	
		#Placeholders to feed data through later on
		self.Input_Placeholder=tf.placeholder('float',name="Input_Placeholder")
		self.KnownLabels_Placeholder=tf.placeholder('float',name="Label_Placeholder")
		
		#Placeholders to feed the trained dictionaries through.
		self.HiddenLayerOneWeights=tf.placeholder('float',name="HL1_Weights_Placeholder")
		self.HiddenLayerOneBiases=tf.placeholder('float',name="HL1_Biases_Placeholder")

		self.OutputLayerWeights=tf.placeholder('float',name="Output_Layer_Weights_Placeholder")
		self.OutputLayerBiases=tf.placeholder('float',name="Output_Layer_Biases_Placeholder")

                #Establishing the activation functions of HL1 and the output layer.
		l1=tf.add(tf.matmul(self.Input_Placeholder,self.HiddenLayerOneWeights,name="Layer_One_Multiplication"),self.HiddenLayerOneBiases,name="Layer_One_Addition")
		l1=tf.nn.sigmoid(l1,name="Layer_One_Activation_Function")
		output=tf.add(tf.matmul(l1,self.OutputLayerWeights,name="Output_Layer_Multiplication"),self.OutputLayerBiases,name="Output_Layer_Addition")
		self.output=tf.nn.softmax(logits=output,name="Output_Layer_Activation_Function")

		#Setting up the CostFunction operation and the Optimizer operation.
		self.CostFunction=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output,labels=self.KnownLabels_Placeholder),name="Cost_Function")
	
		self.ComputationGraph=tf.get_default_graph()	

	#HiddenLayer1Dict and OutputLayerDict are dictionaries of Weights and Biases, TrainingDataPoints and TrainingLabels are np arrays of data points and labels.
	def Train(self,HiddenLayer1Dict,OutputLayerDict,TrainingDataPoints,TrainingLabels,BatchSize):

	
		with tf.device('/device:GPU:0'):
		
			with tf.Session(graph=self.ComputationGraph) as sess:
		
				sess.run(tf.global_variables_initializer())
				

				#Batching the input data
				i=0			
				while i < len(TrainingDataPoints):
					start=i
					end=i+BatchSize
					DataPointsBatch=TrainingDataPoints[start:end]
					LabelBatch=TrainingLabels[start:end]



					_,ANNOutput=sess.run([tf.train.AdamOptimizer().minimize(self.CostFunction,name="Optimizer"),self.output],feed_dict={self.Input_Placeholder: DataPointsBatch, self.KnownLabels_Placeholder:LabelBatch,self.HiddenLayerOneWeights: HiddenLayer1Dict["Weights"],self.HiddenLayerOneBiases: HiddenLayer1Dict["Biases"],self.OutputLayerWeights:OutputLayerDict["Weights"],self.OutputLayerBiases: OutputLayerDict["Biases"]})

					i+=BatchSize
			
		#Returns the altered HL1 and OutputLayer dictionaries. 				
		return (HiddenLayer1Dict, OutputLayerDict)
