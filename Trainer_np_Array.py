import tensorflow as tf
import numpy as np


class Trainer_np_Array():


	def __init__(self,Dimensionality,num_Nodes_HL1,num_Output_Nodes):
	
		#Placeholders to feed data through later on
		self.Input_Placeholder=tf.placeholder('float',name="Input_Placeholder")
		self.KnownLabels_Placeholder=tf.placeholder('float',name="Label_Placeholder")
		

		self.HiddenLayer1Dict={"Weights":tf.Variable(tf.random_normal([Dimensionality,num_Nodes_HL1])),"Biases":tf.Variable(tf.random_normal([num_Nodes_HL1]))}
		self.OutputLayerDict={"Weights":tf.Variable(tf.random_normal([num_Nodes_HL1,num_Output_Nodes])),"Biases":tf.Variable(tf.random_normal([num_Output_Nodes]))}


                #Establishing the activation functions of HL1 and the output layer.
		l1=tf.add(tf.matmul(self.Input_Placeholder,self.HiddenLayer1Dict["Weights"],name="Layer_One_Multiplication"),self.HiddenLayer1Dict["Biases"],name="Layer_One_Addition")
		l1=tf.nn.sigmoid(l1,name="Layer_One_Activation_Function")
		output=tf.add(tf.matmul(l1,self.OutputLayerDict["Weights"],name="Output_Layer_Multiplication"),self.OutputLayerDict["Biases"],name="Output_Layer_Addition")
		self.output=tf.nn.softmax(logits=output,name="Output_Layer_Activation_Function")

		#Setting up the CostFunction operation and the Optimizer operation.
		CostFunction=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.output,labels=self.KnownLabels_Placeholder,name="Cost_Function"))
		self.Optimizer=tf.train.AdamOptimizer().minimize(CostFunction,name="Optimizer")	

		TestingCorrectness=tf.equal(tf.argmax(self.output,1), tf.argmax(self.KnownLabels_Placeholder,1))
                self.Accuracy=tf.reduce_mean(tf.cast(TestingCorrectness,'float'))



		self.ComputationGraph=tf.get_default_graph()	

	#TrainingDataPoints and TrainingLabels are np arrays of data points and labels.
	def Train(self,TrainingDataPoints,TrainingLabels,BatchSize,TestingDataPoints,TestingLabels):

		
		with tf.Session(graph=self.ComputationGraph) as sess:
		
			sess.run(tf.global_variables_initializer())
				

			#Batching the input data
			i=0			
			while i < len(TrainingDataPoints):
				start=i
				end=i+BatchSize
				DataPointsBatch=TrainingDataPoints[start:end]
				LabelBatch=TrainingLabels[start:end]



				ANNOutput,_=sess.run([self.output,self.Optimizer],feed_dict={self.Input_Placeholder: DataPointsBatch, self.KnownLabels_Placeholder:LabelBatch})

				i+=BatchSize


		

			AccuracyValue=sess.run(self.Accuracy,feed_dict={self.Input_Placeholder: TestingDataPoints, self.KnownLabels_Placeholder:TestingLabels})
			print AccuracyValue



	def get_Layers(self):
		return (self.HiddenLayer1Dict,self.OutputLayerDict)	
