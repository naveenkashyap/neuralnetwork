import tensorflow as tf



class CSV_to_DS()
	

	def __init__(Dimensionality,FilePathName):
		
		#Generating the array of default values for tf.decode_csv later on.
		self.DefaultArray=self.GenerateDefaultArray(Dimensionality)

		#Making the FIFO queue of file names to go through when making the dataset. 
		self.FileQueue=self.GenerateFileQueue(FilePathName)

		#Setting up a TextLineReader to bring in the lines from the CSV
		Reader=tf.TextLineReader()

		#An operation that will read one line from a file in the Queue. Will move on to the next file in the queue when needed.
		_,Row=Reader.read(self.FileQueue)

		#Problem with this line, Row is not reading in correctly, not sure why.
		#Takes the line from reader and breaks it up into the structure defined by record_defaults. The current default structure is: [Coord1,Coord2,...CoordN,[OuterSphereLabel,InnerSphereLabel]]
		self.CSVLineToTensorList=tf.decode_csv(Row,record_defaults=self.DefaultArray)		

		self.ComputationGraph=tf.get_default_graph()


		

	#Generating the default array that tells tf.decode_csv what data types to expect when reading a line from the csv and default values to use if an expected part of a line is missing.
	def GenerateDefaultArray(self,Dimensionality):
		#Building the default array for the tf.decode_csv line. 
		GenDataPoint=tf.constant(-9999,dtype=tf.float32)
		GenLabel=[tf.constant(-1,dtype=tf.int32),tf.constant(-1,dtype=tf.int32)]
		Default=[]
		for x in range(0,int(dimensionality):
			Default.append(GenDataPoint)
		Default.append([GenLabel])	
		
		return Default


	#To-do:Make more general (pass in a list of path names and loop over the length of the list to add all the files to queue. For now, since we're just using one file, just expecting one string "FilePathName")
	def GenerateFileQueue(self,FilePathName):
		
		#Setting up the Queue of files for the Reader to read from. Max capacity of Queue is 10, and it is a queue of string tensors.
		QueueOfFiles=tf.FIFOQueue(10,tf.string)

		#Temporary fix. Need to make this cleaner for later on so input filename goes straight into queue of files so that we can use multiple files later on.
		FileName=tf.constant(FilePathName)
		
		#Add the tf.string tensor to the QueueOfFiles.
		QueueOfFiles.enqueue(FileName)

		#Close the QueueOfFiles so nothing can be added to it later on. 
		QueueOfFiles.close()

		return QueueOfFiles


	#Generating the datasets using the user entered information. PercentOfTrainingData is a decimal value that indicates how much of the total data set to use as training (the rest will be used for testing).
	def GenerateDatasets(self,TotalNumPoints,PercentOfTrainingData):

	with tf.Session(self.ComputationGraph) as sess:
		sess.run(tf.global_variables_initializer())
	

		#Making the empty training and testing datasets.
		TrainingDataset=tf.Data.Dataset()
		TestingDataset=tf.Data.Dataset()

		#Constructing the datasets
		for i in range(TotalNumPoints):
		
			#Getting one line from the CSV as a list of tensors in the format determined by record_defaults.
			DataCoordsAndLabels=sess.run(self.CSVLineToTensorList)
	
			#Now cutting up the list of input information
			IndexOfLastCoord=len(DataCoordsAndLabels)-2
			DataPointList=DataCoordsAndLabels[:IndexOfLastCoord]
			Label=DataCoordsAndLabels[IndexOfLastCoord:]

			#Stacking all the data points into one tensor
			DataPointTensor=tf.stack(DataPointList)

			#Concatenating the label to the tensor that contains all of the data points. The shape of the result is: [Coord1,Coord2,...CoordN,[OuterSphereLabel,InnerSphereLabel]] where Coord1 to CoordN are one tensor object and the set of labels are one tensor object as well.
			OneDataPointAndLabel=tf.concat(DataPointTensor,Label)
				

			DataPointAndLabel=sess.run(OneDataPointAndLabel)
		

			#Splitting into training and testing data sets.
			if i <= (TotalNumPoints)*PercentOfTrainingData:
				TrainingDataset=TrainingDataset.concatenate(DataPointAndLabel)	
			else:
				TestingDataset=TestingDataset.concatenate(DataPointAndLabel)
			


		return TrainingDataset,TestingDataset
