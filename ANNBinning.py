import numpy as np


#Have sucessfuly isolated the coords. 





BatchedDataArray=np.zeros((10,4,3))

#Assuming dataset has batches as columns, coords as rows, and the the actual data points as the depth of the array.
SizeOfArray=BatchedDataArray.shape


#Num Batches = num columns, which is the second index returned by BatchedDataArray.shape.
NumBatches=SizeOfArray[1]


#Setting up the arrays to store the results in
ISPredictionResults=np.zeros(NumBatches)
OSPredictionResults=np.zeros(NumBatches)
BatchCounter=0

#Setting up a dummy label array  to feed the LabelPlaceholder so the session will run. Assuming the max number of coords in any batch is less than 300.
#DummyLabelArray is being set up correctly
DummyLabel=[1,0]
DummyLabelArray=np.zeros((300,2))
for i in range(len(DummyLabelArray)):
	DummyLabelArray[i]=DummyLabel


for i in range(NumBatches):
	#Picking all rows of the ith column. At this point, have isolated a column (one batch). Am just going to feed the batch directly (b/c assuming len(BatchedDataArray[:,i]) is small).
	#Assuming that BatchArray is just pure data points w/o any placeholders.
	#Batches are being taken correctly
	BatchArray=np.array(BatchedDataArray[:,i])
	
	#Making the DummyLabelArraySlice have the same number of rows as the BatchArray.
	#DummyLabelArraySlice is being correctly constructed.
	DummyLabelArraySlice=DummyLabelArray[0:len(BatchArray),:]		

	#Now have the array output stored as an array called ANNOutput
#	ANNOutput=sess.run([prediction], feed_dict={Input_Placeholder: BatchArray, Labels_Placeholder: DummyLabelArraySlice})

			
	#Seperating the ANNOutput. These values are between 0 and 1.
	ISPredictionResults=ANNOutput[:,0]
	OSPredictionResults=ANNOutput[:,1]
	

	#Summing over all of the results w/in a bin, then dividing by the total number that was summed over.
	ISPredictionDataPoint=(np.sum(ISPredictionResults))/float(len(BatchArray))
	OSPredictionDataPoint=(np.sum(OSPredictionResults))/float(len(BatchArray))
	ISPredictionResults[BatchCounter]=ISPredictionDataPoint
	OSPredictionResults[BatchCounter]=OSPredictionDataPoint
	BatchCounter+=1

		
	#At the end of the day need to return/write to file ISPredictionResults and OSPredictionResults, that way we can graph them later.


#From here on out assuming the ISPredictionResults array and the OSPredictionResults Array are correctly filled. 
	#To-do: Write the results of ISPrediction and OSPrediction to file.
	#Compute the relative PDF values given the radii
	#Write the radii array to file.
	#Write the PDF results arrays to file. 	


ISPredictedProbArray=np.zeros(len(RadiiArray))
OSPredictedProbArray=np.zeros(len(RadiiArray))


	#Assuming the ISRelativeProbArray and the OSRelativeProbArray are correctly filled from here on out.
	#Computing the PDF Values for the given radii
for i in range(len(RadiiArray)):
	OSPDFValue=(1/math.sqrt(2*math.pi*math.pow(OS_sigma,2)))*math.exp(-(math.pow(RadiiArray[i]-OS_mu,2))/(2*math.pow(OS_sigma,2)))
	ISPDFValue=(1/math.sqrt(2*math.pi*math.pow(IS_sigma,2)))*math.exp(-(math.pow(RadiiArray[i]-IS_mu,2))/(2*math.pow(IS_sigma,2)))
	
	ISRelProb=(ISPDFValue)/(ISPDFValue+OSPDFValue)
	OSRelProb=(OSPDFValue)/(ISPDFValue+OSPDFValue)
	
	ISRelativeProbArray[i]=ISRelProb
	OSRelativeProbArray[i]=OSRelProb



#Now that all of the arrays (ISPredictionResults,OSPredictionResults,ISRelativeProbArray,OSRelativeProbArray,RadiiArray) are formed, time to write to file.


	#len(RadiiArray)=len(ISPredictionResults)=len(OSPredictionResults)=len(RelativeProbArray)=len(OSRelativeProbArray)=Number of Batches. Therefore,writing to all of these files in one loop.

	#Writing all the names of the files. 
	RadiiArrayFileName="results/"+"Radii_Values"+"_".join([str(OS_mu),str(OS_sigma),str(IS_mu),str(IS_sigma)])+".txt"
	ISPredictionFileName="results/"+"IS_Pred"+"_".join([str(num_Layers),str(Dimensionality),str(Num_Points),str(OS_mu), str(OS_sigma), str(IS_mu), str(IS_sigma), str(num_Nodes), str(num_epochs)])+".txt"
	OSPredictionFileName="results/"+"OS_Pred"+"_".join([str(num_Layers),str(Dimensionality),str(Num_Points),str(OS_mu), str(OS_sigma), str(IS_mu), str(IS_sigma), str(num_Nodes), str(num_epochs)])+".txt"
	ISRelativeProbFileName="results/"+"IS_Rel_Prob"+"_".join([str(num_Layers),str(Dimensionality),str(Num_Points),str(OS_mu), str(OS_sigma), str(IS_mu), str(IS_sigma), str(num_Nodes), str(num_epochs)])+".txt"
	OSRelativeProbFileName="results/"+"OS_Rel_Prob"+"_".join([str(num_Layers),str(Dimensionality),str(Num_Points),str(OS_mu), str(OS_sigma), str(IS_mu), str(IS_sigma), str(num_Nodes), str(num_epochs)])+".txt"



	#Opening all of the files.
	RadiiArrayFile=open(RadiiArrayFileName,"w")
	ISPredictionFile=open(ISPredictionFileName,"w")
	OSPredictionFile=open(OSPredictionFileName,"w")
	ISRelativeProbFile=open(ISRelativeProbFileName,"w")
	OSRelativeProbFile=open(OSRelativeProbFileName,"w")





	#Writing all of the arrays to file.
	for i in range(len(RadiiArray)):
		RadiiArrayFile.write(str(RadiiArray[i])+"\n")

		ISPredictionFile.write(str(ISPredictionResults[i])+"\n")
		OSPredictionFile.write(str(OSPredictionResults[i])+"\n")
		ISRelativeProbFile.write(str(ISRelativeProbArray[i])+"\n")
		OSRelativeProbFile.write(str(OSRelativeProbArray[i])+"\n")


	#Closing the files.
	RadiiArrayFile.close()
	ISPredictionFile.close()
        OSPredictionFile.close()
        ISRelativeProbFile.close()
        OSRelativeProbFile.close()
