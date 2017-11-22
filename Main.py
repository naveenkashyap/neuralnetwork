import tensorflow as tf
import numpy as np
import generator
from computation.trainer import Trainer_np_Array
from computation.trainer import Tester_np_Array
import sys
import math





def main():
	
	#Generating np arrays
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




	BatchSize=100
	num_Nodes_HL1=100
	num_Output_Nodes=2




	Tester=Tester_np_Array.Tester_np_Array()
	Trainer=Trainer_np_Array.Trainer_np_Array(Dimensionality,num_Nodes_HL1,num_Output_Nodes)

	HiddenLayer1Dict,OutputLayerDict=Trainer.get_Layers()

	for i in range(50):
		Trainer.Train(TrainingDataPoints,TrainingLabels,BatchSize,TestingDataPoints,TestingLabels)
	#Tester.test(HiddenLayer1Dict,OutputLayerDict,TestingDataPoints,TestingLabels,"DummyFile.txt")
	


if __name__ == "__main__":
	main()
