import numpy as np
import math



class GeneratingSquare():



	#Note: TotalNumPoints, LengthOfSide, and DistSigma should all be floats. DistPerSide is an int.
	def generate_points(self,TotalNumPoints,LengthOfSide,DistSigma,DistPerSide):

		NumberOfSides=4

		XDistPoints = self.generate_vertical_edges(float(TotalNumPoints),float(LengthOfSide),float(DistSigma),int(DistPerSide),NumberOfSides)
		YDistPoints = self.generate_horizontal_edges(float(TotalNumPoints),float(LengthOfSide),float(DistSigma),int(DistPerSide),NumberOfSides)

		EntireSquarePoints=np.append(XDistPoints,YDistPoints,axis=0)
		
		#Error correction    
		#Fixing any unassigned points (in case the all the points don't fit cleanly into the desired number of distributions) by assigning the left over positions to the bottom left hand corner.
		for j in range(len(EntireSquarePoints)):
			if EntireSquarePoints[j,0]==0 and EntireSquarePoints[j,1]==0:
				EntireSquarePoints[j,0]=-1*(float(LengthOfSide)/2.0)
				EntireSquarePoints[j,1]=np.random.normal(-1*(float(LengthOfSide)/2.0),DistSigma)

		LabeledPoints=self.label_points(EntireSquarePoints,float(TotalNumPoints))

		return LabeledPoints
	


	#This generates one distribution, copies it all along the edge, then reflects it across the y axis in order to get the other side of points
	def generate_vertical_edges(self,TotalNumPoints,LengthOfSide,DistSigma,DistPerSide,NumberOfSides):

		#This along with the rounded down points per side alows for the desired number of distributions to be set up on each side.
		StepSize=float(LengthOfSide)/(DistPerSide-1)

		#Points per side=(Total Points)/(Number of Sides).
		PointsPerSide=math.floor((TotalNumPoints/NumberOfSides))

		#Points per distribution=(Total Number of Points)/((Number Of Distributions Per Side)*(Number Of Sides))
		PointsPerDist=int(math.floor(TotalNumPoints/(DistPerSide*NumberOfSides)))

		#Setting up storage
		OneSideXPoints=np.zeros((int(TotalNumPoints/NumberOfSides),2))
		XOneDistPoints=np.zeros((int(PointsPerDist),2))
		VerticalEdgePoints=np.zeros((int(TotalNumPoints)/2,2))	

		#Starting at the bottom left corner of the square
		StartingXCoord=-1*(LengthOfSide/2.0)
		StartingYCoord=-1*(LengthOfSide/2.0)
		DistNumber=1

		#Generating the first distribution
		XOneDistPoints[:,1]=StartingYCoord
		for i in range(len(XOneDistPoints)):
			XOneDistPoints[i,0]=np.random.normal(StartingXCoord,DistSigma)

		#Copying the points from the first dist into OneSideXPoints
		for i in range(len(XOneDistPoints)):
			OneSideXPoints[i]=XOneDistPoints[i]

		#Translating the distribution up along the leftmost vertical edge
		for i in range(int(PointsPerDist),int(len(OneSideXPoints))):
			#If the next step would still be within the bounds of the square, copy the x-coordinates and shift up the y-coordinates
			if XOneDistPoints[i%PointsPerDist,1]+DistNumber*StepSize<=(-1*StartingYCoord):
				#Shifting the dist up by a multiple of step size and copying it into OneSideXPoints
				OneSideXPoints[i,0]=XOneDistPoints[i%PointsPerDist,0]
				OneSideXPoints[i,1]=XOneDistPoints[i%PointsPerDist,1]+DistNumber*StepSize
				if (i%PointsPerDist)==PointsPerDist-1:
					DistNumber=DistNumber+1

			#If there are left over points (points that are supposed to be on an edge but the distributions have already hit the end of an edge [if we incremented anymore, the distributions would be centered off of the LHS of the square] ), put all those points on the end of the edge.
		for i in range(len(OneSideXPoints)):	
			#The x coordinate would be zero iff that location wasn't already put into a distribution. If this is the case, generating a new distribution at the end of the edge with new x-coordinates.
			if OneSideXPoints[i,0]==0:
				OneSideXPoints[i,1]=-1*StartingYCoord
				OneSideXPoints[i,0]=np.random.normal(-1*StartingXCoord,DistSigma)

		#Copying the points into the VerticalEdgePoints Array
		for i in range(len(OneSideXPoints)*2):
			#Copying over the flipped version
			if i >= len(OneSideXPoints):
				VerticalEdgePoints[i]=-1*OneSideXPoints[i-(len(OneSideXPoints))]
			#Copying the original version
			else:
				VerticalEdgePoints[i]=OneSideXPoints[i]

		return VerticalEdgePoints 



	#Generate one distribution, copy it all along the edge, then reflect that edge across the x axis in order to get the other side of points
	def generate_horizontal_edges(self,TotalNumPoints,LengthOfSide,DistSigma,DistPerSide,NumberOfSides):
	
		#This along with the rounded down points per side alows for the desired number of distributions to be set up on each side.
		StepSize=float(LengthOfSide)/(DistPerSide-1)

		#Points per side=(Total Points)/(Number of Sides).
		PointsPerSide=math.floor((TotalNumPoints/NumberOfSides))

		#Points per distribution=(Total Number of Points)/((Number Of Distributions Per Side)*(Number Of Sides))
		PointsPerDist=int(math.floor(TotalNumPoints/(DistPerSide*NumberOfSides)))

		#Starting at the bottom left corner of the square
                StartingXCoord=-1*(LengthOfSide/2.0)
                StartingYCoord=-1*(LengthOfSide/2.0)
                DistNumber=1

		#Setting up storage
		OneSideYPoints=np.zeros((int(TotalNumPoints/NumberOfSides),2))
		YOneDistPoints=np.zeros((int(PointsPerDist),2))
		HorizontalEdgePoints=np.zeros((int(TotalNumPoints)/2,2))

		#Generating the first distribution
		YOneDistPoints[:,0]=StartingXCoord
		for i in range(len(YOneDistPoints)):
			YOneDistPoints[i,1]=np.random.normal(StartingYCoord,DistSigma)

		#Copying the points from the first dist into OneSideYPoints
		for i in range(len(YOneDistPoints)):
			OneSideYPoints[i]=YOneDistPoints[i]

		#Shifting the distribution along the horizontal lower edge.
		for i in range(int(PointsPerDist),int(len(OneSideYPoints))):
			#If the x-coordinates are still on the square
			if YOneDistPoints[i%PointsPerDist,0]+DistNumber*StepSize<=(-1*StartingXCoord):
				#Shifting the dist a multiple of step size to the right and copying it into OneSideYPoints
				OneSideYPoints[i,1]=YOneDistPoints[i%PointsPerDist,1]
				OneSideYPoints[i,0]=YOneDistPoints[i%PointsPerDist,0]+DistNumber*StepSize
				if (i%PointsPerDist)==PointsPerDist-1:
					DistNumber=DistNumber+1

		#If there are left over points (points that are supposed to be on an edge but the distributions have already hit the end of an edge [if we incremented anymore, the distributions would be centered off of the bottom side of the square] ), put all those points on the end of the edge.
		for i in range(len(OneSideYPoints)):
			 #The y-coordinate would be zero iff the point was not assigned to a distribution
			 if OneSideYPoints[i,1]==0:
				OneSideYPoints[i,0]=-1*StartingXCoord
				OneSideYPoints[i,1]=np.random.normal(StartingYCoord,DistSigma)

		#Copying the points into the HorizontalEdgePoints Array
		for i in range(len(OneSideYPoints)*2):
			#Copying over the flipped version
			if i >= len(OneSideYPoints):
				HorizontalEdgePoints[i]=-1*OneSideYPoints[i-(len(OneSideYPoints))]

			#Copying over the original version
			else:
				HorizontalEdgePoints[i]=OneSideYPoints[i]

		return HorizontalEdgePoints



	def label_points(self,SquarePoints,TotalNumPoints):
	
		#Setting up storage
		LabeledSquarePoints=np.zeros((int(TotalNumPoints),4))

		#Labeling Points
		for i in range(int(TotalNumPoints)):
			LabeledSquarePoints[i,0]=SquarePoints[i,0]
			LabeledSquarePoints[i,1]=SquarePoints[i,1]
			LabeledSquarePoints[i,2]=1
			LabeledSquarePoints[i,3]=0

		return LabeledSquarePoints
