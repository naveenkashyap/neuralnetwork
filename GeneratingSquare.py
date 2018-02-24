import numpy as np
import sys
import math



#############################################################################################################################################################STEP ONE: READING IN/CONFIGURING THE NECESSARY VARIABLES/ARRAYS.
#First arg is total number of points desired
#Second arg is length of one side
#Third arg is the sigma for the distribution 
#Fourth arg is how the number of distributions per side


#Setting up the needed variables
#Reading in command line arguments 
TotalNumPoints=float(sys.argv[1])
LengthOfSide=float(sys.argv[2])
DistSigma=float(sys.argv[3])
DistPerSide=int(sys.argv[4])


#This along with the rounded down points per side alows for the desired number of distributions to be set up on each side.
StepSize=LengthOfSide/(DistPerSide-1)


#Setting up extra variables
NumberOfSides=4.


#Points per side=(Total Points)/(Number of Sides).
PointsPerSide=math.floor((TotalNumPoints/NumberOfSides))


#Points per distribution=(Total Number of Points)/((Number Of Distributions Per Side) *(Number Of Sides))
PointsPerDist=int(math.floor(TotalNumPoints/(DistPerSide*NumberOfSides)))


#Setting up the necessary storage.
#A 2D array that will contain the points of the square in cartesian coordinates [2D for now, can generalize dimensions by changing the size]
EntireSquarePoints=np.zeros((int(TotalNumPoints),2))



############################################################################################################################################################




############################################################################################################################################################
#STEP TWO: GENERATING THE POINTS ALONG THE LEFTMOST EDGE OF THE SQUARE
#This generates one distribution, copies it all along the edge, then reflects it across the y axis in order to get the other side of points



#Setting up storage
TotalXDistPoints=np.zeros((int(TotalNumPoints/NumberOfSides),2))
XOneDistPoints=np.zeros((int(PointsPerDist),2))


#Starting at the bottom left corner of the square
StartingXCoord=-1*(LengthOfSide/2)
StartingYCoord=-1*(LengthOfSide/2)
DistNumber=1


#Generating the first distribution
XOneDistPoints[:,1]=StartingYCoord
for i in range(len(XOneDistPoints)):
	XOneDistPoints[i,0]=np.random.normal(StartingXCoord,DistSigma)


#Copying the points from the first dist into TotalXDistPoints
for i in range(len(XOneDistPoints)):
	TotalXDistPoints[i]=XOneDistPoints[i]


#Translating the distribution up along the leftmost vertical edge
for i in range(int(PointsPerDist),int(len(TotalXDistPoints))):
	#If the next step would still be within the bounds of the square, copy the x-coordinates and shift up the y-coordinates
	if XOneDistPoints[i%PointsPerDist,1]+DistNumber*StepSize<=(-1*StartingYCoord):
		#Shifting the dist up by a multiple of step size and copying it into TotalXDistPoints
		TotalXDistPoints[i,0]=XOneDistPoints[i%PointsPerDist,0]
		TotalXDistPoints[i,1]=XOneDistPoints[i%PointsPerDist,1]+DistNumber*StepSize
		if (i%PointsPerDist)==PointsPerDist-1:
			DistNumber=DistNumber+1


#If there are left over points (points that are supposed to be on an edge but the distributions have already hit the end of an edge [if we incremented anymore, the distributions would be centered off of the LHS of the square] ), put all those points on the end of the edge.
for i in range(len(TotalXDistPoints)):	
	#The x coordinate would be zero iff that location wasn't already put into a distribution. If this is the case, generating a new distribution at the end of the edge with new x-coordinates.
	if TotalXDistPoints[i,0]==0:
		TotalXDistPoints[i,1]=-1*StartingYCoord
		TotalXDistPoints[i,0]=np.random.normal(-1*StartingXCoord,DistSigma)


#Copying the points into the EntirePointsArray
for i in range(len(TotalXDistPoints)*2):
	#Copying over the flipped version
	if i >= len(TotalXDistPoints):
		EntireSquarePoints[i]=-1*TotalXDistPoints[i-(len(TotalXDistPoints))]
	#Copying the original version
	else:
		EntireSquarePoints[i]=TotalXDistPoints[i]



############################################################################################################################################################




############################################################################################################################################################
#STEP THREE: GENERATING THE POINTS ALONG THE TOP AND BOTTOM OF THE SQUARE (Now, the distribution is in the y direction)
#Going to generate one distribution, copy it all along the edge, then reflect that edge across the x axis in order to get the other side of points



#Setting up storage
TotalYDistPoints=np.zeros((int(TotalNumPoints/NumberOfSides),2))
YOneDistPoints=np.zeros((int(PointsPerDist),2))
DistNumber=1


#Generating the first distribution
YOneDistPoints[:,0]=StartingXCoord
for i in range(len(YOneDistPoints)):
        YOneDistPoints[i,1]=np.random.normal(StartingYCoord,DistSigma)


#Copying the points from the first dist into TotalYDistPoints
for i in range(len(YOneDistPoints)):
        TotalYDistPoints[i]=YOneDistPoints[i]


#Shifting the distribution along the horizontal lower edge.
for i in range(int(PointsPerDist),int(len(TotalYDistPoints))):
	#If the x-coordinates are still on the square
	if YOneDistPoints[i%PointsPerDist,0]+DistNumber*StepSize<=(-1*StartingXCoord):
                #Shifting the dist a multiple of step size to the right and copying it into TotalYDistPoints
                TotalYDistPoints[i,1]=YOneDistPoints[i%PointsPerDist,1]
                TotalYDistPoints[i,0]=YOneDistPoints[i%PointsPerDist,0]+DistNumber*StepSize
                if (i%PointsPerDist)==PointsPerDist-1:
                        DistNumber=DistNumber+1


#If there are left over points (points that are supposed to be on an edge but the distributions have already hit the end of an edge [if we incremented anymore, the distributions would be centered off of the bottom side of the square] ), put all those points on the end of the edge.
for i in range(len(TotalYDistPoints)):
         #The y-coordinate would be zero iff the point was not assigned to a distribution
	 if TotalYDistPoints[i,1]==0:
                TotalYDistPoints[i,0]=-1*StartingXCoord
                TotalYDistPoints[i,1]=np.random.normal(StartingYCoord,DistSigma)




#Copying the points into the EntirePointsArray
for i in range(len(TotalYDistPoints)*2):
        #Copying over the flipped version
        if i >= len(TotalYDistPoints):
                EntireSquarePoints[i+len(TotalXDistPoints)*2]=-1*TotalYDistPoints[i-(len(TotalYDistPoints))]

	#Copying over the original version
        else:
                EntireSquarePoints[i+len(TotalXDistPoints)*2]=TotalYDistPoints[i]



#Fixing any unassigned points (in case the all the points don't fit cleanly into the desired number of distributions) by assigning the left over positions to the bottom left hand corner. 
for j in range(len(EntireSquarePoints)):
	if EntireSquarePoints[j,0]==0 and EntireSquarePoints[j,1]==0:
		EntireSquarePoints[j,0]=StartingXCoord
		EntireSquarePoints[j,1]=np.random.normal(StartingYCoord,DistSigma)


############################################################################################################################################################





############################################################################################################################################################
#STEP FOUR: WRITE TO FILE


TestFile=open("FirstGeneration.txt","w")
for i in range(int(TotalNumPoints)):
	TestFile.write(str(EntireSquarePoints[i,0])+","+str(EntireSquarePoints[i,1])+"\n")
TestFile.close()



############################################################################################################################################################
