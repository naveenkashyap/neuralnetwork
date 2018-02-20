import numpy as np
import sys
import math



############################################################################################################################################################
#STEP ONE: READING IN/CONFIGURING THE NECESSARY VARIABLES/ARRAYS.
#First arg is total number of points desired
#Second arg is length of one side
#Third arg is the sigma for the distribution



#Setting up the needed variables
#Reading in command line arguments
TotalNumPoints=float(sys.argv[1])
LengthOfSide=float(sys.argv[2])
DistSigma=float(sys.argv[3])


#Need to figure out a way to dynamically compute step size based on TotalNumPoints and NumSides in a way s.t. all points will be used and sides of the square will be covered. For now, manually setting it to something that I have to change based on the number of points we want and the step size.
StepSize=0.5


#Setting up extra variables
NumberOfSides=4.


#Points per side=(Total Points)/(Number of Sides).
PointsPerSide=math.ceil((TotalNumPoints/NumberOfSides))


#Number of Distributions per Side=(Length of One Side)/(Step Size [size between distributions])
#Points per distribution=(Total Number of Points)/((Number Of Distributions Per Side) *(Number Of Sides))
DistPerSide=math.ceil(LengthOfSide/StepSize)
PointsPerDist=int(math.floor(TotalNumPoints/(DistPerSide*NumberOfSides)))


#Setting up the necessary storage.
#A 2D array that will contain the points of the square in cartesian coordinates [2D for now, can generalize dimensions by changing the size]
EntireSquarePoints=np.zeros((int(TotalNumPoints),2))



############################################################################################################################################################




############################################################################################################################################################
#STEP TWO: GENERATING THE POINTS ALONG THE LEFTMOST EDGE OF THE SQUARE
#This generates one distribution, copies it all along the edge, then reflects it across the y axis in order to get the other side of points



#Setting up storage
XOneSidePoints=np.zeros((int(TotalNumPoints/NumberOfSides),2))
XOneDistPoints=np.zeros((int(PointsPerDist),2))


#Starting at the bottom left corner of the square
StartingXCoord=-1*(LengthOfSide/2)
StartingYCoord=-1*(LengthOfSide/2)
DistNumber=1


#Generating the first distribution
XOneDistPoints[:,1]=StartingYCoord
for i in range(len(XOneDistPoints)):
        XOneDistPoints[i,0]=np.random.normal(StartingXCoord,DistSigma)


#Copying the points from the first dist into XOneSidePoints
for i in range(len(XOneDistPoints)):
        XOneSidePoints[i]=XOneDistPoints[i]


#Translating the distribution up along the leftmost vertical edge
for i in range(int(PointsPerDist),int(len(XOneSidePoints))):
        #If the next step would still be within the bounds of the square, copy the x-coordinates and shift up the y-coordinates
        if XOneDistPoints[i%PointsPerDist,1]+DistNumber*StepSize<=(-1*StartingYCoord):
                #Shifting the dist up by a multiple of step size and copying it into XOneSidePoints
                XOneSidePoints[i,0]=XOneDistPoints[i%PointsPerDist,0]
                XOneSidePoints[i,1]=XOneDistPoints[i%PointsPerDist,1]+DistNumber*StepSize
                 if (i%PointsPerDist)==PointsPerDist-1:
                        DistNumber=DistNumber+1


#If there are left over points (points that are supposed to be on an edge but the distributions have already hit the end of an edge [if we incremented anymore, the distributions would be centered off of the LHS of the square] ), put all those points on the end of the edge.
for i in range(len(XOneSidePoints)):
        #The x coordinate would be zero iff that location wasn't already put into a distribution. If this is the case, generating a new distribution at the end of the edge with new x-coordinates.
        if XOneSidePoints[i,0]==0:
                XOneSidePoints[i,1]=-1*StartingYCoord
                XOneSidePoints[i,0]=np.random.normal(-1*StartingXCoord,DistSigma)


#Copying the points into the EntirePointsArray
for i in range(len(XOneSidePoints)*2):
        #Copying over the flipped version
        if i >= len(XOneSidePoints):
                EntireSquarePoints[i]=-1*XOneSidePoints[i-(len(XOneSidePoints))]
        #Copying the original version
        else:
                EntireSquarePoints[i]=XOneSidePoints[i]



############################################################################################################################################################




############################################################################################################################################################
#STEP THREE: GENERATING THE POINTS ALONG THE TOP AND BOTTOM OF THE SQUARE (Now, the distribution is in the y direction)
#Going to generate one distribution, copy it all along the edge, then reflect that edge across the x axis in order to get the other side of points



#Setting up storage
YOneSidePoints=np.zeros((int(TotalNumPoints/NumberOfSides),2))
YOneDistPoints=np.zeros((int(PointsPerDist),2))
DistNumber=1


#Generating the first distribution
YOneDistPoints[:,0]=StartingXCoord
for i in range(len(YOneDistPoints)):
        YOneDistPoints[i,1]=np.random.normal(StartingYCoord,DistSigma)


#Copying the points from the first dist into YOneSidePoints
for i in range(len(YOneDistPoints)):
        YOneSidePoints[i]=YOneDistPoints[i]


#Shifting the distribution along the horizontal lower edge.
for i in range(int(PointsPerDist),int(len(YOneSidePoints))):
        #If the x-coordinates are still on the square
        if YOneDistPoints[i%PointsPerDist,0]+DistNumber*StepSize<=(-1*StartingXCoord):
                #Shifting the dist a multiple of step size to the right and copying it into YOneSidePoints
                YOneSidePoints[i,1]=YOneDistPoints[i%PointsPerDist,1]
                YOneSidePoints[i,0]=YOneDistPoints[i%PointsPerDist,0]+DistNumber*StepSize
                if (i%PointsPerDist)==PointsPerDist-1:
                        DistNumber=DistNumber+1


#If there are left over points (points that are supposed to be on an edge but the distributions have already hit the end of an edge [if we incremented anymore, the distributions would be centered off of the bottom side of the square] ), put all those points on the end of the edge.
for i in range(len(YOneSidePoints)):
         #The y-coordinate would be zero iff the point was not assigned to a distribution
         if YOneSidePoints[i,1]==0:
                YOneSidePoints[i,0]=-1*StartingXCoord
                YOneSidePoints[i,1]=np.random.normal(-1*StartingYCoord,DistSigma)


#Copying the points into the EntirePointsArray
for i in range(len(YOneSidePoints)*2):
        #Copying over the flipped version
        if i >= len(YOneSidePoints):
                EntireSquarePoints[i+len(XOneSidePoints)*2]=-1*YOneSidePoints[i-(len(YOneSidePoints))]
        #Copying over the original version
        else:
                EntireSquarePoints[i+len(XOneSidePoints)*2]=YOneSidePoints[i]



############################################################################################################################################################




############################################################################################################################################################
#STEP FOUR: WRITE TO FILE



TestFile=open("FirstGeneration.txt","w")
for i in range(int(TotalNumPoints-1)):
        TestFile.write(str(EntireSquarePoints[i,0])+","+str(EntireSquarePoints[i,1])+"\n")
TestFile.close()



############################################################################################################################################################
