import numpy as np
import math



class Generator():
	


	def generate_all_points(self,PointsPerShape,LengthOfSide,SquareDistSigma,DistPerSide,circle_sigma):
		
		#Assuming total number of points in each distribution
		size=PointsPerShape

		#Calculating Area and circle_mu to ensure the circle covers the same area
		Area=LengthOfSide*LengthOfSide
		circle_mu=math.sqrt(Area/math.pi)
		
		circle_area=math.pi*math.pow(circle_mu,2)

		#2-Dimensions for now
		n=2
	
		#Generating all points
		CirclePoints=self.generate_circle(circle_mu,circle_sigma,n,size)
		SquarePoints=self.generate_square_points(PointsPerShape,LengthOfSide,SquareDistSigma,DistPerSide)

		AllLabeledData=np.append(CirclePoints,SquarePoints,axis=0)

		return AllLabeledData		



	#Square generation first
		
	#Note: TotalNumPoints, LengthOfSide, and DistSigma should all be floats. DistPerSide is an int.
        def generate_square_points(self,TotalNumPoints,LengthOfSide,SquareDistSigma,DistPerSide):

                NumberOfSides=4

                XDistPoints = self.generate_vertical_edges(float(TotalNumPoints),float(LengthOfSide),float(SquareDistSigma),int(DistPerSide),NumberOfSides)
                YDistPoints = self.generate_horizontal_edges(float(TotalNumPoints),float(LengthOfSide),float(SquareDistSigma),int(DistPerSide),NumberOfSides)

                EntireSquarePoints=np.append(XDistPoints,YDistPoints,axis=0)

                #Error correction
                #Fixing any unassigned points (in case the all the points don't fit cleanly into the desired number of distributions) by assigning the left over positions to the bottom left hand corner.
                for j in range(len(EntireSquarePoints)):
                        if EntireSquarePoints[j,0]==0 and EntireSquarePoints[j,1]==0:
                                EntireSquarePoints[j,0]=-1*(float(LengthOfSide)/2.0)
                                EntireSquarePoints[j,1]=np.random.normal(-1*(float(LengthOfSide)/2.0),SquareDistSigma)

                LabeledPoints=self.label_square_points(EntireSquarePoints,float(TotalNumPoints))

		#Shuffling the points
		#np.random.shuffle(LabeledPoints)

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



	def label_square_points(self,SquarePoints,TotalNumPoints):

                #Setting up storage
                LabeledSquarePoints=np.zeros((int(TotalNumPoints),4))

                #Labeling Points
                for i in range(int(TotalNumPoints)):
                        LabeledSquarePoints[i,0]=SquarePoints[i,0]
                        LabeledSquarePoints[i,1]=SquarePoints[i,1]
                        LabeledSquarePoints[i,2]=1
                        LabeledSquarePoints[i,3]=0

                return LabeledSquarePoints



	#Now circle generation
	def generate_circle(self, mu, sigma, n, size):
		hs_points = self.gen_hs_points(mu, sigma, n, size)
		c_points = self.hs_to_c(hs_points, size, n)
		l_points = self.labeled_points_circle(c_points, size, n)
		np.random.shuffle(l_points)
		return l_points



	def gen_hs_points(self, mu, sigma, n, size):
		hs_points = np.zeros((size, n)) 
		for i in range(size):
			# generate a value for each dimension in hyperspherical coordinates
			hs_point = hs_points[i]
			for j in range(n):
				if j == 0:
					# normal distribution on radius
					coord = np.random.normal(mu, sigma)
				elif j < n-1:
					# uniform distribution [0,pi] on phi
					coord = np.random.rand()*np.pi
				else:
					# uniform distribution [0,2pi] on theta
					coord = np.random.rand()*(2*np.pi)
				hs_point[j] = coord
			hs_points[i] = hs_point
		return hs_points



	def hs_to_c(self, hs_points, size, n):
		c_points = np.zeros((size, n))
		# translate each hyperspherical point into a cartesian point
		for i in range(size):
	
			hs_point = hs_points[i]
			c_point = c_points[i]
	
			sin = np.ones((n))
			sin[0] = hs_point[0]
	
			for j in range(n):
				if j == 0:
					coord = sin[j] * np.cos(hs_point[j+1])
				elif j < n-1:
					sin[j] = sin[j-1] * np.sin(hs_point[j])
					coord = sin[j] *  np.cos(hs_point[j+1])
				else:
					coord = sin[j-1] * np.sin(hs_point[j])
				c_point[j] = coord
			c_points[i] = c_point
		return c_points



	#Made label [0,1] for all circular points 
	def labeled_points_circle(self, points, size, n):
		label = [0,1]
		l_points = np.ones((size, n+2))
		for i in range(size):
			l_points[i] = np.append(points[i], label)
		return l_points
 

