import numpy as np
import math
import random


class generator3():
	


	def generate_all_points(self,VolMu,VolSigma,circle_sigma,NumPoints):
		

		#Calculating circle_mu to ensure the circle covers the same area
		#circle_mu=math.sqrt(VolMu/math.pi)
		#SPECIAL CASE OF 5/12/18
		circle_mu=5

		#2-Dimensions for now
		n=2
	
		#Generating all points
		CirclePoints=self.generate_circle(circle_mu,circle_sigma,n,NumPoints)
		SquarePoints=self.generate_square_points(VolMu,VolSigma,NumPoints)

		AllLabeledData=np.append(CirclePoints,SquarePoints,axis=0)

		return AllLabeledData		



	#Square generation first	
        def generate_square_points(self,VolMu,VolSigma,NumPoints):


		SquarePoints=np.zeros((NumPoints,2))
		
		for j in range(0,NumPoints):
			Vol=np.random.normal(VolMu, VolSigma)
			SideLength=math.sqrt(Vol)
			FixedPoint=SideLength/2
		
			RandomPoint=(random.randint(0,math.pow(10,6))*1/(math.pow(10,6)))*SideLength-SideLength/2
			
			WhichSide=random.randint(1,4)
			if (WhichSide==1):
				SquarePoints[j,0]=-FixedPoint
				SquarePoints[j,1]=RandomPoint
			if (WhichSide==2):
				SquarePoints[j,0]=RandomPoint
				SquarePoints[j,1]=FixedPoint
			if (WhichSide==3):
				SquarePoints[j,0]=FixedPoint
				SquarePoints[j,1]=RandomPoint
			if (WhichSide==4):
				SquarePoints[j,0]=RandomPoint
				SquarePoints[j,1]=-FixedPoint

		Labeled_Square_Points=self.label_square_points(SquarePoints,NumPoints)

                return Labeled_Square_Points




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
		print size
		print n
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
 

