import numpy as np
import math
import random



#There will be 18 total boxes we're sampling in. One at the midpoint of each edge (12 edges total) and one in the middle of each face. For a box's side length I'm using the volume sigma of the distributions.

class generator_3D():
	

	def generate_all_points(self,volume_mu,volume_sigma,NumPoints):

                #3D
                n=3

                #Generating all points

                CirclePoints=self.generate_circle(volume_mu,volume_sigma,n,NumPoints)
	
                SquarePoints=self.generate_square_points(volume_mu,volume_sigma,NumPoints)
	
                AllLabeledData=np.append(CirclePoints,SquarePoints,axis=0)

                return AllLabeledData


	def TestGen(self, AllLabeledData,volume_mu,test_box_SL,NumPoints):
		#First index in box arrays will be the number of sphere points, second index will be the number of cube points.
		
		#All boxes if statements checked
		#Center of face boxes (SL=Side Length):
		#Box 1: +SL/2, 0, 0
		#Box 2: -SL/2, 0, 0
		#Box 3: 0, +SL/2, 0
		#Box 4: 0, -SL/2, 0
		#Box 5: 0, 0, +SL/2
		#Box 6: 0, 0, -SL/2
		
		#Center of edge boxes:
		
		#Fixed Positive x face first:
		#Box 7: (SL/2,-SL/2,0)
		#Box 8: (SL/2,SL/2,0)
		#Box 9: (SL/2,0,SL/2)
		#Box 10: (SL/2,0,-SL/2)
		

		#Fixed Negative x face:
		#Box 11: (-SL/2,-SL/2,0)
		#Box 12: (-SL/2,SL/2,0)
		#Box 13: (-SL/2,0,SL/2)
		#Box 14: (-SL/2,0,-SL/2)


		#Fixed Positive y face:
		#Box 15: (0,+SL/2,-SL/2)
		#Box 16: (0,+SL/2,SL/2) 

		#Fixed Negative y face:
		#Box 17: (0,-SL/2,SL/2)
		#Box 18: (0,-SL/2,-SL/2)

		#Setting up the arrays. The first index will be the number of sphere points, the second index will be the number of cube points.
		Box1=np.zeros(2)
		Box2=np.zeros(2)
		Box3=np.zeros(2)
		Box4=np.zeros(2)
		Box5=np.zeros(2)
		Box6=np.zeros(2)
		Box7=np.zeros(2)
		Box8=np.zeros(2)
		Box9=np.zeros(2)
		Box10=np.zeros(2)
		Box11=np.zeros(2)
		Box12=np.zeros(2)
		Box13=np.zeros(2)
		Box14=np.zeros(2)
		Box15=np.zeros(2)
		Box16=np.zeros(2)
		Box17=np.zeros(2)
		Box18=np.zeros(2)
		

		SL= volume_mu ** (1./3)
		#Filling up the boxes.

		#Categorizing the points
		for i in range(0,NumPoints*2): 
			x=AllLabeledData[i,0]
			y=AllLabeledData[i,1]
			z=AllLabeledData[i,2]
		
			#If this is a 0 the point is in the sphere group, if this is a 1 the point is in the cube group.
			CubeOrSphere=AllLabeledData[i,3]
			#print CubeOrSphere

			#Box 1
			if SL/2.-test_box_SL/2.<x<SL/2.+test_box_SL/2. and -test_box_SL/2.<y<test_box_SL/2. and -test_box_SL/2.<z<test_box_SL/2.:
				if CubeOrSphere==0:
					Box1[0]=Box1[0]+1
				else:
					Box1[1]=Box1[1]+1		
			
			#Box 2
			if -SL/2.-test_box_SL/2.<x<-SL/2.+test_box_SL/2. and -test_box_SL/2.<y<test_box_SL/2. and -test_box_SL/2.<z<test_box_SL/2.:
                                if CubeOrSphere==0:
                                        Box2[0]=Box2[0]+1
                                else:   
                                        Box2[1]=Box2[1]+1   

			#Box 3
			if -test_box_SL/2.<x<test_box_SL/2. and SL/2.-test_box_SL/2.<y<SL/2.+test_box_SL/2. and -test_box_SL/2.<z<test_box_SL/2.:
                                if CubeOrSphere==0:
                                        Box3[0]=Box3[0]+1
                                else:
                                        Box3[1]=Box3[1]+1  

			#Box 4
			if -test_box_SL/2.<x<test_box_SL/2. and -SL/2.-test_box_SL/2.<y<-SL/2.+test_box_SL/2. and -test_box_SL/2.<z<test_box_SL/2.:
                                if CubeOrSphere==0:
                                        Box4[0]=Box4[0]+1
                                else:
                                        Box4[1]=Box4[1]+1 

			#Box 5
			if -test_box_SL/2.<x<test_box_SL/2. and -test_box_SL/2.<y<test_box_SL/2. and SL/2.-test_box_SL/2.<z<SL/2.+test_box_SL/2.:
                                if CubeOrSphere==0:
                                        Box5[0]=Box5[0]+1
                                else:
                                        Box5[1]=Box5[1]+1 

			#Box 6
			if -test_box_SL/2.<x<test_box_SL/2. and -test_box_SL/2.<y<test_box_SL/2. and -SL/2.-test_box_SL/2.<z<-SL/2.+test_box_SL/2.:
                                if CubeOrSphere==0:
                                        Box6[0]=Box6[0]+1
                                else:
                                        Box6[1]=Box6[1]+1

			#Boxes 1-6 are good, have been checked
			#Box 7
			if SL/2.-test_box_SL/2.<x<SL/2.+test_box_SL/2. and -SL/2.-test_box_SL/2.<y<-SL/2.+test_box_SL/2. and -test_box_SL/2.<z<test_box_SL/2.:
				if CubeOrSphere==0:
                                        Box7[0]=Box7[0]+1
                                else:
                                        Box7[1]=Box7[1]+1
			
			#Box 8
			if SL/2.-test_box_SL/2.<x<SL/2.+test_box_SL/2. and SL/2.-test_box_SL/2.<y<SL/2.+test_box_SL/2. and -test_box_SL/2.<z<test_box_SL/2.:
                                if CubeOrSphere==0:
                                        Box8[0]=Box8[0]+1
                                else:
                                        Box8[1]=Box8[1]+1

			#Box 9
			if SL/2.-test_box_SL/2.<x<SL/2.+test_box_SL/2. and -test_box_SL/2.<y<test_box_SL/2. and SL/2.-test_box_SL/2.<z<SL/2.+test_box_SL/2.:
                                if CubeOrSphere==0:
                                        Box9[0]=Box9[0]+1
                                else:
                                        Box9[1]=Box9[1]+1
			
			#Box 10
			if SL/2.-test_box_SL/2.<x<SL/2.+test_box_SL/2. and -test_box_SL/2.<y<test_box_SL/2. and -SL/2.-test_box_SL/2.<z<-SL/2.+test_box_SL/2.:
                                if CubeOrSphere==0:
                                        Box10[0]=Box10[0]+1
                                else:
                                        Box10[1]=Box10[1]+1
			#Boxes up to here (10) have been checked
			#Box 11
			if -SL/2.-test_box_SL/2.<x<-SL/2.+test_box_SL/2. and -SL/2.-test_box_SL/2.<y<-SL/2.+test_box_SL/2. and -test_box_SL/2.<z<test_box_SL/2.:
                                if CubeOrSphere==0:
                                        Box11[0]=Box11[0]+1
                                else:
                                        Box11[1]=Box11[1]+1
			#Up to Box 12 have been checked
			#Box 12
			if -SL/2.-test_box_SL/2.<x<-SL/2.+test_box_SL/2. and SL/2.-test_box_SL/2.<y<SL/2.+test_box_SL/2. and -test_box_SL/2.<z<test_box_SL/2.:
                                if CubeOrSphere==0:
                                        Box12[0]=Box12[0]+1
                                else:
                                        Box12[1]=Box12[1]+1

			#Box 13
			if -SL/2.-test_box_SL/2.<x<-SL/2.+test_box_SL/2. and -test_box_SL/2.<y<test_box_SL/2. and SL/2.-test_box_SL/2.<z<SL/2.+test_box_SL/2.:
                                if CubeOrSphere==0:
                                        Box13[0]=Box13[0]+1
                                else:
                                        Box13[1]=Box13[1]+1
			
			#Box 14
			if -SL/2.-test_box_SL/2.<x<-SL/2.+test_box_SL/2. and -test_box_SL/2.<y<test_box_SL/2. and -SL/2.-test_box_SL/2.<z<-SL/2.+test_box_SL/2.:
                                if CubeOrSphere==0:
                                        Box14[0]=Box14[0]+1
                                else:
                                        Box14[1]=Box14[1]+1	

			#Box 15
			if -test_box_SL/2.<x<test_box_SL/2. and SL/2.-test_box_SL/2.<y<SL/2.+test_box_SL/2. and -SL/2.-test_box_SL/2.<z<-SL/2.+test_box_SL/2.:
                                if CubeOrSphere==0:
                                        Box15[0]=Box15[0]+1
                                else:
                                        Box15[1]=Box15[1]+1

			#Box 16
			if -test_box_SL/2.<x<test_box_SL/2. and SL/2.-test_box_SL/2.<y<SL/2.+test_box_SL/2. and SL/2.-test_box_SL/2.<z<SL/2.+test_box_SL/2.:
                                if CubeOrSphere==0:
                                        Box16[0]=Box16[0]+1
                                else:
                                        Box16[1]=Box16[1]+1

			#Box 17
			if -test_box_SL/2.<x<test_box_SL/2. and -SL/2.-test_box_SL/2.<y<-SL/2.+test_box_SL/2. and -SL/2.-test_box_SL/2.<z<-SL/2.+test_box_SL/2.:
                                if CubeOrSphere==0:
                                        Box17[0]=Box17[0]+1
                                else:
                                        Box17[1]=Box17[1]+1

			#Box 18
			if -test_box_SL/2.<x<test_box_SL/2. and -SL/2.-test_box_SL/2.<y<-SL/2.+test_box_SL/2. and SL/2.-test_box_SL/2.<z<SL/2.+test_box_SL/2.:
                                if CubeOrSphere==0:
                                        Box18[0]=Box18[0]+1
                                else:
                                        Box18[1]=Box18[1]+1

		return Box1,Box2,Box3,Box4,Box5,Box6,Box7,Box8,Box9,Box10,Box11,Box12,Box13,Box14,Box15,Box16,Box17,Box18
		
	def generate_square_points(self,volume_mu,volume_sigma,NumPoints):
		
                #Setting up storage
                SquarePoints=np.zeros((NumPoints,3))


                PointCounter=0
		
		#Randomly generating a volume according to a normal distribution with user input mu and sigma, compute the side length, and assign the point to a random face of the cube.
                for i in range(NumPoints):
                        Volume=np.random.normal(volume_mu,volume_sigma)
                        SideLength= Volume ** (1./3)
			#print Volume
			#print SideLength
			#print "\n"

                        #Assigning the point
                        
			WhichSide=np.random.randint(1,7)
			RandCoord1=round(random.uniform(-SideLength/2.0,SideLength/2.0),15)
			RandCoord2=round(random.uniform(-SideLength/2.0,SideLength/2.0),15)

			#Fixed Z
			if WhichSide==1:
				SquarePoints[PointCounter,0]=RandCoord1
				SquarePoints[PointCounter,1]=RandCoord2
				SquarePoints[PointCounter,2]=SideLength/2.0
			elif WhichSide==2:
				SquarePoints[PointCounter,0]=RandCoord1
				SquarePoints[PointCounter,1]=RandCoord2
				SquarePoints[PointCounter,2]=-SideLength/2.0
			#Fixed Y
			elif WhichSide==3:
				SquarePoints[PointCounter,0]=RandCoord1
				SquarePoints[PointCounter,1]=SideLength/2.0
				SquarePoints[PointCounter,2]=RandCoord2
			elif WhichSide==4:
				SquarePoints[PointCounter,0]=RandCoord1
				SquarePoints[PointCounter,1]=-SideLength/2.0
				SquarePoints[PointCounter,2]=RandCoord2
			#Fixed X
			elif WhichSide==5:
				SquarePoints[PointCounter,0]=SideLength/2.0
				SquarePoints[PointCounter,1]=RandCoord1
				SquarePoints[PointCounter,2]=RandCoord2
			elif WhichSide==6:
				SquarePoints[PointCounter,0]=-SideLength/2.0
				SquarePoints[PointCounter,1]=RandCoord1
				SquarePoints[PointCounter,2]=RandCoord2

			PointCounter=PointCounter+1
		Labeled_Square_Points=self.label_square_points(SquarePoints,NumPoints)

                return Labeled_Square_Points



	def label_square_points(self,SquarePoints,NumPoints):

                #Setting up storage
                LabeledSquarePoints=np.zeros((int(NumPoints),5))

                #Labeling Points
                for i in range(int(NumPoints)):
                        LabeledSquarePoints[i,0]=SquarePoints[i,0]
                        LabeledSquarePoints[i,1]=SquarePoints[i,1]
			LabeledSquarePoints[i,2]=SquarePoints[i,2]
                        LabeledSquarePoints[i,3]=1
                        LabeledSquarePoints[i,4]=0

                return LabeledSquarePoints



	#Now circle generation
	def generate_circle(self, mu, sigma, n, size):
		c_points = self.gen_c_points(size, n, mu, sigma)
		l_points = self.labeled_points_circle(c_points, size, n)
		np.random.shuffle(l_points)
		return l_points



	def gen_c_points(self, size, n, mu, sigma):
		nSpherePoints=np.zeros((size,n))

		for i in range(size):
        		#Randomly generate a volume for an n-dimensional sphere of radius R centered at the origin.
        		Vol=np.random.normal(mu,sigma)
        		Radius=math.pow(((Vol*math.gamma(n/2.0+1))/math.pow(np.pi,n/2.0)),1.0/n)
        		
			#Generate a 1 by n array that has a normal distribution along each of its coordinates with mu=0 and sigma=1.
        		x = np.random.normal(size=(1, n))

        		#Normalize the point, which is equivalent to making the distance from the origin for that point equal to one, thus placing it on the n-dimensional unit sphere.
        		x /= np.linalg.norm(x, axis=1)[:, np.newaxis]

        		#Multiplying the point by the previous computed radius places it on the surface of the sphere with the desired radius.
        		x=x*Radius

        		#Writing the point to a seperate array.
        		for j in range(3):
                		nSpherePoints[i,j]=x[0,j]

		return nSpherePoints



	#Made label [0,1] for all circular points 
	def labeled_points_circle(self, points, size, n):
		label = [0,1]
		l_points = np.ones((size, n+2))
		for i in range(size):
			l_points[i] = np.append(points[i], label)
		return l_points
 
