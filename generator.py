import numpy as np

class Generator():

	def __init__(self, mu, sigma, n, size, label=None, filename=None):
		self.mu = mu
		self.sigma = sigma
		self.dimensions = n
		self.size = size
		self.label = label
		self.filename = filename
	
	def generate(self):
		hs_points = self.gen_hs_points()
		c_points = self.hs_to_c(hs_points)
		write_points(c_points, filename=self.filename, label=self.label)
	
	def gen_hs_points(self):
		hs_points = np.zeros((self.size, self.dimensions)) 
		for i in range(self.size):
			# generate a value for each dimension in hyperspherical coordinates
			hs_point = hs_points[i]
			for j in range(self.dimensions):
				if j == 0:
					# normal distribution on radius
					coord = np.random.normal(self.mu, self.sigma)
				elif j < self.dimensions-1:
					# uniform distribution [0,pi] on phi
					coord = np.random.rand()*np.pi
				else:
					# uniform distribution [0,2pi] on theta
					coord = np.random.rand()*(2*np.pi)
				hs_point[j] = coord
			hs_points[i] = hs_point
		return hs_points
	
	def hs_to_c(self, hs_points):
		c_points = np.zeros((self.size, self.dimensions))
		# translate each hyperspherical point into a cartesian point
		for i in range(self.size):
	
			hs_point = hs_points[i]
			c_point = c_points[i]
	
			sin = np.ones((self.dimensions))
			sin[0] = hs_point[0]
	
			for j in range(self.dimensions):
				if j == 0:
					coord = sin[j] * np.cos(hs_point[j+1])
				elif j < self.dimensions-1:
					sin[j] = sin[j-1] * np.sin(hs_point[j])
					coord = sin[j] *  np.cos(hs_point[j+1])
				else:
					coord = sin[j-1] * np.sin(hs_point[j])
				c_point[j] = coord
			c_points[i] = c_point
		return c_points
	
		
def write_points(points, filename=None, label=None):
	if filename is None:
		filename = "points.csv"
	f = open(filename, 'w') # TODO Error handle
	line = ""
	for point in points:
		for coord in point:
			line += str(coord) + ", "
		if label is not None:
			line += str(label)
		else:
			line = line.rstrip(', ')
		line += '\n'
	f.write(line)

