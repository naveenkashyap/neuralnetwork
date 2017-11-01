import numpy as np

class Generator():

	def generate(self, mu, sigma, n, size, label=None):
		hs_points = self.gen_hs_points(mu, sigma, n, size)
		c_points = self.hs_to_c(hs_points, size, n)
		l_points = self.labeled_points(c_points, size, n, label)
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

	def labeled_points(self, points, size, n, label):
		l_points = np.ones((size, n+1))
		for i in range(size):
			l_points[i] = np.append(points[i], label)
		return l_points
		
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

