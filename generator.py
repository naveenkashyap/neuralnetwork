import numpy as np

class Generator():

	def generate(self, mu, sigma, n, size, label=None):
		hs_points = self.gen_hs_points(mu, sigma, n, size)
		bins, incs = self.get_bins(hs_points, mu, sigma, size, n, label)
		c_points = self.hs_to_c(hs_points, size, n)
		l_points = self.labeled_points(c_points, size, n, label)
		return l_points, bins, incs

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
		if label == 0:
			label = [1,0]
		else:
			label = [0,1]
		l_points = np.ones((size, n+2))
		for i in range(size):
			l_points[i] = np.append(points[i], label)
		return l_points

	def get_bins(self, hs_points, mu, sigma, size, n, label):

		low = mu - (4*sigma)
		high = mu + (4*sigma)
		inc = low

		incs = [x for x in range(low, high+1)]
		bins = [[] for x in range(low, high+1)]

		for index, bin_max in enumerate(incs):
			bin_min = bin_max-1
			bucket = []
			for point in hs_points:
				radius = point[0]
				if radius <= bin_max and radius > bin_min:
					sin = np.ones((n))
					c_point = np.ones((n))
					l_point = np.ones((n+2))
					sin[0] = radius
	
					for j in range(n):
						if j == 0:
							coord = sin[j] * np.cos(point[j+1])
						elif j < n-1:
							sin[j] = sin[j-1] * np.sin(point[j])
							coord = sin[j] *  np.cos(point[j+1])
						else:
							coord = sin[j-1] * np.sin(point[j])
						c_point[j] = coord
					
					if label == 0:
						l_point = np.append(c_point, [1,0])
					else:
						l_point = np.append(c_point, [0,1])

					bucket.append(l_point)
			bins[index] = bucket
		
		return bins, incs

