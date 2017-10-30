import generator

INNER_SPHERE_LABEL = 0
OUTER_SPHERE_LABEL = 1
INNER_SPHERE_FILE = "inner_sphere_points.csv"
OUTER_SPHERE_FILE = "outer_sphere_points.csv"

def main():
	inner_sphere = generator.Generator(0, 10, 60, 15000, INNER_SPHERE_LABEL, INNER_SPHERE_FILE)  # mu=0, sigma=10, dimensions=60, points=15000
	outer_sphere = generator.Generator(5, 10, 60, 15000, OUTER_SPHERE_LABEL, OUTER_SPHERE_FILE)  # mu=5, sigma=10, dimensions=60, points=15000

	print("Generating inner_sphere points and writing them to " + INNER_SPHERE_FILE)
	inner_sphere.generate()
	print("Generating outer_sphere points and writing them to " + OUTER_SPHERE_FILE)
	outer_sphere.generate()

if __name__ == "__main__":
	main()
