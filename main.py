import sys
import numpy as np
import generator
from computation.trainer import Trainer_np_Array as trainer
from computation.trainer import Tester_np_Array as tester

CSV_LOC = "csv/"
INNER_SPHERE_LABEL = 0
OUTER_SPHERE_LABEL = 1

def main():
	n = sys.argv[1]
	size = sys.argv[2]
	mu = sys.argv[3]
	sigma = sys.argv[4]

	print("Using:\n\tn = " + n + "\n\tsize = " + size + "\n\tmu = " + mu + "\n\tsigma = " + sigma + "\n")
	filename = "_".join(sys.argv[1:5]) + ".csv"

	g = generator.Generator()

	print("Generating inner_sphere")
	inner_sphere = g.generate(int(mu), int(sigma), int(n), int(size), INNER_SPHERE_LABEL)
	print("Generating outer_sphere")
	outer_sphere = g.generate(int(mu), int(sigma), int(n), int(size), OUTER_SPHERE_LABEL)
	
	print("Combining and shuffling generated points")
	toy = np.append(inner_sphere, outer_sphere, axis=0)
	np.random.shuffle(toy)

	tr = trainer.Trainer_np_Array()
	te = tester.Tester_np_Array()

	print("Writing points to " + CSV_LOC + filename)
	f = open(CSV_LOC + filename, 'w')
	for point in toy:
		line = ""
		for coord in point:
			line += str(coord) + ", "
		f.write(line.rstrip(", ") + '\n')
	f.close()

if __name__ == "__main__":
	main()
