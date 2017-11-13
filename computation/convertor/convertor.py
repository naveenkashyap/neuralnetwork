import tensorflow as tf

def convert(filename, size):
	with tf.device('/cpu:0'):
		input_file_queue = tf.train.string_input_producer([filename])
		reader = tf.TextLineReader()
		key, val = reader.read(input_file_queue)

	print("done")

if __name__ == "__main__":
	convert("../../csv/60_15000_5_10.csv", 30000)
