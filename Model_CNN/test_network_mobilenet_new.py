
#   Imports
import tensorflow as tf
import numpy as np
import argparse
import time

# Paths to files producted as part of retraining Inception.  Change these if you saved your files in
#   a different location.
#   Retrained graph


MODEL_PATH = "/Users/nitishmathur/Unimelb/Computing project/Trained_Models/output_graph_mobilenet_run_sep_27_8K.pb"

LABEL_PATH = "/Users/nitishmathur/Unimelb/Computing project/Trained_Models/output_labels_mobilenet_run_sep_27_8K.txt"

SCALAR_RED = (0.0, 0.0, 255.0)
SCALAR_BLUE = (255.0, 0.0, 0.0)


if __name__ == '__main__':
	
	# Ensure the user passes the image_path
	parser = argparse.ArgumentParser(description="Process arguments")
	parser.add_argument(
	  'image_path',
	  type=str,
	  default='',
	  help='Path of image to classify.'
	)
	args = parser.parse_args()

	# We can only handle jpeg images.   
	if args.image_path.lower().endswith(('.jpg', '.jpeg')):
		# predict the class of the image
		predict_image_class(args.image_path, LABEL_PATH)
	else:
		print('File must be a jpeg image.')
