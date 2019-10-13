
"""
 AUTHOR : Kage Zhuang & Lei Ren
 PURPOSE :  Testing re-training model with all the testing image in the folder.
			The program go through each folder inside the test_images folder and
			its sub-folder.
			The original file is from:
			https://github.com/MicrocontrollersAndMore/TensorFlow_Tut_2_Classification_Walk-through/blob/master/test.py
"""

import os
import tensorflow as tf
import numpy as np
import cv2

# Variables --------------------------------------------------------------------------------------------
# This section is for inception v3
# MODEL_NAME = "inception_v3"
# IMAGE_ENTRY = 'DecodeJpeg:0'

# This section is for mobilenet
MODEL_NAME = 'mobilenet_1.0_224'
IMAGE_ENTRY = 'DecodeJpgInput:0'

RETRAINED_LABELS_TXT_FILE_LOC = "/mnt/project/MobileNet/output_labels_mobilenet_run_sep_27_8K.txt"
RETRAINED_GRAPH_PB_FILE_LOC = "/mnt/project/MobileNet/output_graph_mobilenet_run_sep_27_8K.pb"

TEST_IMAGES_DIR = "/mnt/project/NPDI/test_images"
NON_PORN_DIR = TEST_IMAGES_DIR + '/non_porn'
PORN_DIR = TEST_IMAGES_DIR + '/porn'

SCALAR_RED = (0.0, 0.0, 255.0)
SCALAR_BLUE = (255.0, 0.0, 0.0)
# ---------------------------------------------------------------------------------------------------------

def main():
	print ("-"*10,"1","-"*10)
	# Read classification from label file
	classifications = []
	for currentLine in tf.gfile.GFile(RETRAINED_LABELS_TXT_FILE_LOC):
		classification = currentLine.rstrip()
		classifications.append(classification)

	print ("-"*10,"2","-"*10)
	# load the graph from file
	with tf.gfile.FastGFile(RETRAINED_GRAPH_PB_FILE_LOC, 'rb') as retrainedGraphFile:
		# instantiate a GraphDef object
		graphDef = tf.GraphDef()
		# read in retrained graph into the GraphDef object
		graphDef.ParseFromString(retrainedGraphFile.read())
		# import the graph into the current default Graph, note that we don't need to be concerned with the return value
		_ = tf.import_graph_def(graphDef, name='')

	print ("-"*10,"3","-"*10)
	# if the test image directory listed above is not valid, show an error message and bail
	if not os.path.isdir(NON_PORN_DIR):
		print("Directory ", NON_PORN_DIR, ' doesnt exist')
		return
	if not os.path.isdir(PORN_DIR):
		print("Directory ", PORN_DIR, ' doesnt exist')
		return
	# end if
	print ("-"*10,"4","-"*10)
	with tf.Session() as sess:
		print ("-"*10,"5","-"*10)
		# for each folder in the test images directory . . .
		for group in next(os.walk(TEST_IMAGES_DIR))[1]:
			# Get classification group folder dir
			print ("-"*10,group,"-"*10)
			groupDir = os.path.join(TEST_IMAGES_DIR, group)

			# for each sub-folder in each classification folder
			# for subgroup in next(os.walk(groupDir))[1]:
			# 	print ("-"*10,subgroup,"-"*10)
			# 	# Get subgroup folder dir
			# 	subgroupDir = os.path.join(groupDir, subgroup)
			# 	print ("-"*10,subgroupDir,"-"*10)
				# Create result log for this subgroup of test images
			resultLog = open(groupDir + '_' + MODEL_NAME + '.txt', 'w')

				# For each image file inside the sub-group folder
			for fileName in os.listdir(groupDir):
				print ("-"*10,fileName,"-"*10)
				# if the file does not end in .jpg or .jpeg (case-insensitive), continue with the next iteration of the for loop
				if not (fileName.lower().endswith(".jpg") or fileName.lower().endswith(".jpeg")):
					continue
				# show the file name on std out
				print(fileName)

				imageFileWithPath = os.path.join(groupDir, fileName)
				# attempt to open the image with OpenCV
				openCVImage = cv2.imread(imageFileWithPath)

				# get the final tensor from the graph
				finalTensor = sess.graph.get_tensor_by_name('final_result:0')

				# convert the OpenCV image (numpy array) to a TensorFlow image
				# Some image may corrupt from google search
				try:
					tfImage = np.array(openCVImage)[:, :, 0:3]
				except IndexError:
					continue

				# run the network to get the predictions
				predictions = sess.run(finalTensor, {IMAGE_ENTRY : tfImage})

				# sort predictions from most confidence to least confidence
				sortedPredictions = predictions[0].argsort()[-len(predictions[0]):][::-1]

				print("---------------------------------------")

				onMostLikelyPrediction = True
				# for each prediction . . .
				for prediction in sortedPredictions:
					strClassification = classifications[prediction]

					# if the classification (obtained from the directory name) ends with the letter "s", remove the "s" to change from plural to singular
					if strClassification.endswith("s"):
						strClassification = strClassification[:-1]
					# end if

					# get confidence, then get confidence rounded to 2 places after the decimal
					confidence = predictions[0][prediction]

					# if we're on the first (most likely) prediction, state what the object appears to be and show a % confidence to two decimal places
					if onMostLikelyPrediction:

						# Write result log
						resultLog.write(fileName + '\t' + strClassification + '\t' + str(confidence) + '\n')

						onMostLikelyPrediction = False

			resultLog.close()



	# write the graph to file so we can view with TensorBoard
	tfFileWriter = tf.summary.FileWriter(TEST_IMAGES_DIR)
	tfFileWriter.add_graph(sess.graph)
	tfFileWriter.close()


if __name__ == "__main__":
	main()
