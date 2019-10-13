import os
import tensorflow as tf
import numpy as np
import argparse
import time

# Variables --------------------------------------------------------------------------------------------
# This section is for mobilenet
# MODEL_NAME = 'mobilenet_1.0_224'
# IMAGE_ENTRY = 'DecodeJpgInput:0'

MODEL_NAME = "inception_v3"
IMAGE_ENTRY = 'DecodeJpeg:0'

LABEL_PATH = "/mnt/project/MobileNet/output_labels_mobilenet_run_sep_27_8K.txt"
MODEL_PATH = "/mnt/project/MobileNet/output_graph_mobilenet_run_sep_27_8K.pb"

TEST_IMAGES_DIR = "/mnt/project/NPDI/test_images"
NON_PORN_DIR = TEST_IMAGES_DIR + '/non_porn'
PORN_DIR = TEST_IMAGES_DIR + '/porn'

SCALAR_RED = (0.0, 0.0, 255.0)
SCALAR_BLUE = (255.0, 0.0, 0.0)


def filter_delimiters(text):
    filtered = text[:-3]
    filtered = filtered.strip("b'")
    filtered = filtered.strip("'")
    return filtered


def predict_image_class(imagePath, labelPath):
    matches = None # Default return to none

    # Load the image from file
    image_data = tf.gfile.FastGFile(imagePath, 'rb').read()

    # Load the retrained mobilenet based graph
    with tf.gfile.FastGFile(MODEL_PATH, 'rb') as f:
            # init GraphDef object
            graph_def = tf.GraphDef()
            # Read in the graphy from the file
            graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(graph_def, name='')
        # this point the retrained graph is the default graph

    with tf.Session() as sess:
        # These 2 lines are the code that does the classification of the images 
        # using the new classes we retrained mobilenet to recognize. 
        #   We find the final result tensor by name in the retrained model
        start = time.time()
        if not tf.gfile.Exists(imagePath):
            tf.logging.fatal('File does not exist %s', imagePath)
            return matches

        softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
        predictions = sess.run(softmax_tensor,
                            {'DecodeJpeg/contents:0': image_data})

        predictions = np.squeeze(predictions)
        top_k = predictions.argsort()[-5:][::-1]
        f = open(labelPath, 'rb')
        lines = f.readlines()
        labels = [str(w).replace("\n", "") for w in lines]
        print("")
        print ("Image Classification Probabilities")
        #   Output the class probabilites in descending order
        for node_id in top_k:
            human_string = filter_delimiters(labels[node_id])
            score = predictions[node_id]
            print('{0:s} (score = {1:.5f})'.format(human_string, score))

        print("")

        answer = labels[top_k[0]]

        print ("time to classify-",(time.time()-start))
        return answer

# ---------------------------------------------------------------------------------------------------------

# def main():

#     # Read classification from label file
#     classifications = []
#     for currentLine in tf.gfile.GFile(RETRAINED_LABELS_TXT_FILE_LOC):
#         classification = currentLine.rstrip()
#         classifications.append(classification)

#     # load the graph from file
#     with tf.gfile.FastGFile(RETRAINED_GRAPH_PB_FILE_LOC, 'rb') as retrainedGraphFile:
#         # instantiate a GraphDef object
#         graphDef = tf.GraphDef()
#         # read in retrained graph into the GraphDef object
#         graphDef.ParseFromString(retrainedGraphFile.read())
#         # import the graph into the current default Graph, note that we don't need to be concerned with the return value
#         _ = tf.import_graph_def(graphDef, name='')

#     # if the test image directory listed above is not valid, show an error message and bail
#     if not os.path.isdir(NON_PORN_DIR):
#         print("Directory ", NON_PORN_DIR, ' doesnt exist')
#         return
#     if not os.path.isdir(PORN_DIR):
#         print("Directory ", PORN_DIR, ' doesnt exist')
#         return
#     # end if

#     with tf.Session() as sess:

#         # for each folder in the test images directory . . .
#         for group in next(os.walk(TEST_IMAGES_DIR))[1]:
#             # Get classification group folder dir
#             groupDir = os.path.join(TEST_IMAGES_DIR, group)
#             print (groupDir)

#             # for each sub-folder in each classification folder
#             for fileName in os.listdir(groupDir):

#                 resultLog = open(groupDir + '_' + MODEL_NAME + '.txt', 'w')

#                 # if the file does not end in .jpg or .jpeg (case-insensitive), continue with the next iteration of the for loop
#                 if not (fileName.lower().endswith(".jpg") or fileName.lower().endswith(".jpeg")):
#                     continue
#                 # show the file name on std out
#                 print(fileName)

#                 imageFileWithPath = os.path.join(groupDir, fileName)
#                 # attempt to open the image with OpenCV
#                 openCVImage = cv2.imread(imageFileWithPath)

#                 # get the final tensor from the graph
#                 finalTensor = sess.graph.get_tensor_by_name('final_result:0')

#                 # convert the OpenCV image (numpy array) to a TensorFlow image
#                 # Some image may corrupt from google search
#                 try:
#                     tfImage = np.array(openCVImage)[:, :, 0:3]
#                 except IndexError:
#                     continue

#                 # run the network to get the predictions
#                 print (finalTensor)
#                 print (tfImage)
#                 predictions = sess.run(finalTensor, {IMAGE_ENTRY : tfImage})

#                 # sort predictions from most confidence to least confidence
#                 sortedPredictions = predictions[0].argsort()[-len(predictions[0]):][::-1]

#                 print("---------------------------------------")

#                 onMostLikelyPrediction = True
#                 # for each prediction . . .
#                 for prediction in sortedPredictions:
#                     strClassification = classifications[prediction]

#                     # if the classification (obtained from the directory name) ends with the letter "s", remove the "s" to change from plural to singular
#                     if strClassification.endswith("s"):
#                         strClassification = strClassification[:-1]
#                     # end if

#                     # get confidence, then get confidence rounded to 2 places after the decimal
#                     confidence = predictions[0][prediction]

#                     # if we're on the first (most likely) prediction, state what the object appears to be and show a % confidence to two decimal places
#                     if onMostLikelyPrediction:

#                         # Write result log
#                         resultLog.write(fileName + '\t' + strClassification + '\t' + str(confidence) + '\n')

#                         onMostLikelyPrediction = False

#                 resultLog.close()



#     # write the graph to file so we can view with TensorBoard
#     tfFileWriter = tf.compat.v1.summary.FileWriter(TEST_IMAGES_DIR)
#     tfFileWriter.add_graph(sess.graph)
#     tfFileWriter.close()


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
