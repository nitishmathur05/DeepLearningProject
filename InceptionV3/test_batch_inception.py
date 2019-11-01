
#   Imports
import tensorflow as tf
import numpy as np
import argparse
import time
import os
# Paths to files producted as part of retraining Inception.  Change these if you saved your files in
#   a different location.
#   Retrained graph



MODEL_PATH = "/mnt/project/InceptionV3/output_graph_inception_run_oct_31_flip.pb"

LABEL_PATH = "/mnt/project/InceptionV3/output_labels_inception_run_oct_31_flip.txt"
IMAGE_ENTRY = 'DecodeJpeg/contents:0'





# LABEL_PATH = "/mnt/project/MobileNet/output_labels_mobilenet_run_oct_13_8K.txt"
# MODEL_PATH = "/mnt/project/MobileNet/output_graph_mobilenet_run_oct_13_8K.pb"
# IMAGE_ENTRY = 'input:0'
# IMAGE_ENTRY = 'DecodeJPGInput/contents:0'

# MODEL_PATH = "/Users/nitishmathur/Unimelb/Computing project/Trained_Models/output_graph_inception_run_sep_13.pb"

# LABEL_PATH = "/Users/nitishmathur/Unimelb/Computing project/Trained_Models/output_labels_inception_run_sep_13.txt"

def filter_delimiters(text):
  filtered = text[:-3]
  filtered = filtered.strip("b'")
  filtered = filtered.strip("'")
  return filtered


def predict_image_class(imagePath, labelPath):
  
  matches = None # Default return to none

  

  # Load the image from file
  image_data = tf.gfile.FastGFile(imagePath, 'rb').read()

  # Load the retrained inception based graph
  with tf.gfile.FastGFile(MODEL_PATH, 'rb') as f:
      # init GraphDef object
      graph_def = tf.GraphDef()
      # Read in the graphy from the file
      graph_def.ParseFromString(f.read())
      _ = tf.import_graph_def(graph_def, name='')
    # this point the retrained graph is the default graph

  with tf.Session() as sess:
    # These 2 lines are the code that does the classification of the images 
    # using the new classes we retrained Inception to recognize. 
    #   We find the final result tensor by name in the retrained model
    if not tf.gfile.Exists(imagePath):
      tf.logging.fatal('File does not exist %s', imagePath)
      return matches

    softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
    #   Get the predictions on our image by add the image data to the tensor
    predictions = sess.run(softmax_tensor,{IMAGE_ENTRY: image_data})
    
    # Format predicted classes for display
    #   use np.squeeze to convert the tensor to a 1-d vector of probability values
    predictions = np.squeeze(predictions)

    top_k = predictions.argsort()[-5:][::-1]  # Getting the indicies of the top 5 predictions

    #   read the class labels in from the label file
    f = open(labelPath, 'rb')
    lines = f.readlines()
    labels = [str(w).replace("\n", "") for w in lines]
    print("")
    print ("Image Classification Probabilities")
    #   Output the class probabilites in descending order
    for node_id in top_k:
      human_string = self.filter_delimiters(labels[node_id])
      score = predictions[node_id]
      if "non porn" == str(human_string):
        human_string = "non_porn"

      break;

  sess.close()
  return human_string


if __name__ == '__main__':
  
  # Ensure the user passes the image_path

  test_image_path = '/mnt/project/NPDI/test_images'
  test_result_file = '/mnt/project/NPDI/test_images_result/test_results_inception/results_oct_31.txt'

  tot_imgs = 0
  correct_guess = 0

  test_folders = os.listdir(test_image_path)
  with open(test_result_file,'w') as fd:
    fd.write("-"*10+"Inception V3"+"-"*10)
    fd.write("\n")

    for correct_label in test_folders: 

      pen = "-"*10 + " Label: " + correct_label + "-"*10
      print (pen)
      fd.write(pen + "\n")

      pen = "Image \t Predicted Label \t Correct Label \t Guess"
      print(pen)
      fd.write(pen + "\n")

      folder_path = test_image_path + '/' + correct_label
      correct_guess = 0
      tot_folder_imgs = len(os.listdir(folder_path))
      for image in os.listdir(folder_path):
        tot_imgs += 1
        guess = "incorrect"
        predicted_label = "Not confident enough"
        try:
          file_name = folder_path + '/' + image
          predicted_label=predict_image_class(file_name, LABEL_PATH)

          if predicted_label == correct_label:
            correct_guess += 1
            guess = "correct"

          pen = image + "\t" +  predicted_label + "\t" + correct_label + "\t" + guess
          print (pen)
          fd.write(pen + "\n")
        except:
          pass
      pen = "-"*10 + " Final Results " + "-"*10
      print(pen)
      fd.write(pen + "\n")

      pen = "Total " + correct_label + " : " + str(tot_folder_imgs)
      fd.write(pen + "\n")
      print (pen)

      pen = "Correct Predicted " + correct_label + " : " + str(correct_guess) 
      fd.write(pen + "\n")
      print (pen)
  
