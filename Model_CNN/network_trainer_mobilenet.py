
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
from datetime import datetime
import hashlib
import os.path
import random
import re
import struct
import sys
import tarfile

import numpy as np
from six.moves import urllib
import tensorflow as tf

from tensorflow.python.framework import graph_util
from tensorflow.python.framework import tensor_shape
from tensorflow.python.platform import gfile
from tensorflow.python.util import compat

FLAGS = None

# These are all parameters that are tied to the particular model architecture
# we're using for Inception v3. These include things like tensor names and their
# sizes. If you want to adapt this script to work with another model, you will
# need to update these to reflect the values in the network you're using.
# pylint: disable=line-too-long
# pylint: enable=line-too-long
MAX_NUM_IMAGES_PER_CLASS = 2 ** 27 - 1  # ~134M

# DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'  
# BOTTLENECK_TENSOR_NAME = 'pool_3/_reshape:0' 
# BOTTLENECK_TENSOR_SIZE = 2048
# MODEL_INPUT_WIDTH = 299
# MODEL_INPUT_HEIGHT = 299
# MODEL_INPUT_DEPTH = 3
# JPEG_DATA_TENSOR_NAME = 'DecodeJpeg/contents:0' 
# RESIZED_INPUT_TENSOR_NAME = 'ResizeBilinear:0' 

# Mobilenet Parameters NOT quantized -  mobilenet_1.0_224
DATA_URL = 'http://download.tensorflow.org/models/mobilenet_v1_1.0_224_frozen.tgz'
BOTTLENECK_TENSOR_NAME = 'MobilenetV1/Predictions/Reshape:0'
RESIZED_INPUT_TENSOR_NAME = 'input:0'
BOTTLENECK_TENSOR_SIZE = 1001
MODEL_INPUT_WIDTH = 224
MODEL_INPUT_HEIGHT = 224
MODEL_INPUT_DEPTH = 3
JPEG_DATA_TENSOR_NAME = 'DecodeJpeg/contents:0'

# Mobilenet Parameters NOT quantized -  mobilenet_v2_1.4_224
# DATA_URL = 'http://download.tensorflow.org/models/mobilenet_v2_1.4_224.tgz'
# BOTTLENECK_TENSOR_NAME = 'MobilenetV1/Predictions/Reshape:0'
# RESIZED_INPUT_TENSOR_NAME = 'input:0'
# BOTTLENECK_TENSOR_SIZE = 1001
# MODEL_INPUT_WIDTH = 224
# MODEL_INPUT_HEIGHT = 224
# MODEL_INPUT_DEPTH = 3
# JPEG_DATA_TENSOR_NAME = 'DecodeJpeg/contents:0'


def create_model_info(architecture):

    architecture = architecture.lower()
    is_quantized = False
    if architecture == 'inception_v3':
        # pylint: disable=line-too-long
        data_url = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
        # pylint: enable=line-too-long
        bottleneck_tensor_name = 'pool_3/_reshape:0'
        bottleneck_tensor_size = 2048
        input_width = 299
        input_height = 299
        input_depth = 3
        resized_input_tensor_name = 'Mul:0'
        model_file_name = 'classify_image_graph_def.pb'
        input_mean = 128
        input_std = 128

    elif architecture.startswith('mobilenet_'): # mobilenet_1.0_224
        parts = architecture.split('_')
        if len(parts) != 3 and len(parts) != 4:
            tf.logging.error("Couldn't understand architecture name '%s'", architecture)
            return None
        # end if
        version_string = parts[1]
        if (version_string != '1.0' and version_string != '0.75' and version_string != '0.50' and version_string != '0.25'):
            tf.logging.error(""""The Mobilenet version should be '1.0', '0.75', '0.50', or '0.25', but found '%s' for architecture '%s'""", version_string, architecture)
            return None
        # end if
        size_string = parts[2]
        if (size_string != '224' and size_string != '192' and size_string != '160' and size_string != '128'):
            tf.logging.error("""The Mobilenet input size should be '224', '192', '160', or '128', but found '%s' for architecture '%s'""", size_string, architecture)
            return None
        # end if
        if len(parts) == 3:
            is_quantized = False
        else:
            if parts[3] != 'quantized':
                tf.logging.error(
                    "Couldn't understand architecture suffix '%s' for '%s'", parts[3], architecture)
                return None
            is_quantized = True
        # end if

        if is_quantized:
            data_url = 'http://download.tensorflow.org/models/mobilenet_v1_'
            data_url += version_string + '_' + size_string + '_quantized_frozen.tgz'
            bottleneck_tensor_name = 'MobilenetV1/Predictions/Reshape:0'
            resized_input_tensor_name = 'Placeholder:0'
            model_dir_name = ('mobilenet_v1_' + version_string + '_' + size_string + '_quantized_frozen')
            model_base_name = 'quantized_frozen_graph.pb'
        else:
            data_url = 'http://download.tensorflow.org/models/mobilenet_v1_'
            data_url += version_string + '_' + size_string + '_frozen.tgz'
            bottleneck_tensor_name = 'MobilenetV1/Predictions/Reshape:0'
            resized_input_tensor_name = 'input:0'
            model_dir_name = 'mobilenet_v1_' + version_string + '_' + size_string
            model_base_name = 'frozen_graph.pb'
        # end if

        bottleneck_tensor_size = 1001
        input_width = int(size_string)
        input_height = int(size_string)
        input_depth = 3
        model_file_name = os.path.join(model_dir_name, model_base_name)
        input_mean = 127.5
        input_std = 127.5
    else:
        tf.logging.error("Couldn't understand architecture name '%s'", architecture)
        raise ValueError('Unknown architecture', architecture)
    # end if

    return {'data_url': data_url, 'bottleneck_tensor_name': bottleneck_tensor_name, 'bottleneck_tensor_size': bottleneck_tensor_size,
            'input_width': input_width, 'input_height': input_height, 'input_depth': input_depth, 'resized_input_tensor_name': resized_input_tensor_name,
            'model_file_name': model_file_name, 'input_mean': input_mean, 'input_std': input_std, 'quantize_layer': is_quantized, }
# end function

def download_and_extract(data_url):
  """Download and extract model tar file.

  If the pretrained model we're using doesn't already exist, this function
  downloads it from the TensorFlow.org website and unpacks it into a directory.
  """
  dest_directory = FLAGS.model_dir
  if not os.path.exists(dest_directory):
    os.makedirs(dest_directory)
  filename = data_url.split('/')[-1]
  filepath = os.path.join(dest_directory, filename)
  if not os.path.exists(filepath):

    def _progress(count, block_size, total_size):
      sys.stdout.write('\r>> Downloading %s %.1f%%' %
                       (filename,
                        float(count * block_size) / float(total_size) * 100.0))
      sys.stdout.flush()

    filepath, _ = urllib.request.urlretrieve(data_url,
                                             filepath,
                                             _progress)
    print()
    statinfo = os.stat(filepath)
    print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
  tarfile.open(filepath, 'r:gz').extractall(dest_directory)

def create_model_graph(model_info):

    with tf.Graph().as_default() as graph:
        model_path = os.path.join(FLAGS.model_dir, model_info['model_file_name'])
        print('Model path: ', model_path)
        with gfile.FastGFile(model_path, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            bottleneck_tensor, resized_input_tensor = (tf.import_graph_def(graph_def, name='',
                             return_elements=[model_info['bottleneck_tensor_name'], model_info['resized_input_tensor_name'],]))
        # end with
    # end with
    return graph, bottleneck_tensor, resized_input_tensor
# end function

def main(_):
    print("-"*10,"Starting program","-"*10)
    # make sure the logging output is visible, see https://github.com/tensorflow/tensorflow/issues/3047
    tf.logging.set_verbosity(tf.logging.INFO)

    if tf.gfile.Exists(FLAGS.summaries_dir):
      tf.gfile.DeleteRecursively(FLAGS.summaries_dir)
    tf.gfile.MakeDirs(FLAGS.summaries_dir)


    # Gather information about the model architecture we'll be using.
    model_info = create_model_info(FLAGS.model_type)
    if not model_info:
        tf.logging.error('Did not recognize architecture flag')
        return -1
    # end if

    download_and_extract(model_info['data_url'])
    print("-"*10,"Creating model graph -",FLAGS.model_type,"-"*10)
    graph, bottleneck_tensor, resized_image_tensor = (create_model_graph(model_info))
    print (graph)

    # Look at the folder structure, and create lists of all the images.
    print("creating image lists . . .")
    image_lists = create_image_lists(TRAINING_IMAGES_DIR, TESTING_PERCENTAGE, VALIDATION_PERCENTAGE)
    class_count = len(image_lists.keys())
    if class_count == 0:
        tf.logging.error('No valid folders of images found at ' + TRAINING_IMAGES_DIR)
        return -1
    # end if
    if class_count == 1:
        tf.logging.error('Only one valid folder of images found at ' + TRAINING_IMAGES_DIR + ' - multiple classes are needed for classification.')
        return -1
    # end if

    # determinf if any of the distortion command line flags have been set
    doDistortImages = False
    if (FLIP_LEFT_RIGHT == True or RANDOM_CROP != 0 or RANDOM_SCALE != 0 or RANDOM_BRIGHTNESS != 0):
        doDistortImages = True
    # end if

    print("starting session . . .")
    with tf.Session(graph=graph) as sess:
        # Set up the image decoding sub-graph.
        print("performing jpeg decoding . . .")
        jpeg_data_tensor, decoded_image_tensor = add_jpeg_decoding( model_info['input_width'],
                                                                    model_info['input_height'],
                                                                    model_info['input_depth'],
                                                                    model_info['input_mean'],
                                                                    model_info['input_std'])
        print("caching bottlenecks . . .")
        distorted_jpeg_data_tensor = None
        distorted_image_tensor = None
        if doDistortImages:
            # We will be applying distortions, so setup the operations we'll need.
            (distorted_jpeg_data_tensor, distorted_image_tensor) = add_input_distortions(FLIP_LEFT_RIGHT, RANDOM_CROP, RANDOM_SCALE,
                                                                                         RANDOM_BRIGHTNESS, model_info['input_width'],
                                                                                         model_info['input_height'], model_info['input_depth'],
                                                                                         model_info['input_mean'], model_info['input_std'])
        else:
            # We'll make sure we've calculated the 'bottleneck' image summaries and
            # cached them on disk.
            cache_bottlenecks(sess, image_lists, TRAINING_IMAGES_DIR, BOTTLENECK_DIR, jpeg_data_tensor, decoded_image_tensor,
                              resized_image_tensor, bottleneck_tensor, ARCHITECTURE)
        # end if

        # Add the new layer that we'll be training.
        print("adding final training layer . . .")
        (train_step, cross_entropy, bottleneck_input, ground_truth_input, final_tensor) = add_final_training_ops(len(image_lists.keys()),
                                                                                                                 FINAL_TENSOR_NAME,
                                                                                                                 bottleneck_tensor,
                                                                                                                 model_info['bottleneck_tensor_size'],
                                                                                                                 model_info['quantize_layer'])
        # Create the operations we need to evaluate the accuracy of our new layer.
        print("adding eval ops for final training layer . . .")
        evaluation_step, prediction = add_evaluation_step(final_tensor, ground_truth_input)

        # Merge all the summaries and write them out to the tensorboard_dir
        print("writing TensorBoard info . . .")
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(TENSORBOARD_DIR + '/train', sess.graph)
        validation_writer = tf.summary.FileWriter(TENSORBOARD_DIR + '/validation')

        # Set up all our weights to their initial default values.
        init = tf.global_variables_initializer()
        sess.run(init)

        # Run the training for as many cycles as requested on the command line.
        print("performing training . . .")
        for i in range(HOW_MANY_TRAINING_STEPS):
            # Get a batch of input bottleneck values, either calculated fresh every
            # time with distortions applied, or from the cache stored on disk.
            if doDistortImages:
                (train_bottlenecks, train_ground_truth) = get_random_distorted_bottlenecks(sess, image_lists, TRAIN_BATCH_SIZE, 'training',
                                                                                           TRAINING_IMAGES_DIR, distorted_jpeg_data_tensor,
                                                                                           distorted_image_tensor, resized_image_tensor, bottleneck_tensor)
            else:
                (train_bottlenecks, train_ground_truth, _) = get_random_cached_bottlenecks(sess, image_lists, TRAIN_BATCH_SIZE, 'training',
                                                                                           BOTTLENECK_DIR, TRAINING_IMAGES_DIR, jpeg_data_tensor,
                                                                                           decoded_image_tensor, resized_image_tensor, bottleneck_tensor,
                                                                                           ARCHITECTURE)
            # end if

            # Feed the bottlenecks and ground truth into the graph, and run a training
            # step. Capture training summaries for TensorBoard with the `merged` op.
            train_summary, _ = sess.run([merged, train_step], feed_dict={bottleneck_input: train_bottlenecks, ground_truth_input: train_ground_truth})
            train_writer.add_summary(train_summary, i)

            # Every so often, print out how well the graph is training.
            is_last_step = (i + 1 == HOW_MANY_TRAINING_STEPS)
            if (i % EVAL_STEP_INTERVAL) == 0 or is_last_step:
                train_accuracy, cross_entropy_value = sess.run([evaluation_step, cross_entropy], feed_dict={bottleneck_input: train_bottlenecks, ground_truth_input: train_ground_truth})
                tf.logging.info('%s: Step %d: Train accuracy = %.1f%%' % (datetime.now(), i, train_accuracy * 100))
                tf.logging.info('%s: Step %d: Cross entropy = %f' % (datetime.now(), i, cross_entropy_value))
                validation_bottlenecks, validation_ground_truth, _ = (get_random_cached_bottlenecks(sess, image_lists, VALIDATION_BATCH_SIZE, 'validation',
                                                                                                    BOTTLENECK_DIR, TRAINING_IMAGES_DIR, jpeg_data_tensor,
                                                                                                    decoded_image_tensor, resized_image_tensor, bottleneck_tensor,
                                                                                                    ARCHITECTURE))
                # Run a validation step and capture training summaries for TensorBoard with the `merged` op.
                validation_summary, validation_accuracy = sess.run(
                    [merged, evaluation_step], feed_dict={bottleneck_input: validation_bottlenecks, ground_truth_input: validation_ground_truth})
                validation_writer.add_summary(validation_summary, i)
                tf.logging.info('%s: Step %d: Validation accuracy = %.1f%% (N=%d)' % (datetime.now(), i, validation_accuracy * 100, len(validation_bottlenecks)))
            # end if

            # Store intermediate results
            intermediate_frequency = INTERMEDIATE_STORE_FREQUENCY

            if (intermediate_frequency > 0 and (i % intermediate_frequency == 0) and i > 0):
                intermediate_file_name = (INTERMEDIATE_OUTPUT_GRAPHS_DIR + 'intermediate_' + str(i) + '.pb')
                tf.logging.info('Save intermediate result to : ' + intermediate_file_name)
                save_graph_to_file(sess, graph, intermediate_file_name)
            # end if
        # end for

        # We've completed all our training, so run a final test evaluation on some new images we haven't used before
        print("running testing . . .")
        test_bottlenecks, test_ground_truth, test_filenames = (get_random_cached_bottlenecks(sess, image_lists, TEST_BATCH_SIZE, 'testing', BOTTLENECK_DIR,
                                                                                             TRAINING_IMAGES_DIR, jpeg_data_tensor, decoded_image_tensor, resized_image_tensor,
                                                                                             bottleneck_tensor, ARCHITECTURE))
        test_accuracy, predictions = sess.run([evaluation_step, prediction], feed_dict={bottleneck_input: test_bottlenecks, ground_truth_input: test_ground_truth})
        tf.logging.info('Final test accuracy = %.1f%% (N=%d)' % (test_accuracy * 100, len(test_bottlenecks)))

        if PRINT_MISCLASSIFIED_TEST_IMAGES:
            tf.logging.info('=== MISCLASSIFIED TEST IMAGES ===')
            for i, test_filename in enumerate(test_filenames):
                if predictions[i] != test_ground_truth[i]:
                    tf.logging.info('%70s  %s' % (test_filename, list(image_lists.keys())[predictions[i]]))
                # end if
            # end for
        # end if

        # write out the trained graph and labels with the weights stored as constants
        print("writing trained graph and labbels with weights")
        save_graph_to_file(sess, graph, OUTPUT_GRAPH)
        with gfile.FastGFile(OUTPUT_LABELS, 'w') as f:
            f.write('\n'.join(image_lists.keys()) + '\n')
        # end with

        print("done !!")
# end function

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--image_dir',
      type=str,
      default='',
      help='Path to folders of labeled images.'
  )
  parser.add_argument(
      '--model_type',
      type=str,
      default='mobilenet_1.0_224',
      help='Path to folders of labeled images.'
  )
  parser.add_argument(
      '--output_graph',
      type=str,
      default='/tmp/output_graph_mobilenet_run_1.pb',
      help='Where to save the trained graph.'
  )
  parser.add_argument(
      '--output_labels',
      type=str,
      default='/tmp/output_labels_mobilenet_run_1.txt',
      help='Where to save the trained graph\'s labels.'
  )
  parser.add_argument(
      '--summaries_dir',
      type=str,
      default='/tmp/retrain_logs_mobilenet',
      help='Where to save summary logs for TensorBoard.'
  )
  parser.add_argument(
      '--how_many_training_steps',
      type=int,
      default=4000,
      help='How many training steps to run before ending.'
  )
  parser.add_argument(
      '--learning_rate',
      type=float,
      default=0.01,
      help='How large a learning rate to use when training.'
  )
  parser.add_argument(
      '--testing_percentage',
      type=int,
      default=10,
      help='What percentage of images to use as a test set.'
  )
  parser.add_argument(
      '--validation_percentage',
      type=int,
      default=10,
      help='What percentage of images to use as a validation set.'
  )
  parser.add_argument(
      '--eval_step_interval',
      type=int,
      default=10,
      help='How often to evaluate the training results.'
  )
  parser.add_argument(
      '--train_batch_size',
      type=int,
      default=100,
      help='How many images to train on at a time.'
  )
  parser.add_argument(
      '--test_batch_size',
      type=int,
      default=-1,
      help="""\
      How many images to test on. This test set is only used once, to evaluate
      the final accuracy of the model after training completes.
      A value of -1 causes the entire test set to be used, which leads to more
      stable results across runs.\
      """
  )
  parser.add_argument(
      '--validation_batch_size',
      type=int,
      default=100,
      help="""\
      How many images to use in an evaluation batch. This validation set is
      used much more often than the test set, and is an early indicator of how
      accurate the model is during training.
      A value of -1 causes the entire validation set to be used, which leads to
      more stable results across training iterations, but may be slower on large
      training sets.\
      """
  )
  parser.add_argument(
      '--print_misclassified_test_images',
      default=False,
      help="""\
      Whether to print out a list of all misclassified test images.\
      """,
      action='store_true'
  )
  parser.add_argument(
      '--model_dir',
      type=str,
      default='/tmp/mobilenet',
      help="""\
      Path to classify_image_graph_def.pb,
      imagenet_synset_to_human_label_map.txt, and
      imagenet_2012_challenge_label_map_proto.pbtxt.\
      """
  )
  parser.add_argument(
      '--bottleneck_dir',
      type=str,
      default='/tmp/mobilenet_bottleneck',
      help='Path to cache bottleneck layer values as files.'
  )
  parser.add_argument(
      '--final_tensor_name',
      type=str,
      default='final_result',
      help="""\
      The name of the output classification layer in the retrained graph.\
      """
  )
  parser.add_argument(
      '--flip_left_right',
      default=False,
      help="""\
      Whether to randomly flip half of the training images horizontally.\
      """,
      action='store_true'
  )
  parser.add_argument(
      '--random_crop',
      type=int,
      default=0,
      help="""\
      A percentage determining how much of a margin to randomly crop off the
      training images.\
      """
  )
  parser.add_argument(
      '--random_scale',
      type=int,
      default=0,
      help="""\
      A percentage determining how much to randomly scale up the size of the
      training images by.\
      """
  )
  parser.add_argument(
      '--random_brightness',
      type=int,
      default=0,
      help="""\
      A percentage determining how much to randomly multiply the training image
      input pixels up or down by.\
      """
  )
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
