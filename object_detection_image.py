import numpy as np
import argparse
import os
import tensorflow as tf
from PIL import Image
from io import BytesIO
import glob
import matplotlib.pyplot as plt

from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

# Patch tf1 into `utils.ops`
utils_ops.tf = tf.compat.v1

# Patch the location of gfile
tf.gfile = tf.io.gfile

def load_model(model_path):
    """Loads a TensorFlow SavedModel from a given path."""
    model = tf.saved_model.load(model_path)
    return model

def load_image_into_numpy_array(path):
    """Loads an image from a given path and converts it to a numpy array."""
    img_data = tf.io.gfile.GFile(path, 'rb').read()
    image = Image.open(BytesIO(img_data))
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


def run_inference_for_single_image(model, image):
    """
       Runs object detection on a single image using the given model.
       Returns a dictionary containing detection results.
       """
    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(image)
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis, ...]

    # Run inference
    output_dict = model(input_tensor)

    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    num_detections = int(output_dict.pop('num_detections'))
    output_dict = {key: value[0, :num_detections].numpy()
                   for key, value in output_dict.items()}
    output_dict['num_detections'] = num_detections

    # detection_classes should be ints.
    output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)

    # Handle models with masks:
    if 'detection_masks' in output_dict:
        # Reframe the bbox mask to the image size.
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            output_dict['detection_masks'], output_dict['detection_boxes'],
            image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5, tf.uint8)
        output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()

    return output_dict

def run_inference(model, category_index, image_path):
    """
       Runs object detection on one or more images using the given model and saves the
       detection results as image files in a specified folder.
       """
    if os.path.isdir(image_path):
        image_paths = []
        for file_extension in ('*.png', '*jpg'):
            image_paths.extend(glob.glob(os.path.join(image_path, file_extension)))

        """add iterator here"""
        i = 0
        for i_path in image_paths:
            image_np = load_image_into_numpy_array(i_path)
            # Actual detection.
            output_dict = run_inference_for_single_image(model, image_np)
            # Visualization of the results of a detection.
            vis_util.visualize_boxes_and_labels_on_image_array(
                image_np,
                output_dict['detection_boxes'],
                output_dict['detection_classes'],
                output_dict['detection_scores'],
                category_index,
                instance_masks=output_dict.get('detection_masks_reframed', None),
                use_normalized_coordinates=True,
                line_thickness=8)
            """The existing plt lines do not work on local pc as they are not setup for GUI
                Use plt.savefig() to save the results instead and view them in a folder"""
            plt.imshow(image_np)
            plt.savefig("outputs/detection_output{}.png".format(i))
            i = i + 1


if __name__ == '__main__':
    # Define command line arguments
    parser = argparse.ArgumentParser(description='Detect objects Inside Image')
    parser.add_argument('-m', '--model', type=str, required=True, help='Model Path')
    parser.add_argument('-l', '--labelmap', type=str, required=True, help='Path to Labelmap')
    parser.add_argument('-i', '--image_path', type=str, required=True, help='Path to image folder')
    args = parser.parse_args()

    # Load the detection model and create a category index
    detection_model = load_model(args.model)
    category_index = label_map_util.create_category_index_from_labelmap(args.labelmap, use_display_name=True)

    # Run object detection on the image(s)
    run_inference(detection_model, category_index, args.image_path)

# Script
# python object_detection_image.py  -m .\exported-models\my_model\saved_model -l .\annotations\label_map.pbtxt -i .\images\test_images
