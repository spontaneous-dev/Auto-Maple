import numpy as np
import os, cv2

import numpy as np
import os
import tensorflow as tf
import cv2
from object_detection.utils import label_map_util


from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util


#################################
#       Global Variables        #
#################################
crop_state = True
filter_color_state = True
canny_state = True
crop_arrows_state = True
model_name = 'rune_model_rnn_filtered_cannied'
export_folder = 'classify'


#################################
#             Main              #
#################################
def load_model():
    model_dir = f'exported_models/{model_name}/saved_model'
    model = tf.saved_model.load(str(model_dir))
    return model


category_index = label_map_util.create_category_index_from_labelmap('annotations/label_map.pbtxt', use_display_name=True)
detection_model = load_model()


def run_inference_for_single_image(model, image):
    image = np.asarray(image)

    input_tensor = tf.convert_to_tensor(image)
    input_tensor = input_tensor[tf.newaxis,...]

    model_fn = model.signatures['serving_default']
    output_dict = model_fn(input_tensor)
    
    num_detections = int(output_dict.pop('num_detections'))
    output_dict = {key: value[0,:num_detections].numpy() 
                   for key, value in output_dict.items()}
    output_dict['num_detections'] = num_detections
    output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)
    return output_dict


def get_boxes(model, image):
    output_dict = run_inference_for_single_image(model, image)
    zipped = list(zip(output_dict['detection_scores'], output_dict['detection_boxes'], output_dict['detection_classes']))
    pruned = [tuple for tuple in zipped if tuple[0] > 0.5]
    pruned.sort(key=lambda x: x[0], reverse=True)
    pruned = pruned[:4]
    boxes = [tuple[1:] for tuple in pruned]
    return boxes


# def show_inference(model, image_path='', image_np=None):
#     # the array based representation of the image will be used later in order to prepare the
#     # result image with boxes and labels on it.
#     if image_path:
#         image_np = np.array(Image.open(image_path))

#     # Actual detection.
#     output_dict = run_inference_for_single_image(model, image_np)
#     # Visualization of the results of a detection.
#     vis_util.visualize_boxes_and_labels_on_image_array(
#         image_np,
#         output_dict['detection_boxes'],
#         output_dict['detection_classes'],
#         output_dict['detection_scores'],
#         category_index,
#         instance_masks=output_dict.get('detection_masks_reframed', None),
#         use_normalized_coordinates=True,
#         line_thickness=8)

#     cv2.imshow('inference', image_np)


def filter_color(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (1, 100, 100), (75, 255, 255))

    # Slice the mask
    imask = mask > 0
    arrows = np.zeros_like(image, np.uint8)
    arrows[imask] = image[imask]
    return arrows


def canny(image):
    height, width, channels = image.shape
    image = cv2.Canny(image, 200, 300)
    # cropped = image[:height//2,width//4:3*width//4]
    colored = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return colored


counter = 0
for file in os.listdir():
    if '.jpg' in file:
        # counter += 1
        image = cv2.imread(file)
        height, width, channels = image.shape
        if crop_state:
            image = image[:height//2,width//3:2*width//3]
        if filter_color_state:
            image = filter_color(image)
        if canny_state:
            image = canny(image)
        if crop_arrows_state:
            boxes = get_boxes(detection_model, image)
            height, width, channels = image.shape
            # print(boxes)
            for box, c in boxes:
                counter += 1
                print(f"class: {category_index[c]['name']}")
                left = int(round(box[1] * width))
                right = int(round(box[3] * width))
                top = int(round(box[0] * height))
                bottom = int(round(box[2] * height))
                print(left, right, top, bottom)
                cv2.imwrite(f'{export_folder}/cropped{counter}.jpg', image[top:bottom, left:right])
                print(counter, '\n')
        else:
            print(counter)
            cv2.imwrite(f'{export_folder}/cropped{counter}.jpg', image)
        
        
print('Finished!')