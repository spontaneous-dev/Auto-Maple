#################################
#       Global Variables        #
#################################
# model_name = 'rune_model_rnn'
model_name = 'my_model'


#################################
#        Initialization         #
#################################
# Instructions for training the model
print('\nTrain the model:')
print(f'python model_main_tf2.py\
    --model_dir=models/{model_name}\
    --pipeline_config_path=models/{model_name}/pipeline.config\n')

# Instructions for monitoring training using Tensorboard
print('Monitor training using Tensorboard:')
print('cd c:/users/tanje/appdata/roaming/python/python37/site-packages/tensorboard')
print(f'python main.py --logdir=c:/Users/tanje/Desktop/Rune\ Detection/models/{model_name}\n')

# Instructions for exporting the trained model to .pb
print('Export the trained model:')
print(f'python exporter_main_v2.py\
    --input_type image_tensor\
    --pipeline_config_path models/{model_name}/pipeline.config\
    --trained_checkpoint_dir models/{model_name}\
    --output_directory exported_models/{model_name}\n')

input('Press ENTER to continue with inference')


#################################
#           Imports             #
#################################
import numpy as np
import os
import tensorflow as tf
import cv2
import math
from object_detection.utils import label_map_util


from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util




#################################
#             Main              #
#################################
def load_model():
    model_dir = f'exported_models/{model_name}/saved_model'
    model = tf.saved_model.load(str(model_dir))
    return model


category_index = label_map_util.create_category_index_from_labelmap('annotations/label_map.pbtxt', use_display_name=True)
detection_model = load_model()


def canny(image):
    # height, width, channels = image.shape
    image = cv2.Canny(image, 200, 300)
    # cropped = image[:height//2,width//3:2*width//3]
    colored = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return colored

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



# def classify_arrow(model, image):       # TODO: Does not work! None of the detection boxes are > 0.5 confidence 
#     output_dict = run_inference_for_single_image(model, image)
#     zipped = list(zip(output_dict['detection_scores'], output_dict['detection_classes']))
#     pruned = [tuple for tuple in zipped if tuple[0] > 0.5]
#     pruned.sort(key=lambda x: x[0], reverse=True)
#     # pruned = pruned[:1]
#     # classes = [category_index[pruned[0][1]]['name'] for tuple in pruned]
#     cv2.imshow('arrow', image)
#     cv2.waitKey(0)
#     return category_index[pruned[0][1]]['name']

def show_inference(model, image_path='', image_np=None):
    # the array based representation of the image will be used later in order to prepare the
    # result image with boxes and labels on it.
    if image_path:
        image_np = np.array(Image.open(image_path))

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

    cv2.imshow('inference', image_np)
    cv2.waitKey(0)

def filter_color(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (1, 100, 100), (75, 255, 255))

    # Mask the image
    imask = mask > 0
    arrows = np.zeros_like(image, np.uint8)
    arrows[imask] = image[imask]
    return arrows

def get_boxes(model, image):
    output_dict = run_inference_for_single_image(model, image)
    zipped = list(zip(output_dict['detection_scores'], output_dict['detection_boxes']))
    pruned = [tuple for tuple in zipped if tuple[0] > 0.5]
    pruned.sort(key=lambda x: x[0], reverse=True)
    pruned = pruned[:4]
    boxes = [tuple[1] for tuple in pruned]
    boxes.sort(key=lambda x: x[1])
    return boxes

def classify_arrow(average_45, average_135):
    if average_45 < 0 and average_135 < 0:
        return 'up'
    elif average_45 > 0 and average_135 > 0:
        return 'down'
    elif average_45 < 0 and average_135 > 0:
        return 'left'
    elif average_45 > 0 and average_135 < 0:
        return 'right'



def sort_by_confidence(model, image):
    output_dict = run_inference_for_single_image(model, image)
    zipped = list(zip(output_dict['detection_scores'], output_dict['detection_boxes'], output_dict['detection_classes']))
    pruned = [tuple for tuple in zipped if tuple[0] > 0.5]
    pruned.sort(key=lambda x: x[0], reverse=True)
    result = pruned[:4]
    # pruned.sort(key=lambda x: x[1][1])
    # arrows = [category_index[tuple[2]]['name'] for tuple in pruned]
    return result

def merge_detection(model, image):
    label_map = {1: 'up', 2: 'down', 3: 'left', 4: 'right'}
    converter = {'up': 'right', 'down': 'left'}
    
    # Preprocessing
    height, width, channels = image.shape
    cropped = image[:height//2,width//4:3*width//4]
    filtered = filter_color(cropped)
    cannied = canny(filtered)

    # Run detection on preprocessed image
    lst = sort_by_confidence(model, cannied)
    lst.sort(key=lambda x: x[1][1])
    classes = [label_map[item[2]] for item in lst]
    print(f'non-rotated: {classes}')

    # Run detection rotated image
    rotated = cv2.rotate(cannied, cv2.ROTATE_90_COUNTERCLOCKWISE)
    lst = sort_by_confidence(model, rotated)
    lst.sort(key=lambda x: x[1][2], reverse=True)
    rotated_classes = [converter[label_map[item[2]]]
                       for item in lst
                       if item[2] in [1, 2]]
        
    # Merge the two detection results
    for i in range(len(classes)):
        if rotated_classes and classes[i] in ['left', 'right']:
            classes[i] = rotated_classes.pop(0)

    return classes




use_dataset = True

if use_dataset:
    # os.chdir('filtered_cannied/train')
    pass
else:
    # os.chdir('TEST_IMAGES/')
    os.chdir('C:/Users/tanje/Desktop/')

files = [file for file in os.listdir() if os.path.isfile(file) and '.jpg' in file]
for file_name in files:
    img = cv2.imread(file_name)
    if not use_dataset:
        height, width, channels = img.shape
        # cropped = img
        cropped = img[150:height//2,width//4:3*width//4]
        # height, width, channels = cropped.shape
        # filtered = filter_color(cropped)
        # img = canny(filtered)
    
    height, width, channels = img.shape
        # cropped = img
    cropped = img[150:height//2,width//4:3*width//4]

    #################################
    #       Original Approach       #
    #################################
    # img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

    show_inference(detection_model, image_np=cropped)
    # print(merge_detection(detection_model, img), '\n')
    # print(classify_arrow(detection_model, img))
    # print(get_arrows(detection_model, processed))
    # cv2.waitKey(0)
    # cv2.imshow('processed', processed)
    # cv2.imshow('cropped', cropped)
    # cv2.waitKey(0)

"""
    #############################################
    #       Trying out a two step process       #
    #############################################
    boxes = get_boxes(detection_model, img)
    for box in boxes:
        print(box)
        left = int(round(box[1] * width))
        right = int(round(box[3] * width))
        top = int(round(box[0] * height))
        bottom = int(round(box[2] * height))
        arrow = img[top:bottom, left:right]

        # # Sobel calculations
        # sobel_x = cv2.Sobel(arrow, cv2.CV_64F, 1, 0, ksize=5)
        # sobel_y = cv2.Sobel(arrow, cv2.CV_64F, 0, 1, ksize=5)
        # res_x, res_y = np.square(sobel_x * math.sin(math.pi / 4)), np.square(sobel_y * math.cos(math.pi / 4))
        # result = np.sqrt(np.add(res_x, res_y))

        
        gray = cv2.cvtColor(arrow, cv2.COLOR_BGR2GRAY)
        
        
        #############################
        #       Hough Transform     #
        #############################

        # Blur
        kernel = np.ones((3, 3), np.float32) / 3 ** 2
        gray = cv2.filter2D(gray, -1, kernel)

        # # # Dilation and then erosion
        # # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        # # dilated = cv2.dilate(gray, kernel)
        # # eroded = cv2.erode(dilated, kernel)

        # # for i in range(1, 21):
        # #     lines = cv2.HoughLines(gray, 1, 0.5 * np.pi / 180, i * 5)
        # #     print(f'i={i * 5}\n{lines}\n')


        # lines = cv2.HoughLines(gray, 1, np.pi / 180, 10)
        # hlines_45 = [line[0] for line in filter(lambda t: np.pi / 6 < t[0][1] < np.pi / 3 and abs(t[0][0]) > 5, lines)]
        # hlines_135 = [line[0] for line in filter(lambda t: 2 * np.pi / 3 < t[0][1] < 5 * np.pi / 6 and abs(t[0][0]) > 5, lines)]
        # average_45 = sum([line[0] for line in hlines_45]) / len(hlines_45)
        # average_135 = sum([line[0] for line in hlines_135]) / len(hlines_135)
        # print(hlines_45, hlines_135)
        # print(classify_arrow(average_45, average_135))
        # # lines = cv2.HoughLinesP(gray, 1, np.pi / 180, 15, minLineLength=5)
        # # print(len(lines))
        # # for x1, y1, x2, y2 in lines[0]:
        # #     cv2.line(gray, (x1, y1), (x2, y2), (255, 255, 255), 2)

        # cv2.imshow('arrow', gray)
        # cv2.waitKey(0)


        ######################################
        #       Harris Corner Detection      #
        ######################################
        gray = np.float32(gray)
        dst = cv2.cornerHarris(gray,2,3,0.04)
        dst = cv2.dilate(dst,None)
        ret, dst = cv2.threshold(dst,0.01*dst.max(),255,0)
        dst = np.uint8(dst)

        # find centroids
        ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)

        # define the criteria to stop and refine the corners
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
        corners = cv2.cornerSubPix(gray,np.float32(centroids),(5,5),(-1,-1),criteria)

        # Now draw them
        res = np.hstack((centroids,corners))
        res = np.int0(res)
        arrow[res[:,1],res[:,0]]=[0,0,255]
        arrow[res[:,3],res[:,2]] = [0,255,0]

        cv2.imshow('arrow', arrow)
        cv2.waitKey(0)

    
    print(boxes)
"""