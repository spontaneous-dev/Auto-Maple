import numpy as np
import os, cv2
# import tensorflow as tf

crop_state = True
filter_color_state = True
canny_state = True


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
        counter += 1
        image = cv2.imread(file)
        height, width, channels = image.shape
        if crop_state:
            image = image[:height//2,width//3:2*width//3]
        if filter_color_state:
            image = filter_color(image)
        if canny_state:
            image = canny(image)
        # cropped = image[:height//2,width//3:2*width//3]
        # colored = cv2.cvtColor(cropped, cv2.COLOR_GRAY2BGR)
        # print(counter, tf.shape(colored))
        print(counter)
        cv2.imwrite(f'filtered_cannied/cropped{counter}.jpg', image)
        
        
print('Finished!')