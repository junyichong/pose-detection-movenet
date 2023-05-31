import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import cv2
import csv
import os


# Get coordinates of body part
def get_coordinates(landmarks, body_part):
    if hasattr(mp_pose.PoseLandmark, body_part):
        body_part_index = mp_pose.PoseLandmark[body_part].value
        x = landmarks[body_part_index].x
        y = landmarks[body_part_index].y
        z = landmarks[body_part_index].z

        return [x, y]
    else:
        return None


# Calculate angle between three points
def calculate_angle(a,b,c):
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    
    angle_rad = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle_deg = np.abs(angle_rad*180.0/np.pi)
    
    if angle_deg >180.0:
        angle_deg = 360-angle_deg
        
    return angle_deg

# Calculate angle between three points in 3D space
def calculate_angle_3d(a,b,c):
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End

    vector_ab = b - a
    vector_bc = c - b

    unit_vector_ab = vector_ab / np.linalg.norm(vector_ab)
    unit_vector_bc = vector_bc / np.linalg.norm(vector_bc)
    dot_product = np.dot(unit_vector_ab, unit_vector_bc)
    angle_rad = np.arccos(dot_product)
    angle_deg = np.degrees(angle_rad)
        
    return angle_deg

# a = [1,2,3]
# b = [2,5,6]
# c = [1,8,9]

# print(calculate_angle_3d(a,b,c))


# Modify the keypoints to fit the frame
def modify_keypoints(frame, keypoints):
    height, width, *_ = frame.shape
    box_size = max(height, width)
    modified_keypoints = []

    for y, x, confidence in keypoints[0][0]:
        # calculate the positions of body parts on the frame
        modified_x = int(x * box_size - (box_size - width) / 2)
        modified_y = int(y * box_size - (box_size - height) / 2)
        modified_keypoints.append((modified_y, modified_x, confidence))

    return modified_keypoints


# Draw keypoints
def draw_keypoints(frame, keypoints, confidence_threshold):
    for y, x, confidence in keypoints:
        if confidence > confidence_threshold:
            # draw a point on each body part
            cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)

EDGES = {
    (0, 1): 'm',
    (0, 2): 'c',
    (1, 3): 'm',
    (2, 4): 'c',
    (0, 5): 'm',
    (0, 6): 'c',
    (5, 7): 'm',
    (7, 9): 'm',
    (6, 8): 'c',
    (8, 10): 'c',
    (5, 6): 'y',
    (5, 11): 'm',
    (6, 12): 'c',
    (11, 12): 'y',
    (11, 13): 'm',
    (13, 15): 'm',
    (12, 14): 'c',
    (14, 16): 'c'
}

# Draw lines between keypoints
def draw_connections(frame, keypoints, edges, confidence_threshold): 
    for edge, color in edges.items():
        p1, p2 = edge
        y1, x1, c1 = modified_keypoints[p1]
        y2, x2, c2 = modified_keypoints[p2]
        
        if (c1 > confidence_threshold) & (c2 > confidence_threshold):      
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,0,255), 2)


interpreter = tf.lite.Interpreter(model_path='lite-model_movenet_singlepose_thunder_3.tflite')
interpreter.allocate_tensors()

video_path = 'test_videos/talk.mp4'
cap = cv2.VideoCapture(video_path)

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

folder_path = os.path.join(os.getcwd(), 'results')
os.makedirs(folder_path, exist_ok=True)
csv_path = os.path.join(folder_path, 'talk.csv')

with open(csv_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['left angle', 'right angle'])


while cap.isOpened():
    ret, frame = cap.read()
    
    # Reshape image
    img = frame.copy()
    img = tf.image.resize_with_pad(np.expand_dims(img, axis=0), 256, 256)
    input_image = tf.cast(img, dtype=tf.float32)
    
    # Setup input and output 
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Make predictions 
    interpreter.set_tensor(input_details[0]['index'], input_image.numpy())
    interpreter.invoke()
    keypoints_with_scores = interpreter.get_tensor(output_details[0]['index'])
    
    # Rendering 
    modified_keypoints = modify_keypoints(frame, keypoints_with_scores)
    draw_connections(frame, modified_keypoints, EDGES, 0.3)
    draw_keypoints(frame, modified_keypoints, 0.3)
    
    cv2.imshow('MoveNet Lightning', frame)
    
    if cv2.waitKey(10) & 0xFF==ord('q'):
        break
        
cap.release()
cv2.destroyAllWindows()