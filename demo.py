###conda activate test_yolov5

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'} to control the verbosity
import cv2
import numpy as np
import time
from ultralytics import YOLO

from libs.bbox3d_utils import *
from train import * 


####### select model  ########
# select_model = 'resnet50'
# select_model ='resnet101'
# select_model = 'resnet152'
# select_model = 'vgg11'
# select_model = 'vgg16'
# select_model = 'vgg19'
# select_model = 'efficientnetb0'
# select_model = 'efficientnetb5'
select_model = 'mobilenetv2'



# Load the 3D model
bbox3d_model = load_model('./'+select_model+'/'+select_model+'_weights.h5')
bin_size = 6
input_shape = (224, 224, 3)
trained_classes = ['Car', 'Cyclist', 'Pedestrian']
# print(bbox3d_model.summary())
print('loading file ...'+select_model+'_weights.h5...!')
P2 = np.array([[718.856, 0.0, 607.1928, 45.38225], [0.0, 718.856, 185.2157, -0.1130887], [0.0, 0.0, 1.0, 0.003779761]])
dims_avg = {'Car': np.array([1.52131309, 1.64441358, 3.85728004]),
'Van': np.array([2.18560847, 1.91077601, 5.08042328]),
'Truck': np.array([3.07044968,  2.62877944, 11.17126338]),
'Pedestrian': np.array([1.75562272, 0.67027992, 0.87397566]),
'Person_sitting': np.array([1.28627907, 0.53976744, 0.96906977]),
'Cyclist': np.array([1.73456498, 0.58174006, 1.77485499]),
'Tram': np.array([3.56020305,  2.40172589, 18.60659898])}
# print(dims_avg)



# Load a 2D model
bbox2d_model = YOLO('yolov8n-seg.pt')  # load an official model
# bbox2d_model.cuda()
# set model parameters
bbox2d_model.overrides['conf'] = 0.9  # NMS confidence threshold
bbox2d_model.overrides['iou'] = 0.45  # NMS IoU threshold
bbox2d_model.overrides['agnostic_nms'] = False  # NMS class-agnostic
bbox2d_model.overrides['max_det'] = 1000  # maximum number of detections per image
bbox2d_model.overrides['classes'] = 2 ## define classes
yolo_classes = ['Pedestrian', 'Cyclist', 'Car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat']


# Load the video
video = cv2.VideoCapture('./assets/2011_10_03_drive_0034_sync_video_trimmed.mp4')


### svae results
# Get video information (frame width, height, frames per second)
frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(video.get(cv2.CAP_PROP_FPS))
# Define the codec and create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Change the codec if needed (e.g., 'XVID')
out = cv2.VideoWriter(select_model+'_output_video.mp4', fourcc, 15, (frame_width, frame_height))




frameId = 0
start_time = time.time()
fps = str()
DIMS = []

# Process each frame of the video
while True:
  frameId+=1
  success, frame = video.read()


  img = frame.copy() 
  img2 = frame.copy() 
  img3 = frame.copy() 
  if not success:
    break

  # Perform featuresect detection on the frame
  results = bbox2d_model(frame, verbose=False)  # predict on an image
  # print(results)

  ## object detections
  for predictions in results:
      bbox = predictions.boxes.xyxy
      if predictions.boxes.cls.numel() > 0:
          class_id = predictions.boxes.cls[0].cpu()
      else:
          class_id = None
      if bbox is not None:
          for detection in bbox:
              padding = 0  # Set the padding value
              xmin = max(0, detection[0] - padding)
              ymin = max(0, detection[1] - padding)
              xmax = min(frame.shape[1], detection[2] + padding)
              ymax = min(frame.shape[0], detection[3] + padding)
              cv2.rectangle(img2, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0,0,225), 1)
              crop = img[int(ymin) : int(ymax), int(xmin) : int(xmax)]

              patch = tf.convert_to_tensor(crop, dtype=tf.float32)
              patch /= 255.0  # Normalize to [0,1]
              patch = tf.image.resize(patch, (224, 224))  # Resize to 224x224
              patch = tf.expand_dims(patch, axis=0)  # Equivalent to reshape((1, *crop.shape))
              prediction = bbox3d_model.predict(patch, verbose = 0)

              dim = prediction[0][0]
              bin_anchor = prediction[1][0]
              bin_confidence = prediction[2][0]

              ###refinement dimension
              try:
                  dim += dims_avg[str(yolo_classes[int(class_id.cpu().numpy())])] + dim
                  DIMS.append(dim)
              except:
                  dim = DIMS[-1]

              bbox_ = [int(xmin), int(ymin), int(xmax), int(ymax)]
              theta_ray = calc_theta_ray(frame, bbox_, P2)
              # update with predicted alpha, [-pi, pi]
              alpha = recover_angle(bin_anchor, bin_confidence, bin_size)
              alpha = alpha - theta_ray

              ## plot3d bbox on image
              plot3d(img3, P2, bbox_, dim, alpha, theta_ray)


  ## object segmentations
  masks = predictions.masks
  if masks is not None:
      for mask in masks.xy:
          polygon = mask
          cv2.polylines(img3, [np.int32(polygon)], True, (0, 0, 225), thickness=2)


  # Calculate the current time in seconds
  current_time = video.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
  if frameId % 20 == 0:  # Calculate FPS every 10 frames
      end_time = time.time()
      elapsed_time = end_time - start_time
      fps_current = frameId / elapsed_time
      fps =  f'FPS: {fps_current:.2f}'
      # print(f'Frame: {frameId}, FPS: {fps_current:.2f}')
  cv2.putText(img3, select_model+' '+fps, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1, cv2.LINE_AA)


  # Display the frame
  cv2.imshow("2D", img2)
  cv2.imshow("3D", img3)
  out.write(img3)
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

# Release the video capture featuresect
video.release()
cv2.destroyAllWindows()
