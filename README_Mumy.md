# Traffic Light Detection and Classification for Simulator
##### Team Member - Muhamed Kuric
##### Email - mumymania@gmail.com

# Traffic Light Detection Node

* This part followed mostly the Detection Walkthrough by Stephen and Aaron. 

* This node takes in data from the `/image_color`, `/current_pose`, and `/base_waypoints` topics and publishes the locations to stop for red traffic lights to the `/traffic_waypoint` topic. 

![Detector](writeup_imgs/tl-detector-ros-graph.png)

* The topic `/current_pose` receives information about the current pose of the vehicle (i.e., localization data), while the topic `/base_waypoints` receives the waypoints that the vehicle should follow (\i.e., planning). The `/image_color` topic receives the raw image from the front-looking camera sensor mounted to the vehicle. The world locations of all traffic lights in the map are also loaded from a configuration file into a lookup table. 

* Based on the current pose of the vehicle and the traffic light lookup, we find if a traffic light is visible in front of the vehicle. If so, then we run a classifier that classifies the state of the traffic light (i.e., red vs non-red). If the traffic light was red we send the waypoint index of where this traffic light is located on the path via the `/traffic_waypoint` output topic. Otherwise, we publish a default message which means that there is no red traffic light in front of us, and that the vehicle can continue following its path.

## Traffic Light Classification for Simulator

* In order to train the classifier, we have logged the images from the `/image_color` during a successful automated simulation drive that used groundtruth traffic light state data that is also provided by the simulator. These training images can be found at [training-data](ros/src/tl_detector/sim-training-data/).

* The simulator ensures perfect repeatability of the input images, thus our strategy was to train a simple CNN-based classifier in Keras that overfits to these images. Training and prediction test scripts can be found at [learning](ros/src/tl_detector/train.py) [prediction](ros/src/tl_detector/test_prediction.py).

* The following NN architecture was used with full size camera images as input:
```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_1 (Conv2D)            (None, 586, 786, 8)       5408      
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 293, 393, 8)       0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 281, 381, 16)      21648     
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 140, 190, 16)      0         
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 130, 180, 24)      46488     
_________________________________________________________________
max_pooling2d_3 (MaxPooling2 (None, 65, 90, 24)        0         
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 57, 82, 32)        62240     
_________________________________________________________________
max_pooling2d_4 (MaxPooling2 (None, 28, 41, 32)        0         
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 22, 35, 40)        62760     
_________________________________________________________________
max_pooling2d_5 (MaxPooling2 (None, 11, 17, 40)        0         
_________________________________________________________________
conv2d_6 (Conv2D)            (None, 7, 13, 48)         48048     
_________________________________________________________________
max_pooling2d_6 (MaxPooling2 (None, 3, 6, 48)          0         
_________________________________________________________________
flatten_1 (Flatten)          (None, 864)               0         
_________________________________________________________________
dense_1 (Dense)              (None, 32)                27680     
_________________________________________________________________
dropout_1 (Dropout)          (None, 32)                0         
_________________________________________________________________
dense_2 (Dense)              (None, 24)                792       
_________________________________________________________________
dropout_2 (Dropout)          (None, 24)                0         
_________________________________________________________________
dense_3 (Dense)              (None, 2)                 50        
=================================================================
Total params: 275,114
Trainable params: 275,114
Non-trainable params: 0
_________________________________________________________________
```