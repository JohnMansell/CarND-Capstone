# Capstone Project Writeup
#### Team Members
> **John Mansell** - JohnMansellis@gmail.com  
> **Deepanshu Malhotra** - deepanshumalhotra@gmail.com  
> **Muhamed Kuric** - mumymania@gmail.com  
> **Sven Eckelmann** - eckelmann.sven@gmail.com  
> **Faris Janjos** - farisjanjos@gmail.com  

---
# Waypoint Updater Node
> **Name:** John Mansell   
> **E-Mail:** JohnMansellis@gmail.com

---
## Introduction
> My part of the project focused on finishing up the waypoint updater, which had been started by one of the other
students. The function of the waypoint updater is to publish the waypoints in front of the car, including their
associated velocity vector. This data is then used by the car dbw node to drive the car along the highway, ensuring that
the car drives at the appropriate speed, and obeys traffic light rules.

## Procedure
> ### Base Waypoints
> The first step in the waypoint updater is to acquire the base waypoints by subscribing to the /base_waypoints node.
The base waypoints include the desired path and maximum speed (speed limit) for each waypoint.
>
> ### Traffic Lights
> The second step is to determine if the car is approaching a traffic light. If it is, then the traffic light classifier
determines the state of the traffic light. The traffic light state was categorized into two classes, red and not-red.
>
> ### Decelerate
> If the car is approaching a red traffic light the next step is to calculate an appropriate rate of deceleration in
order to come to a stop at the stop line before the traffic light. In this implementation, the car maintains the speed
limit, and then decelerates along a square root curve until it reaches a stop two meters behind the stop line for the
red traffic light.
> 
> ### Final Waypoints
> Once the car has loaded the base waypoints and determined if it needs to stop for a red light, it publishes the new
waypoints, with their updated velocities to the /final_waypoints node. These final waypoints include the velocity
which the car should be driving at each step. The car publishes 100 waypoints in front of itself at a rate of 50 hz.

## Reflection
> ### Future Updates
> The current implementation could be improved by taking into account other road obstacles such as traffic, animals, trash
etc. Any obstacle could be treated as a reason to stop, and the cars waypoints could be updated accordingly.  
> 
> Another improvement would be to decelerate the car along a smoother curve than a square root. For example, the spline
library used in the path-planning project would provide a smoother deceleration and a more comfortable ride for actual
passengers. It could also incorporate lane changes. The current path does not make any lane changes, since there are no
other cars on the road.
>
> It would also be good to implement a safety check in the waypoint updater. Currently, if the waypoint updater publishes
below 30 hz, the DBW node directs control back to the driver. However, it would be good for safety to have a redundant
check built into the waypoint updater. For example, if the latency of the image processing falls, the velocity of the 
vehicle should also decrease. That way we can ensure that the car is not given waypoints which would cause it to crash
into an obstacle on the road.

## Acknowledgements
> I'd like to thank the Udacity team for their teachings throughout the entire Nano Degree, including for this capstone
project. The lessons and resources made it possible for me to implement this part of the project and to understand how
it integrated into the other parts of the whole self driving car.
>
> The waypoint-updater walk through video was an excellent resource for putting together the waypoint updater. Thank you
to Steven and Aaron for providing the guidance of how to layout the waypoint updater.
>
> I'd also like to thank the other members of the project for working together. Each person handled their own portion of
the project as well as helped each other to identify and fix errors in the code throughout the project. The team was
also an excellent resource for troubleshooting bugs with setting up the environment including ROS, dataspeed DBW, and
TensorFlow to get everything working together.

# DBW Node 
> **Name:** Deepanshu Malhotra  
> **E-Mail:** deepanshumalhotra@gmail.com
> 
> - DBW node is the drive by wire node which is used to pulish throttle, brake and steering commands. This helps the car to follow the waypoints that were published in the waypoint updater node. The two main file that I implemented were the dbw_node.py and twist_controller.py. 
> 
> - In dbw_node.py it subscribes to 3 topics namely dbw_enabled, twist_cmd and current_velocity. The twist_cmd topic provides the linear and angular velocities of the car which will be further used by the controller. The dbw_enabled topic is used to provide the current state of the dbw node that whether it is being used or not. This is done because in certain scenarios manual driver may overtake controls and hence all the values need to be reset. The current_velocity is used to provide the current linear velocity of the car. This file also imports Controller class from  twist_controller.py which was used for implementing the necessary controllers.
> 
> - In twist_controller.py throttle is controlled using the pid controller, the kp(proportional), kd(derivative) and ki(integral) are set using appropriate values. Also the minimum and maximum throttle values are set to 0.0 and 0.2 respectively so that jerk is avoided in the car. The velocity of the car is passed through the low pass filter which removes any kind of noise in it. The steering of the car is calculated using the yaw controller in which linear velocity, angular velocity and current velocity are passed as parameters. After that a low pass filter is also applied on the steering in order to smooth the steering angle so that high gradient turns are avoided. 
> 
> - Since the car has an automatic transmission so a breaking torque is applied which is about 400nm in order to prevent the car from moving. The vehicle is decelerated by multiplying deceleration limit, vehicle mass and wheel radius. Finally the command from dbw_node.py are pulished to throttle_cmd, brake_cmd and steering_cmd topic. They are published at a rate of 50 hz because the dbw system on carla expects messages at this frequency and it will disengage if messages are published at a frequency lesser than 10 hz. Also the pid controller needs to be reset if the dbw_enabled is false as the errors will pile up if its not reset which will result in incorrect values being calculated. 
> 
## Further Improvements
> 
> - MPC controller can be implemented in place of a pid controller which gradually reduces the throttle values and hence prevents the car from osscillating around the waypoints. 
> 
> - Pid controller can be applied for braking also which will make the model more robust to errors and deviations. 
> 
> - Yaw controller and low pass filters can be modified to provide more accurate results. 
> 
> - The car currently follows the waypoints only if it exceeds the certain error limit in terms of displacement and steering angle from the waypoints. It (pure_pursuit_core.cpp) can be modified to always follow the waypoints for better results. The model in current state is also competent enough to drive the whole lap without generating any errors. 

# Traffic Light Detection and Classification for Simulator
> **Name:** Muhamed Kuric  
> **Mail:** mumymania@gmail.com   

# Traffic Light Detection Node
> 
> * This part followed mostly the Detection Walkthrough by Stephen and Aaron. 
> 
> * This node takes in data from the `/image_color`, `/current_pose`, and `/base_waypoints` topics and publishes the locations to stop for red traffic lights to the `/traffic_waypoint` topic. 
> 
> ![Detector](writeup_imgs/tl-detector-ros-graph.png)
> 
> * The topic `/current_pose` receives information about the current pose of the vehicle (i.e., localization data), while the topic `/base_waypoints` receives the waypoints that the vehicle should follow (\i.e., planning). The `/image_color` topic receives the raw image from the front-looking camera sensor mounted to the vehicle. The world locations of all traffic lights in the map are also loaded from a configuration file into a lookup table. 
> 
> * Based on the current pose of the vehicle and the traffic light lookup, we find if a traffic light is visible in front of the vehicle. If so, then we run a classifier that classifies the state of the traffic light (i.e., red vs non-red). If the traffic light was red we send the waypoint index of where this traffic light is located on the path via the `/traffic_waypoint` output topic. Otherwise, we publish a default message which means that there is no red traffic light in front of us, and that the vehicle can continue following its path.
> 
## Traffic Light Classification for Simulator
> 
> * In order to train the classifier, we have logged the images from the `/image_color` during a successful automated simulation drive that used groundtruth traffic light state data that is also provided by the simulator. These training images can be found at [training-data](ros/src/tl_detector/sim-training-data/).
> 
> * The simulator ensures perfect repeatability of the input images, thus our strategy was to train a simple CNN-based classifier in Keras that overfits to these images. Training and prediction test scripts can be found at [learning](ros/src/tl_detector/train.py) [prediction](ros/src/tl_detector/test_prediction.py).
> 
> * The following NN architecture was used with full size camera images as input:
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

# Traffic Light Classification  (Real World) 

> **Name:** Sven Eckelmann  
> **Mail:** eckelmann.sven@gmail.com 
> 
> 
> One capstone of the final project is the classification of traffic light signal, provided from the udacity team. 
The following steps describe how to extract the images, the labeling process and the subsequent training of the data. I studied different approaches for the given task and I ended with 2 existing models, the 
*ssd_mobilenet_v1_coco*  and the  *faster_rcnn_resnet101_coco*.
> 
> 
> * Download bag file for the traffic light classification from udacity repositry 
> * Extract Images use:<br>
> 
``
<launch>
  <node pkg="rosbag" type="play" name="rosbag" required="true" args="-d 2 /path/to/bag_file/traffic_light_training.bag"/>  
  <node name="extract" pkg="image_view" type="image_saver" respawn="false" required="true" output="screen" cwd="ROS_HOME">  
    <remap from="image" to="image_raw"/>  
  </node>
</launch>
``	
> 
> * Copy images from ~/.ros folder to destination 
> * Label images use [labelImg](https://github.com/tzutalin/labelImg)  
> * Convert labels to csv and split (train and test set) use<br> 
> `tl_classifier/gen_csv_from_pascal.py`<br>
I created 958 labels (377 green, 327 red and 254 yellow).  
The  train set  includes 764  and the test set 194 traffic lights.
> * Convert datas to TFrecord format <br>
> ` python3 tl_classifier/gen_tfrecord.py --csv_input=tl_train.csv --output_path=train.record`
and
`python3 tl_classifier/gen_tfrecord.py --csv_input=tl_test.csv --output_path=test.record`.
 <br>Finally created `train.record` and the `test.record` file.
> 
> * Copy the train.record and test.record to `tl_data` folder 
> 
> This [good approach](https://becominghuman.ai/tensorflow-object-detection-api-tutorial-training-and-evaluating-custom-object-detector-ed2594afcf73)
describes how to set up the pipeline. 
> 
> For the training I decided to use the Amazon Workspace (AWS)
##Setup AWS 
> Use the AWS template from the udacity carnd project. So most libraries are already inculded. 
Regarding [Weifen](https://github.com/weifen/CarND-Capstone-System-Integration),  there is missing the `libcudnn.so.6`. 
<br>To fix this problem:
> * download `https://developer.nvidia.com/compute/machine-learning/cudnn/secure/v6/prod/8.0_20170307/cudnn-8.0-linux-x64-v6.0-tgz `
> * Extract compressed file <br>
>  ` tar -xzvf cudnn-8.0-linux-x64-v6.0.tgz`
> * Copy files to /usr/local/cuda/...<br>
> ```
>  sudo cp cuda/include/*.h /usr/local/cuda/include/ 
>  sudo cp cuda/lib64/libcudnn.so.6.0.21 /usr/local/cuda/lib64/ 
> ```
  
* Create Softlinks
> ```
> sudo ln -s /usr/local/cuda/lib64/libcudnn.so.6.0.21 /usr/local/cuda/lib64/libcudnn.so.6
> sudo ln -s /usr/local/cuda/lib64/libcudnn.so.6 /usr/local/cuda/lib64/libcudnn.so
> ```
> Keep on working with the setup from [Alex-Lechner](https://github.com/alex-lechner/Traffic-Light-Classification) 
> 
> * Install tensorflow 1.4 `pip install tensorflow==1.4`
> * Install packages `sudo apt-get install protobuf-compiler python-pil python-lxml python-tk`
> * create directory  `~/tensorflow` , change to tensorflow directory and clone
> `git clone https://github.com/tensorflow/models.git`
> * checkout `git checkout f7e99c0`
> * go to the ` research ` folder and execute 
```
protoc object_detection/protos/*.proto --python_out=.
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
```
> * Test your implementation 
> `python object_detection/builders/model_builder_test.py`

### Training 
> **All next steps have to be done in the ` object_detection/ ` folder !!**
> 1.  Download a trained model from [tensorflow zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md)
> 2. Create a `training/`folder and download the matching config file for your choosen model from [tensorflow zoo config](https://github.com/tensorflow/models/tree/master/research/object_detection/samples/configs) or copy the config from the `object_detection/samples/configs`
> 3. Adjust the config file - Set the paths, image size, batch size. The following setup worked for my `ssd_inception_v2_coco.config`:
```
num_classes: 4


Iimage_resizer {
      fixed_shape_resizer {
        height: 200
        width: 200
      }
    }
train_config: {
  batch_size: 16
  .....
  }
fine_tune_checkpoint: "ssd_inception_v2_coco_2018_01_28/model.ckpt"
num_steps: 10000

train_input_reader: {
  tf_record_input_reader {
    input_path: "tl_data/train.record"
  }
  label_map_path: "tl_data/tl_label.pbtxt"
}

eval_config: {
  num_examples: 40
  # Note: The below line limits the evaluation process to 10 evaluations.
  # Remove the below line to evaluate indefinitely.
  max_evals: 10
}

eval_input_reader: {
  tf_record_input_reader {
    input_path: "tl_data/test.record"
  }
  label_map_path: "tl_data/tl_label.pbtxt"
  shuffle: false
  num_readers: 1
  num_epochs: 1
}
```
> 3. Copy your local (host) `tl_data/`folder  to your remote aws
`scp tl_data carnd@ip:/tensorflow/models/research/object_detection/`
> 
> 4. Train the model 
`python train.py --logtostderr --train_dir= training --pipeline_config_path= training/your_tensorflow_model.config
`
> 
> The training tooks about 6 hours on the AWS. 
> 
> 5. Freece the model to create inference graph
```
python export_inference_graph.py --input_type image_tensor --pipeline_config_path training/your_tensorflow_model.config --trained_checkpoint_prefix training/model.ckpt-20000 --output_directory models
```
> This creates the `frozen_inference_graph`
> 
> 
### Test the model  
> The test is done with the `eval_classification.ipynb` notebook. <br>
Make sure  that  you add the `research` and the `research/slim` folder to your python path 
```
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
```
> 
> The final output for the *ssd_mobilenet_v1_coco*  looks as follow
> 
> ![](tl_classifier/results/ssd_test_images/left0118.jpg) 
> ![](tl_classifier/results/ssd_test_images/left1108.jpg) 
> ![](tl_classifier/results/ssd_test_images/left0242.jpg) 
> 
> The average time consumtion is 80 ms (without gpu support) on the jupyter notebook. I finally tested the classification in our ROS implementation. It took between 0.4 -0.5 seconds to classify the image.
> 
> The final output for the *faster_rcnn_resnet101_coco* looks as follow
> 
> ![](tl_classifier/results/rcnn_test_images/left0118.jpg) 
> ![](tl_classifier/results/rcnn_test_images/left1108.jpg) 
> ![](tl_classifier/results/rcnn_test_images/left0242.jpg) 
> 
> The average time consumtion is between 5-6s (without gpu support)
> 
> For our approach we decided to go with the ssd_mobilenet_v1_coco model. This won't work for more general images, but it shows a good performance on the provided bag file. 
