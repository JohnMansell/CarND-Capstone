# Traffic Light Classification  (Real World) 

**Name:** Sven Eckelmann
**Mail:** eckelmann.sven@gmail.com 


One capstone of the final project is the classification of traffic light signal, provided from the udacity team. 
The following steps describe how to extract the images, the labeling process and the subsequent training of the data. I studied different approaches for the given task and I ended with 2 existing models, the 
*ssd_mobilenet_v1_coco*  and the  *faster_rcnn_resnet101_coco*.


* Download bag file for the traffic light classification from udacity repositry 
* Extract Images use:<br>

``
<launch>
  <node pkg="rosbag" type="play" name="rosbag" required="true" args="-d 2 /path/to/bag_file/traffic_light_training.bag"/>
  <node name="extract" pkg="image_view" type="image_saver" respawn="false" required="true" output="screen" cwd="ROS_HOME">
    <remap from="image" to="image_raw"/>
  </node>
</launch>
``	

* Copy images from ~/.ros folder to destination 
* Label images use [labelImg](https://github.com/tzutalin/labelImg)  
* Convert labels to csv and split (train and test set) use<br> 
`tl_classifier/gen_csv_from_pascal.py`<br>
I created 958 labels (377 green, 327 red and 254 yellow).  
The  train set  includes 764  and the test set 194 traffic lights.
* Convert datas to TFrecord format <br>
` python3 tl_classifier/gen_tfrecord.py --csv_input=tl_train.csv --output_path=train.record`
and
`python3 tl_classifier/gen_tfrecord.py --csv_input=tl_test.csv --output_path=test.record`.
 <br>Finally created `train.record` and the `test.record` file.

* Copy the train.record and test.record to `tl_data` folder 

This [good approach](https://becominghuman.ai/tensorflow-object-detection-api-tutorial-training-and-evaluating-custom-object-detector-ed2594afcf73)
describes how to set up the pipeline. 

For the training I decided to use the Amazon Workspace (AWS)
##Setup AWS 
Use the AWS template from the udacity carnd project. So most libraries are already inculded. 
Regarding [Weifen](https://github.com/weifen/CarND-Capstone-System-Integration),  there is missing the `libcudnn.so.6`. 
<br>To fix this problem:
* download `https://developer.nvidia.com/compute/machine-learning/cudnn/secure/v6/prod/8.0_20170307/cudnn-8.0-linux-x64-v6.0-tgz `
* Extract compressed file <br>
 ` tar -xzvf cudnn-8.0-linux-x64-v6.0.tgz`
* Copy files to /usr/local/cuda/...<br>
```
 sudo cp cuda/include/*.h /usr/local/cuda/include/ 
 sudo cp cuda/lib64/libcudnn.so.6.0.21 /usr/local/cuda/lib64/ 
```
 
* Create Softlinks
```
sudo ln -s /usr/local/cuda/lib64/libcudnn.so.6.0.21 /usr/local/cuda/lib64/libcudnn.so.6
sudo ln -s /usr/local/cuda/lib64/libcudnn.so.6 /usr/local/cuda/lib64/libcudnn.so
```
Keep on working with the setup from [Alex-Lechner](https://github.com/alex-lechner/Traffic-Light-Classification) 

* Install tensorflow 1.4 `pip install tensorflow==1.4`
* Install packages `sudo apt-get install protobuf-compiler python-pil python-lxml python-tk`
* create directory  `~/tensorflow` , change to tensorflow directory and clone
`git clone https://github.com/tensorflow/models.git`
* checkout `git checkout f7e99c0`
* go to the ` research ` folder and execute 
```
protoc object_detection/protos/*.proto --python_out=.
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
```
* Test your implementation 
`python object_detection/builders/model_builder_test.py`

### Training 
**All next steps have to be done in the ` object_detection/ ` folder !!**
1.  Download a trained model from [tensorflow zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md)
2. Create a `training/`folder and download the matching config file for your choosen model from [tensorflow zoo config](https://github.com/tensorflow/models/tree/master/research/object_detection/samples/configs) or copy the config from the `object_detection/samples/configs`
3. Adjust the config file - Set the paths, image size, batch size. The following setup worked for my `ssd_inception_v2_coco.config`:
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
3. Copy your local (host) `tl_data/`folder  to your remote aws
`scp tl_data carnd@ip:/tensorflow/models/research/object_detection/`

4. Train the model 
`python train.py --logtostderr --train_dir= training --pipeline_config_path= training/your_tensorflow_model.config
`

The training tooks about 6 hours on the AWS. 

5. Freece the model to create inference graph
```
python export_inference_graph.py --input_type image_tensor --pipeline_config_path training/your_tensorflow_model.config --trained_checkpoint_prefix training/model.ckpt-20000 --output_directory models
```
This creates the `frozen_inference_graph`


### Test the model  
The test is done with the `eval_classification.ipynb` notebook. <br>
Make sure  that  you add the `research` and the `research/slim` folder to your python path 
```
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
```

The final output for the *ssd_mobilenet_v1_coco*  looks as follow

![](tl_classifier/results/ssd_test_images/left0118.jpg) 
![](tl_classifier/results/ssd_test_images/left1108.jpg) 
![](tl_classifier/results/ssd_test_images/left0242.jpg) 

The average time consumtion is 80 ms (without gpu support) on the jupyter notebook. I finally tested the classification in our ROS implementation. It took between 0.4 -0.5 seconds to classify the image.

The final output for the *faster_rcnn_resnet101_coco* looks as follow

![](tl_classifier/results/rcnn_test_images/left0118.jpg) 
![](tl_classifier/results/rcnn_test_images/left1108.jpg) 
![](tl_classifier/results/rcnn_test_images/left0242.jpg) 

The average time consumtion is between 5-6s (without gpu support)

For our approach we decided to go with the ssd_mobilenet_v1_coco model. This won't work for more general images, but it shows a good performance on the provided bag file. 