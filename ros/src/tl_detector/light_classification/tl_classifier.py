from styx_msgs.msg import TrafficLight
import tensorflow as tf
import numpy as np
import rospy
from keras.models import load_model, model_from_json
from keras.utils.generic_utils import get_custom_objects
from PIL import Image
import cv2

graph = tf.get_default_graph()

# -------------------------------------
#       Traffic Light -- Classifier
# -------------------------------------
class TLClassifier(object):

    # -----------------------------------
    #   Traffic Light Classes:
    #   Source -- tl_data/tl_label.pbtxt
    # -------------------------------------
    #       1 = red
    #       2 = yellow
    #       3 = green
    #       4 = None
    # -------------------------


    # -------------------------------
    #       Init
    # -------------------------------
    def __init__(self,_config):

        # Test Site vs Simulator
        is_site = _config['is_site']
        self.is_site = is_site
        path = None

        # Test Site
        if is_site:
            rospy.loginfo('REAL : Loading site classifier')
            path = 'light_classification/graph/frozen_inference_graphssd_mobilenet_v1_coco.pb'
            self.graph              = tf.Graph()
            self.sess               = None
            self.image_tensor       = None
            self.detection_boxes    = None
            self.detection_scores   = None
            self.detection_classes  = None
            self.num_detections     = None

            self.load_graph(path)
            self.init_tensorflow_session()

        # Simulator
        else:
            rospy.loginfo('Simulation : Loading keras classifier')
            path = 'light_classification/graph/model.h5'
            self.model = load_model(path)
            self.graph = tf.get_default_graph() # workaround https://github.com/keras-team/keras/issues/2397

        # TODO load classifier

    # -------------------------------
    #       Load Graph
    # -------------------------------
    def load_graph(self, path):

        with self.graph.as_default():
            od_graph_def = tf.GraphDef()

            with tf.gfile.GFile(path, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

    # -------------------------------
    #       Init - Tensorflow
    # -------------------------------
    def init_tensorflow_session(self):

        # Config
        config = tf.ConfigProto(log_device_placement=False)
        # config.gpu_options.allow_growth = True
        # config.gpu_options.per_process_gpu_memory_fraction = 0.9

        # Session
        self.sess = tf.Session(graph=self.graph, config=config)

        # set graph
        self.image_tensor = self.graph.get_tensor_by_name('image_tensor:0')

        # Boxes
        #   - Each box represents a part of the image where a particular object was detected.
        self.detection_boxes = self.graph.get_tensor_by_name('detection_boxes:0')

        # Score
        #   - Each score represent how level of confidence for each of the objects.
        #   - Score is shown on the result image, together with the class label.
        self.detection_scores = self.graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = self.graph.get_tensor_by_name('detection_classes:0')
        self.num_detections = self.graph.get_tensor_by_name('num_detections:0')

    # -------------------------------
    #       Load Image to NP array
    # -------------------------------
    def load_image_into_np_array(self, image_cv, convert_to_rgb=True):
        if convert_to_rgb:
            image_cv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)

        image = Image.fromarray(image_cv)
        [im_width, im_height] = image.size


        return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)

    # -------------------------------
    #       Get Classification
    # -------------------------------
    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #TODO implement light color prediction

        print("\n\n ----------------------------- \n ----------------------------- ")
        print("\n TESTING")
        print("\n\n ----------------------------- \n ----------------------------- ")

        if self.is_site:
            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            image_np = self.load_image_into_np_array(image, True)
            image_np_expanded = np.expand_dims(image_np, axis=0)
            
            (boxes, scores, classes, num) = self.sess.run(
                [self.detection_boxes,
                 self.detection_scores,
                 self.detection_classes,
                 self.num_detections],
                feed_dict={self.image_tensor: image_np_expanded})

            light = classes[0][0]

            if light == 1:
                return TrafficLight.RED
            elif light == 2:
                return TrafficLight.YELLOW
            elif light == 3:
                return TrafficLight.GREEN
            else:
                return TrafficLight.UNKNOWN

        # Use Keras Model
        else:

            rospy.loginfo('Running Keras ...')
            with self.graph.as_default():
                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                #  Do we have to resize the image??
                #desired_dim=(32,32)
                #image = cv2.resize(image, desired_dim, interpolation=cv2.INTER_LINEAR)
                image_np = self.load_image_into_np_array(image, False) / 255.0

                # Convert Image to image_conv2d_1
                image_np_expanded = np.expand_dims(image_np, axis=0)

                predicted_state = self.model.predict_classes(image_np_expanded, batch_size=4)
                rospy.loginfo('Classified: %i', predicted_state)
                if predicted_state[0] == 0:
                    return TrafficLight.UNKNOWN
                else:
                    return TrafficLight.RED
