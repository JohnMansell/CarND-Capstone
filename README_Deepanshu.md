# DBW Node 
##### Team Member - Deepanshu Malhotra
##### Email - deepanshumalhotra@gmail.com

- DBW node is the drive by wire node which is used to pulish throttle, brake and steering commands. This helps the car to follow the waypoints that were published in the waypoint updater node. The two main file that I implemented were the dbw_node.py and twist_controller.py. 

- In dbw_node.py it subscribes to 3 topics namely dbw_enabled, twist_cmd and current_velocity. The twist_cmd topic provides the linear and angular velocities of the car which will be further used by the controller. The dbw_enabled topic is used to provide the current state of the dbw node that whether it is being used or not. This is done because in certain scenarios manual driver may overtake controls and hence all the values need to be reset. The current_velocity is used to provide the current linear velocity of the car. This file also imports Controller class from  twist_controller.py which was used for implementing the necessary controllers.

- In twist_controller.py throttle is controlled using the pid controller, the kp(proportional), kd(derivative) and ki(integral) are set using appropriate values. Also the minimum and maximum throttle values are set to 0.0 and 0.2 respectively so that jerk is avoided in the car. The velocity of the car is passed through the low pass filter which removes any kind of noise in it. The steering of the car is calculated using the yaw controller in which linear velocity, angular velocity and current velocity are passed as parameters. After that a low pass filter is also applied on the steering in order to smooth the steering angle so that high gradient turns are avoided. 

- Since the car has an automatic transmission so a breaking torque is applied which is about 400nm in order to prevent the car from moving. The vehicle is decelerated by multiplying deceleration limit, vehicle mass and wheel radius. Finally the command from dbw_node.py are pulished to throttle_cmd, brake_cmd and steering_cmd topic. They are published at a rate of 50 hz because the dbw system on carla expects messages at this frequency and it will disengage if messages are published at a frequency lesser than 10 hz. Also the pid controller needs to be reset if the dbw_enabled is false as the errors will pile up if its not reset which will result in incorrect values being calculated. 

## Further Improvements

- MPC controller can be implemented in place of a pid controller which gradually reduces the throttle values and hence prevents the car from osscillating around the waypoints. 

- Pid controller can be applied for braking also which will make the model more robust to errors and deviations. 

- Yaw controller and low pass filters can be modified to provide more accurate results. 

- The car currently follows the waypoints only if it exceeds the certain error limit in terms of displacement and steering angle from the waypoints. It (pure_pursuit_core.cpp) can be modified to always follow the waypoints for better results. The model in current state is also competent enough to drive the whole lap without generating any errors. 
