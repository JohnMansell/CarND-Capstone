# Udacity Self Driving Car Nano Degree
###  Capstone Project
### Waypoint Updater
### John Mansell -- JohnMansellis@gmail.com

---
# Introduction
> My part of the project focused on finishing up the waypoint updater, which had been started by one of the other
students. The function of the waypoint updater is to publish the waypoints in front of the car, including their
associated velocity vector. This data is then used by the car dbw node to drive the car along the highway, ensuring that
the car drives at the appropriate speed, and obeys traffic light rules.

# Procedure
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

# Reflection
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

# Acknowledgements
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