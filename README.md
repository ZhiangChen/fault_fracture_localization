
# fault_fracture_localization

A UAV based solution to earthquake damage zone estimations. 

## Supported Versions
This project is built and tested on ROS 2 Humble, PX4 1.14, Ubuntu Jammy, and JetPack 6.0. Other configurations have not been tested and compatibility is unknown.




## Build 

Install requirements
```
pip install -r requirements.txt
```

Source the ROS 2 distribution 
```
cd ${YOUR_WORKSPACE_PATH}
git clone https://github.com/ZhiangChen/fault_fracture_localization.git
colcon build
```


## Setup for Simulation Testing

First, make sure that ROS, Gazebo, PX4, MicroXRCEAgent, and QGroundcontrol are installed your system. You will also need to install ros_gz_bridge, and depending on yoru version of Gazebo you may have to clone from the repository and build from scratch.

Go to the PX4 folder and run ```make px4_sitl gz_x500 ```
to boot up the Gazebo simulation and PX4. Then, open QGroundControl to find the ground station. Boot up MicroXRCEAgent to allow topic interface with ```MicroXRCEAgent udp4 -p 8888```. 
Go to the ```ros_gz_bridge``` directory, source and run the node using ```ros2 run ros_gz_image image_bridge /camera ```. Finally, go into the fault fracture localization directory and run in with ``` ros2 launch fault_fracture_localization launch.py ```

## Overview of Project Files
There are three main nodes in this repository -

```perception.py``` - Handles everything to do with camera inputs

```state_machine.py``` - Decides what the UAV should be doing at any time given inputs from all another nodes

```offboard_control.py``` - Given waypoints, will navigate the UAV to those points



