# run this script in your remote PC (*Ubuntu 20.04)
# update the system
sudo apt update
sudo apt upgrade -y

# download ROS Noetic installation scripts
cd ~
wget https://raw.githubusercontent.com/ROBOTIS-GIT/robotis_tools/master/install_ros_noetic.sh

# run the script file
chmod 755 ./install_ros_noetic.sh
bash ./install_ros_noetic.sh

# install ROS packages
sudo apt-get install ros-noetic-joy ros-noetic-teleop-twist-joy ros-noetic-teleop-twist-keyboard ros-noetic-laser-proc ros-noetic-rgbd-launch ros-noetic-rosserial-arduino ros-noetic-rosserial-python ros-noetic-rosserial-client ros-noetic-rosserial-msgs ros-noetic-amcl ros-noetic-map-server ros-noetic-move-base ros-noetic-urdf ros-noetic-xacro ros-noetic-compressed-image-transport ros-noetic-rqt* ros-noetic-rviz ros-noetic-gmapping ros-noetic-navigation ros-noetic-interactive-markers

# install TurtleBot3 packages
sudo apt install ros-noetic-dynamixel-sdk
sudo apt install ros-noetic-turtlebot3-msgs
sudo apt install ros-noetic-turtlebot3

# install net-tools
sudo apt install net-tools

# network configuration
ifconfig

# set ip addresses ... should be continued
