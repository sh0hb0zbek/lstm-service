# run this script code in turtlebot system
# update the system
# password is 'turtlebot'
sudo apt update

# install LDS-02 driver
sudo apt install libudev-dev
cd ~/catkin_ws/src
git clone -b develop https://github.com/ROBOTIS-GIT/ld08_driver.git
cd ~/catkin_ws/src/turtlebot3
git pull
rm -r turtlebot3_description/ turtlebot3_teleop/ turtlebot3_navigation/ turtlebot3_slam/ turtlebot3_example/
cd ~/catkin_ws
catkin_make

# export the LDS model to the .bashrc file
echo "export LDS_MODEL=LDS-02" >> ~/.bashrc
