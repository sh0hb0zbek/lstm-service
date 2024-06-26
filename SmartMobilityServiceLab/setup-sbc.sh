# run this script in your remote PC (Ubuntu 20.04)
# install GParted GUI tool and resice the Partition
sudo apt install gparted
gparted

# update the system
sudo apt update

# network configuration for turtlebot3 system
# refer to the lecture note
cd /media/$USER/writable/etc/netplan
sudo nano 50-cloud-init.yaml
