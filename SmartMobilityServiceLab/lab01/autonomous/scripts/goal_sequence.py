#!/usr/bin/env python3
# license removed for brevity

import rospy
import actionlib
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal

def movebase_client():

    client = actionlib.SimpleActionClient('move_base',MoveBaseAction)
    client.wait_for_server()
    # Sequence
    goal_seq=[
        [  2.0,  2.0, 0.0],
        [  2.5, -2.5, 0.0],
        [ -2.0, -2.0, 0.0],
        [ -2.0,  2.2, 0.0],
        [  0.0,  0.0, 0.0]] # at the end get back to where it was parked :)

    goal = MoveBaseGoal()
    goal.target_pose.header.frame_id = "map"
    goal.target_pose.header.stamp = rospy.Time.now()

    # Looping in sequence
    for i in range(0,5):
        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = "map"
        goal.target_pose.header.stamp = rospy.Time.now()
        goal.target_pose.pose.position.x = goal_seq[i][0]
        goal.target_pose.pose.position.y = goal_seq[i][1]
        goal.target_pose.pose.position.z = goal_seq[i][2]
        goal.target_pose.pose.orientation.w = 0.09012465928 # fixed angles
        goal.target_pose.pose.orientation.z = 0.098917586   # of robot position for all goals

        client.send_goal(goal)
        wait = client.wait_for_result()
        if not wait:
            rospy.logerr("Action server DOWN ;/ ")
        else:
            print("A Goal is Executed") 
        # looping each goal update as 
        # it end here if no loop is used
    return 1
if __name__ == '__main__':
    try:
        rospy.init_node('movebaseClient')
        result = movebase_client()
        if result:
            rospy.loginfo("All Goals executed ")
    except rospy.ROSInterruptException:
        rospy.loginfo("Navigation DONE ")