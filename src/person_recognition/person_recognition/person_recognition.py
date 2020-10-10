import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class MyNode(Node):
    def __init__(self):
         super().__init__('PersonRecognition')
         self.subscription = self.create_subscription(
            Image, 'image', self.listener_callback, 10)

    def listener_callback(self, msg):
         bridge = CvBridge()
 
         cv_image = bridge.imgmsg_to_cv2(msg, 'bgr8')
         
         cv2.imshow("Image window", cv_image)
         cv2.waitKey(1)
        
def main(args=None):
    rclpy.init(args=args)
    node = MyNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()

