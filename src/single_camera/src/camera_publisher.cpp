#include "rclcpp/rclcpp.hpp"
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/highgui/highgui.hpp>
#include <image_transport/image_transport.hpp>
#include <camera_info_manager/camera_info_manager.hpp>

#define CAMERA_WIDTH 640
#define CAMERA_HEIGHT 480

class CameraNode : public rclcpp::Node{
public:
    CameraNode() : Node("SingleCameraNode"){
         
         int device = 0;
         cap.open(device, cv::CAP_ANY);
         if (!cap.isOpened()) {
            throw std::runtime_error("Could not open video stream!");
         }else{
            RCLCPP_INFO(this->get_logger(), "Connected to camera!");
         }
         
         camera_pub = image_transport::create_camera_publisher(this, "image", rmw_qos_profile_default);
         camera_info_manager = std::make_unique<camera_info_manager::CameraInfoManager>(this);
         
         // set width and height
         sensor_msgs::msg::CameraInfo ci;
         ci.width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
         ci.height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
         camera_info_manager->setCameraInfo(ci);
         
    }

    void loop(){
        while (rclcpp::ok()) {
          cap >> frame;
          if (frame.empty()){  
             RCLCPP_INFO(this->get_logger(), "Frame empty");
             continue; 
          }

          sensor_msgs::msg::CameraInfo::SharedPtr ci(new sensor_msgs::msg::CameraInfo(camera_info_manager->getCameraInfo()));
          ci->header.stamp = now();
          ci->header.frame_id = "camera_frame";

          sensor_msgs::msg::Image::SharedPtr msg = cv_bridge::CvImage(ci->header, "bgr8", frame).toImageMsg();

          camera_pub.publish(msg, ci);
        }        
  }
  
private:
    cv::VideoCapture cap;
    cv::Mat frame;
    image_transport::CameraPublisher camera_pub;
    std::unique_ptr<camera_info_manager::CameraInfoManager> camera_info_manager;
};

int main(int argc, char ** argv){
    rclcpp::init(argc, argv);
    auto node = std::make_shared<CameraNode>();
    node->loop();
    rclcpp::spin(node);
    rclcpp::shutdown();
}

