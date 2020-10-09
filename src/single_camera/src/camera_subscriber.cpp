#include "rclcpp/rclcpp.hpp"
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/highgui/highgui.hpp>


using std::placeholders::_1;

class CameraSubscriberNode : public rclcpp::Node{
public:
    CameraSubscriberNode() : Node("SingleCameraSubscriberNode"){
         image_sub = this->create_subscription<sensor_msgs::msg::Image>(
      "image", 10, std::bind(&CameraSubscriberNode::camera_callback, this, _1));
    }

    void camera_callback(const sensor_msgs::msg::Image::SharedPtr msg){
        cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
        image = cv_ptr->image;
        cv::imshow("Image", image);
        cv::waitKey(1);       
  }
  
private:
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub;
    cv::Mat image;
};

int main(int argc, char ** argv){
    rclcpp::init(argc, argv);
    auto node = std::make_shared<CameraSubscriberNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
}

