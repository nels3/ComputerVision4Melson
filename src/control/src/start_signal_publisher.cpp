#include <chrono>
#include <memory>
#include "rclcpp/rclcpp.hpp"
#include "custom_srv_and_msg/msg/control.hpp" 
#include <string>

using namespace std::chrono_literals;

class StartNode : public rclcpp::Node{
public:
    StartNode() : Node("StartNode"){
         this->declare_parameter<std::string>("topic", "control");
         std::string topic;
         get_parameter<std::string>("topic", topic);
         this->declare_parameter<int>("start", 1);
         get_parameter<int>("start", start_);
         
         control_pub = this->create_publisher<control::msg::Control>(topic, 10);
         
         timer_ = this->create_wall_timer(500ms, std::bind(&StartNode::timer_callback, this));
        
    }
private:
    void timer_callback(){
         auto message = control::msg::Control();                             
         message.start = start_;                                     
         control_pub->publish(message);
    }
  
     rclcpp::TimerBase::SharedPtr timer_;
    rclcpp::Publisher<control::msg::Control>::SharedPtr control_pub;
    int start_;
    
};

int main(int argc, char ** argv){
    rclcpp::init(argc, argv);
    auto node = std::make_shared<StartNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
}

