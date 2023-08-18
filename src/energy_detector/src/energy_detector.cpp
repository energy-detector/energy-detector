#include "cv_bridge/cv_bridge.h"
#include "energy_detector/energy_search.h"
#include "image_transport/image_transport.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "rclcpp/qos.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "serial_interface/msg/pc_statu.hpp"
#include "target_msg/msg/targets.hpp"

using namespace std::placeholders;

namespace energy_detector {

class EnergyDetector : public rclcpp::Node {
public:
    EnergyDetector(const rclcpp::NodeOptions& options) : Node("energy_detector", options) {
        RCLCPP_INFO(this->get_logger(), "<启动节点> 能量机关追踪器");
        this->camera_received_ = false;
        this->energy_search_ptr_ = std::make_unique<EnergySearch>();

        this->ros_image_object_detect_image_pub_ = this->create_publisher<sensor_msgs::msg::Image>("energy_debug_image", 10);

        // 订阅相机参数的回调函数（Lambda 写法）
        auto camera_info_callback = [this](const sensor_msgs::msg::CameraInfo::SharedPtr _camera_info) -> void {
            if (!rclcpp::ok()) {
                RCLCPP_INFO_ONCE(this->get_logger(), "关闭订阅 %s", __PRETTY_FUNCTION__);
                return;
            }
            this->camera_info_ = *_camera_info;
            this->camera_received_ = true;
            RCLCPP_INFO_ONCE(this->get_logger(), "已获取相机参数");
        };
        this->camera_info_sub_ =
            this->create_subscription<sensor_msgs::msg::CameraInfo>("/camera_node/camera_info", 10, camera_info_callback);

        auto ssi_call_back = [this](const serial_interface::msg::SelfInfo::SharedPtr _ssi) -> void { this->ssi_ = *_ssi; };
        this->ssi_sub_ = this->create_subscription<serial_interface::msg::SelfInfo>("/serial/ssi", 10, ssi_call_back);

        this->ros_image_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            "/camera_node/image_raw", 10, std::bind(&EnergyDetector::rosImageSubCallBack, this, _1));
    }

private:
    void rosImageSubCallBack(const sensor_msgs::msg::Image::SharedPtr _ros_image) const {
        if (!rclcpp::ok()) {
            RCLCPP_INFO_ONCE(this->get_logger(), "关闭订阅 %s", __PRETTY_FUNCTION__);
            return;
        }
        if (!this->camera_received_) {
            RCLCPP_WARN_ONCE(this->get_logger(), "未获得相机参数");
            return;
        }
        auto cv_ptr = cv_bridge::toCvCopy(_ros_image, sensor_msgs::image_encodings::BGR8);
        if (!cv_ptr->image.empty()) {
            target_msg::msg::Targets targets;
            this->energy_search_ptr_->leafDetailedSearch(cv_ptr->image, targets, this->ssi_.robot_color);
            // cv_bridge::CvImage cv_image;
            // cv_image.image = this->energy_search_ptr_->object_detect_image;
            // cv_image.header.frame_id = "energy_debug_img";
            // cv_image.header.stamp = targets.header.stamp = this->now();
            // cv_image.encoding = sensor_msgs::image_encodings::BGR8;

            // this->ros_image_object_detect_image_pub_->publish(*cv_image.toImageMsg());

            cv::imshow("source", cv_ptr->image);
            cv::imshow("debug", this->energy_search_ptr_->debug_image);
            cv::imshow("object", this->energy_search_ptr_->object_detect_image);
            cv::waitKey(60);
        }
    }

    sensor_msgs::msg::CameraInfo camera_info_;
    serial_interface::msg::PcStatu pc_status_;

    bool camera_received_;
    serial_interface::msg::SelfInfo ssi_;

    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr ros_image_object_detect_image_pub_;

    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr ros_image_sub_;
    rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr camera_info_sub_;
    rclcpp::Subscription<serial_interface::msg::SelfInfo>::SharedPtr ssi_sub_;
    std::unique_ptr<EnergySearch> energy_search_ptr_;
};
}  // namespace energy_detector

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(energy_detector::EnergyDetector)
