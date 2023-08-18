#include <camera_info_manager/camera_info_manager.hpp>
#include <thread>

#include "CameraApi.h"
#include "image_transport/image_transport.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/camera_info.hpp"
#include "sensor_msgs/msg/image.hpp"

namespace camera {

class MindVisualCamera : public rclcpp::Node {
public:
    MindVisualCamera(const rclcpp::NodeOptions& options) : Node("mind_visual_camera", options), fail_conut_(0) {
        RCLCPP_INFO(this->get_logger(), "<启动节点> 相机");
        // ====================================相机初始化=======================================
        CameraSdkInit(1);
        do {
            iStatus = CameraEnumerateDevice(&tCameraEnumList, &iCameraCounts);
            if (iCameraCounts == 0) {
                RCLCPP_WARN(this->get_logger(), "No device connect, return : %d", iStatus);
                ++fail_conut_;
            }
            if (fail_conut_ == 20) {
                RCLCPP_FATAL(this->get_logger(), "fail to connect camera, exit");
                rclcpp::shutdown();
            }
        } while (iCameraCounts == 0);  //没有连接设备

        fail_conut_ = 0;

        //相机初始化。初始化成功后，才能调用任何其他相机相关的操作接口
        do {
            iStatus = CameraInit(&tCameraEnumList, -1, -1, &hCamera);
            if (iStatus != CAMERA_STATUS_SUCCESS) {
                RCLCPP_WARN(this->get_logger(), "Device Init Fail, return : %d", iStatus);
                ++fail_conut_;
            }
            if (fail_conut_ == 20) {
                RCLCPP_FATAL(this->get_logger(), "fail to init camera, exit");
                rclcpp::shutdown();
            }
        } while (iStatus != CAMERA_STATUS_SUCCESS);

        //获得相机的特性描述结构体。该结构体中包含了相机可设置的各种参数的范围信息。决定了相关函数的参数
        CameraGetCapability(hCamera, &tCapability);

        // 直接使用vector的内存作为相机输出buffer
        image_msg_.data.reserve(tCapability.sResolutionRange.iHeightMax * tCapability.sResolutionRange.iWidthMax * 3);

        // 设置手动曝光
        CameraSetAeState(hCamera, false);

        CameraPlay(hCamera);

        if (tCapability.sIspCapacity.bMonoSensor) {
            channel = 1;
            CameraSetIspOutFormat(hCamera, CAMERA_MEDIA_TYPE_MONO8);
        } else {
            channel = 3;
            CameraSetIspOutFormat(hCamera, CAMERA_MEDIA_TYPE_BGR8);
        }

        // ===================================================相机初始化结束====================================================

        // Load camera info
        camera_name_ = this->declare_parameter("camera_name", "mv_camera");
        camera_info_manager_ = std::make_unique<camera_info_manager::CameraInfoManager>(this, camera_name_);
        auto camera_info_url = this->declare_parameter("camera_info_url", "package://mindvisual_camera/config/camera_info.yaml");
        if (camera_info_manager_->validateURL(camera_info_url)) {
            camera_info_manager_->loadCameraInfo(camera_info_url);
            camera_info_msg_ = camera_info_manager_->getCameraInfo();
        } else {
            RCLCPP_WARN(this->get_logger(), "Invalid camera info URL: %s", camera_info_url.c_str());
        }

        read_image = std::thread([this]() -> void {
            RCLCPP_INFO(this->get_logger(), "Publishing image!");

            image_msg_.header.frame_id = "camera_optical_frame";
            image_msg_.encoding = "rgb8";

            this->camera_pub_ = image_transport::create_camera_publisher(this, "image");

            while (rclcpp::ok()) {
                int status = CameraGetImageBuffer(hCamera, &sFrameInfo, &pRgbBuffer, 1000);
                if (status == CAMERA_STATUS_SUCCESS) {
                    CameraImageProcess(hCamera, pRgbBuffer, image_msg_.data.data(), &sFrameInfo);

                    // 翻转图像
                    // CameraFlipFrameBuffer(image_msg_.data.data(), &sFrameInfo, 3);

                    camera_info_msg_.header.stamp = image_msg_.header.stamp = this->now();
                    image_msg_.height = sFrameInfo.iHeight;
                    image_msg_.width = sFrameInfo.iWidth;
                    image_msg_.step = sFrameInfo.iWidth * 3;
                    image_msg_.data.resize(sFrameInfo.iWidth * sFrameInfo.iHeight * 3);

                    camera_pub_.publish(image_msg_, camera_info_msg_);

                    // 在成功调用CameraGetImageBuffer后，必须调用CameraReleaseImageBuffer来释放获得的buffer。
                    // 否则再次调用CameraGetImageBuffer时，程序将被挂起一直阻塞，
                    // 直到其他线程中调用CameraReleaseImageBuffer来释放了buffer
                    CameraReleaseImageBuffer(hCamera, pRgbBuffer);
                    fail_conut_ = 0;
                } else {
                    RCLCPP_WARN(this->get_logger(), "Failed to get image buffer, status = %d", status);
                    fail_conut_++;
                }

                if (fail_conut_ > 20) {
                    RCLCPP_FATAL(this->get_logger(), "Failed to get image buffer, exit!");
                    break;
                }
            }
        });

        this->parameter_callback = this->add_on_set_parameters_callback(std::bind(&MindVisualCamera::parametersCallback, this, std::placeholders::_1));
    }

    ~MindVisualCamera() {
        if (read_image.joinable()) {
            read_image.join();
        }
        RCLCPP_INFO(this->get_logger(), "Camera Node Shutdown");
        CameraUnInit(hCamera);
        rclcpp::shutdown();
    }

private:
    void declareParamers() {
        rcl_interfaces::msg::ParameterDescriptor param_desc;
        param_desc.integer_range.resize(1);
        param_desc.integer_range[0].step = 1;

        // Exposure time
        param_desc.description = "Exposure time in microseconds";
        // 对于CMOS传感器，其曝光的单位是按照行来计算的
        double exposure_line_time;
        CameraGetExposureLineTime(hCamera, &exposure_line_time);
        param_desc.integer_range[0].from_value = tCapability.sExposeDesc.uiExposeTimeMin * exposure_line_time;
        param_desc.integer_range[0].to_value = tCapability.sExposeDesc.uiExposeTimeMax * exposure_line_time;
        double exposure_time = this->declare_parameter("exposure_time", 5000, param_desc);
        CameraSetExposureTime(hCamera, exposure_time);
        RCLCPP_INFO(this->get_logger(), "Exposure time = %f", exposure_time);

        // Analog gain
        param_desc.description = "Analog gain";
        param_desc.integer_range[0].from_value = tCapability.sExposeDesc.uiAnalogGainMin;
        param_desc.integer_range[0].to_value = tCapability.sExposeDesc.uiAnalogGainMax;
        int analog_gain;
        CameraGetAnalogGain(hCamera, &analog_gain);
        analog_gain = this->declare_parameter("analog_gain", analog_gain, param_desc);
        CameraSetAnalogGain(hCamera, analog_gain);
        RCLCPP_INFO(this->get_logger(), "Analog gain = %d", analog_gain);

        // RGB Gain
        // Get default value
        CameraGetGain(hCamera, &r_gain_, &g_gain_, &b_gain_);
        // R Gain
        param_desc.integer_range[0].from_value = tCapability.sRgbGainRange.iRGainMin;
        param_desc.integer_range[0].to_value = tCapability.sRgbGainRange.iRGainMax;
        r_gain_ = this->declare_parameter("rgb_gain.r", r_gain_, param_desc);
        // G Gain
        param_desc.integer_range[0].from_value = tCapability.sRgbGainRange.iGGainMin;
        param_desc.integer_range[0].to_value = tCapability.sRgbGainRange.iGGainMax;
        g_gain_ = this->declare_parameter("rgb_gain.g", g_gain_, param_desc);
        // B Gain
        param_desc.integer_range[0].from_value = tCapability.sRgbGainRange.iBGainMin;
        param_desc.integer_range[0].to_value = tCapability.sRgbGainRange.iBGainMax;
        b_gain_ = this->declare_parameter("rgb_gain.b", b_gain_, param_desc);
        // Set gain
        CameraSetGain(hCamera, r_gain_, g_gain_, b_gain_);
        RCLCPP_INFO(this->get_logger(), "RGB Gain: R = %d", r_gain_);
        RCLCPP_INFO(this->get_logger(), "RGB Gain: G = %d", g_gain_);
        RCLCPP_INFO(this->get_logger(), "RGB Gain: B = %d", b_gain_);

        // Saturation
        param_desc.description = "Saturation";
        param_desc.integer_range[0].from_value = tCapability.sSaturationRange.iMin;
        param_desc.integer_range[0].to_value = tCapability.sSaturationRange.iMax;
        int saturation;
        CameraGetSaturation(hCamera, &saturation);
        saturation = this->declare_parameter("saturation", saturation, param_desc);
        CameraSetSaturation(hCamera, saturation);
        RCLCPP_INFO(this->get_logger(), "Saturation = %d", saturation);

        // Gamma
        param_desc.integer_range[0].from_value = tCapability.sGammaRange.iMin;
        param_desc.integer_range[0].to_value = tCapability.sGammaRange.iMax;
        int gamma;
        CameraGetGamma(hCamera, &gamma);
        gamma = this->declare_parameter("gamma", gamma, param_desc);
        CameraSetGamma(hCamera, gamma);
        RCLCPP_INFO(this->get_logger(), "Gamma = %d", gamma);
    }

    rcl_interfaces::msg::SetParametersResult parametersCallback(const std::vector<rclcpp::Parameter>& parameters) {
        rcl_interfaces::msg::SetParametersResult result;
        result.successful = true;
        for (const auto& param : parameters) {
            if (param.get_name() == "exposure_time") {
                int status = CameraSetExposureTime(hCamera, param.as_int());
                if (status != CAMERA_STATUS_SUCCESS) {
                    result.successful = false;
                    result.reason = "Failed to set exposure time, status = " + std::to_string(status);
                }
            } else if (param.get_name() == "analog_gain") {
                int status = CameraSetAnalogGain(hCamera, param.as_int());
                if (status != CAMERA_STATUS_SUCCESS) {
                    result.successful = false;
                    result.reason = "Failed to set analog gain, status = " + std::to_string(status);
                }
            } else if (param.get_name() == "rgb_gain.r") {
                r_gain_ = param.as_int();
                int status = CameraSetGain(hCamera, r_gain_, g_gain_, b_gain_);
                if (status != CAMERA_STATUS_SUCCESS) {
                    result.successful = false;
                    result.reason = "Failed to set RGB gain, status = " + std::to_string(status);
                }
            } else if (param.get_name() == "rgb_gain.g") {
                g_gain_ = param.as_int();
                int status = CameraSetGain(hCamera, r_gain_, g_gain_, b_gain_);
                if (status != CAMERA_STATUS_SUCCESS) {
                    result.successful = false;
                    result.reason = "Failed to set RGB gain, status = " + std::to_string(status);
                }
            } else if (param.get_name() == "rgb_gain.b") {
                b_gain_ = param.as_int();
                int status = CameraSetGain(hCamera, r_gain_, g_gain_, b_gain_);
                if (status != CAMERA_STATUS_SUCCESS) {
                    result.successful = false;
                    result.reason = "Failed to set RGB gain, status = " + std::to_string(status);
                }
            } else if (param.get_name() == "saturation") {
                int status = CameraSetSaturation(hCamera, param.as_int());
                if (status != CAMERA_STATUS_SUCCESS) {
                    result.successful = false;
                    result.reason = "Failed to set saturation, status = " + std::to_string(status);
                }
            } else if (param.get_name() == "gamma") {
                int gamma = param.as_int();
                int status = CameraSetGamma(hCamera, gamma);
                if (status != CAMERA_STATUS_SUCCESS) {
                    result.successful = false;
                    result.reason = "Failed to set Gamma, status = " + std::to_string(status);
                }
            } else {
                result.successful = false;
                result.reason = "Unknown parameter: " + param.get_name();
            }
        }
        return result;
    }

    unsigned char* pRgbBuffer;  //处理后数据缓存区
    int iCameraCounts = 1;
    int iStatus = -1;
    tSdkCameraDevInfo tCameraEnumList;
    int hCamera;
    tSdkCameraCapbility tCapability;  //设备描述信息
    tSdkFrameHead sFrameInfo;
    BYTE* pbyBuffer;
    int iDisplayFrames = 10000;
    int channel = 3;

    int fail_conut_;
    std::string camera_name_;
    int r_gain_, g_gain_, b_gain_;

    std::unique_ptr<camera_info_manager::CameraInfoManager> camera_info_manager_;

    sensor_msgs::msg::Image image_msg_;             // 原始图像信息
    sensor_msgs::msg::CameraInfo camera_info_msg_;  // 相机信息，包含内参、畸变矩阵等

    image_transport::CameraPublisher camera_pub_;  // 图像发布

    OnSetParametersCallbackHandle::SharedPtr parameter_callback;

    std::thread read_image;
};

}  // namespace camera

#include "rclcpp_components/register_node_macro.hpp"

// Register the component with class_loader.
// This acts as a sort of entry point, allowing the component to be discoverable
// when its library is being loaded into a running process.
RCLCPP_COMPONENTS_REGISTER_NODE(camera::MindVisualCamera)
