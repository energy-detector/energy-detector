#ifndef ENERGY_SEARCH_H__
#define ENERGY_SEARCH_H__

#include <memory>
#include <string>
#include <vector>

#include "serial_interface/msg/self_info.hpp"
#include "energy_detector/nanodet_openvino.h"
#include "target_msg/msg/targets.hpp"

namespace energy_detector {

using namespace serial_interface::msg;

class EnergySearch {
public:
    EnergySearch();

    cv::Mat debug_image;
    cv::Mat object_detect_image;

    /**
     * @brief 详细搜索叶片
     *
     * @param _frame 输入一帧图片
     * @param _output 输出一组详细的目标数据
     * @param _enemy_color 敌方颜色，具体定义查看serial_interface/msg/SelfInfo.msg
     * @return true
     * @return false
     */
    bool leafDetailedSearch(cv::Mat& _frame, target_msg::msg::Targets& _output, uint8_t _enemy_color);

private:
    /**
     * @brief 粗略搜索叶片
     *
     * @param _frame 输入一帧图片
     * @param _output 输出一组方框
     * @param _enemy_color 敌方颜色，具体定义查看serial_interface/msg/SelfInfo.msg
     * @return true
     * @return false
     */
    bool leafCoarseSearching(cv::Mat& _frame, std::vector<cv::Rect2d>& _output, uint8_t _enemy_color);

    /**
     * @brief 用于搜索给PNP使用的四个点
     *
     * @param _frame 原始图像
     * @param _fan_roi 提取通道后的叶片ROI图像
     * @param _output_pnp_pos 返回PNP需要的四个点，这四个点实在叶片的ROI图像上的，格式为{ul, ur, bl, br}
     * @param _enemy_color 对面的颜色，具体定义查看serial_interface/msg/SelfInfo.msg
     * @return true
     * @return false
     */
    [[deprecated("已经有新的函数，此函数将不再使用")]]
    bool searchLeafPos(cv::Mat& _frame, cv::Mat& _fan_roi, std::vector<cv::Point2f>& _output_pnp_pos, uint8_t _enemy_color);

    /**
     * @brief 用于搜索给PNP使用的四个点
     *
     * @param _frame 原始图像
     * @param _leaf_rect 整个风车叶片的Rect矩形框
     * @param _leaf_body_rect 风车下半部分的Rect矩形框
     * @param _output_pnp_pos 输出和PNP有关的四个点
     * @param _enemy_color 敌方颜色
     * @return true 搜索成功
     * @return false 搜索失败
     */
    bool searchLeafPos(cv::Mat& _frame, cv::Rect2d& _leaf_rect, cv::Rect2d& _leaf_body_rect, std::vector<cv::Point2f>& _output_pnp_pos, uint8_t _enemy_color);

    cv::Mat erode_struct;
    cv::Mat dilate_struct;
    std::string model_path_;
    std::unique_ptr<NanodetOpenVino> nanodet_detector_ptr_;
    int input_width_;
    int input_height_;
};
}  // namespace energy_detector

#endif  // ENERGY_SEARCH_H__