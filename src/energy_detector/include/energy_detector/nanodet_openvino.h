#ifndef NANODET_OPENVINO_H__
#define NANODET_OPENVINO_H__

#include <opencv2/opencv.hpp>
#include <string>

#include "ie/inference_engine.hpp"

namespace energy_detector {

typedef struct HeadInfo {
    std::string cls_layer;
    std::string dis_layer;
    int stride;
} HeadInfo;

struct CenterPrior {
    int x;
    int y;
    int stride;
};

typedef struct BoxInfo {
    float x1;
    float y1;
    float x2;
    float y2;
    float score;
    int label;
} BoxInfo;

struct object_rect {
    int x;
    int y;
    int width;
    int height;
};

class NanodetOpenVino {
public:
    NanodetOpenVino(const char* _param);

    ~NanodetOpenVino();

    InferenceEngine::ExecutableNetwork network_;
    InferenceEngine::InferRequest infer_request_;
    // static bool hasGPU;

    // 以下为你在训练使用的参数
    // modify these parameters to the same with your config if you want to use your own model
    int input_size[2] = {416, 416};              // input height and width
    int num_class = 3;                           // number of classes. 80 for COCO
    int reg_max = 7;                             // `reg_max` set in the training config. Default: 7.
    std::vector<int> strides = {8, 16, 32, 64};  // strides of the multi-level feature.

    std::vector<BoxInfo> detect(cv::Mat image, float score_threshold, float nms_threshold);

    cv::Mat draw_bboxes(const cv::Mat& bgr, const std::vector<BoxInfo>& bboxes, object_rect effect_roi);

    int resize_uniform(cv::Mat& src, cv::Mat& dst, cv::Size dst_size, object_rect& effect_area);

private:
    void preprocess(cv::Mat& image, InferenceEngine::Blob::Ptr& blob);
    void decode_infer(const float*& pred, std::vector<CenterPrior>& center_priors, float threshold,
                      std::vector<std::vector<BoxInfo>>& results);
    BoxInfo disPred2Bbox(const float*& dfl_det, int label, float score, int x, int y, int stride);
    static void nms(std::vector<BoxInfo>& result, float nms_threshold);
    std::string input_name_ = "data";
    std::string output_name_ = "output";
};
}  // namespace energy_detector

#endif  // NANODET_OPENVINO_H__