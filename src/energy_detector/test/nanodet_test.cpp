#include <string>

#include "energy_detector/energy_search.h"
#include "gtest/gtest.h"
#include "target_msg/msg/targets.hpp"

// Nanodet-Plus OpenVino 测试
TEST(energy_detector, nanodet_test) {
#ifdef TEST_DIR
#ifdef CURRENT_PKG_DIR
    energy_detector::EnergySearch energy_search;

    std::string video_path = std::string(TEST_DIR) + "/video/buff_blue.mp4";
    cv::VideoCapture cap(video_path);
    while (true) {
        cv::Mat frame;
        cap >> frame;
        if (frame.empty()) {
            break;
        }
        target_msg::msg::Targets targets;
        cv::Mat color_frame = frame.clone();

        energy_search.leafDetailedSearch(color_frame, targets, 0);

        cv::imshow("object_img", energy_search.object_detect_image);
        // cv::imshow("frame", frame);
        cv::waitKey(10);

        if (!cap.read(frame)) break;
    }
#else
    EXPECT_FALSE(true) << "宏 CURRENT_PKG_DIR 没有定义";
#endif

#else
    EXPECT_FALSE(true) << "宏 TEST_DIR 没有定义";
#endif
}

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
