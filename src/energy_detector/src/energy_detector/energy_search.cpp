#include "energy_detector/energy_search.h"

namespace energy_detector {

void drawRotatedRect(cv::Mat &_src, const cv::Mat &_bin_mat, cv::RotatedRect &_rect) {
    std::array<cv::Point2f, 4> points;
    _rect.points(points.begin());
    float x_ratio = static_cast<float>(_src.cols) / static_cast<float>(_bin_mat.cols);
    float y_ratio = static_cast<float>(_src.rows) / static_cast<float>(_bin_mat.rows);
    for (size_t i = 0; i < points.size(); ++i) {
        cv::Point2f p1(points[i].x * x_ratio, points[i].y * y_ratio);
        cv::Point2f p2(points[(i + 1) % 4].x * x_ratio, points[(i + 1) % 4].y * y_ratio);
        cv::line(_src, p1, p2, cv::Scalar(0, 0, 255), 2);
    }
}

void drawPloy(cv::Mat &_src, const cv::Mat &_bin_mat, std::vector<cv::Point2f> &_points) {
    float x_ratio = static_cast<float>(_src.cols) / static_cast<float>(_bin_mat.cols);
    float y_ratio = static_cast<float>(_src.rows) / static_cast<float>(_bin_mat.rows);

    for (size_t i = 0; i < _points.size(); ++i) {
        cv::Point2f p1(_points[i].x * x_ratio, _points[i].y * y_ratio);
        cv::Point2f p2(_points[(i + 1) % _points.size()].x * x_ratio, _points[(i + 1) % _points.size()].y * y_ratio);
        cv::line(_src, p1, p2, cv::Scalar(0, 255, 0), 2);
    }
}

void drawPoint(cv::Mat &_src, const cv::Mat &_bin_mat, cv::Point2f &_points) {
    float x_ratio = static_cast<float>(_src.cols) / static_cast<float>(_bin_mat.cols);
    float y_ratio = static_cast<float>(_src.rows) / static_cast<float>(_bin_mat.rows);

    cv::circle(_src, cv::Point(_points.x * x_ratio, _points.y * y_ratio), 1, cv::Scalar(0, 255, 255), 2);
}

void drawText(cv::Mat &_src, const cv::Mat &_bin_mat, cv::Point2f &_points, std::string _txt) {
    float x_ratio = static_cast<float>(_src.cols) / static_cast<float>(_bin_mat.cols);
    float y_ratio = static_cast<float>(_src.rows) / static_cast<float>(_bin_mat.rows);

    cv::putText(_src, _txt, cv::Point(_points.x * x_ratio, _points.y * y_ratio), cv::FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(255, 255, 0), 1);
}

void drawText(cv::Mat &_src, cv::Point2f &_points, std::string _txt) {
    cv::putText(_src, _txt, cv::Point(_points.x, _points.y), cv::FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(255, 255, 0), 1);
}

double getTwoPointDistance(cv::Point2f &_p1, cv::Point2f &_p2) {
    auto ret = _p1 - _p2;
    return sqrt(pow(ret.x, 2) + pow(ret.y, 2));
}

double getRotatedRectArea(cv::RotatedRect &_rect) {
    std::array<cv::Point2f, 4> points;
    _rect.points(points.begin());
    double l1 = getTwoPointDistance(points.at(0), points.at(1));
    double l2 = getTwoPointDistance(points.at(1), points.at(2));
    return l1 * l2;
}

double getRotatedRectRatio(cv::RotatedRect &_rect) {
    std::array<cv::Point2f, 4> points;
    _rect.points(points.begin());
    double l1 = getTwoPointDistance(points.at(0), points.at(1));
    double l2 = getTwoPointDistance(points.at(1), points.at(2));
    return l1 < l2 ? l2 / l1 : l1 / l2;
}

/**
 * 搜索一组点中距离参考点最近的点
 * @param _vec_points   输入的一组点
 * @param _close_point  参考的点
 * @param _count        最近的几个
 * @return
 */
std::vector<cv::Point2f> getClosestPoints(std::vector<cv::Point2f> _vec_points, cv::Point2f _close_point, int _count = 1) {
    double distance = DBL_MAX;
    int min_index = -1;
    std::vector<cv::Point2f> result;
    std::vector<cv::Point2f> vec_points(_vec_points);
    while (_count--) {
        for (size_t i = 0; i < vec_points.size(); ++i) {
            double tmp_distance = getTwoPointDistance(_close_point, vec_points.at(i));
            if (distance > tmp_distance) {
                min_index = i;
                distance = tmp_distance;
            }
        }
        result.emplace_back(vec_points.at(min_index));
        vec_points.erase(vec_points.begin() + min_index);
        distance = DBL_MAX;
    }
    return result;
}

cv::Rect2d getRelativeRect(cv::Rect2d &_whole, cv::Rect2d &_parted) {
    double x, y, width, height;
    double dx = _parted.x - _whole.x, dy = _parted.y - _whole.y, d_height = _parted.y + _parted.height, d_width = _parted.x + _parted.width;

    x = dx <= 0 ? 0 : dx;
    y = dy <= 0 ? 0 : dy;
    width = d_width >= (_whole.x + _whole.width) ? (_whole.width - x - 1) : _parted.width;
    height = d_height >= (_whole.y + _whole.height) ? (_whole.height - y - 1) : _parted.height;

    return {x, y, width, height};
}

cv::Mat eraseWhitePollution(cv::Mat &_input, uint8_t _enemy_color) {
    std::vector<cv::Mat> channels;
    cv::split(_input, channels);
    cv::Mat b_channel, g_channel, r_channel, output;
    // BLUE
    if (_enemy_color == SelfInfo::BLUE) {
        cv::threshold(channels[0], b_channel, 140, 255, cv::THRESH_BINARY);
        cv::threshold(channels[1], g_channel, 255, 255, cv::THRESH_BINARY);
        cv::threshold(channels[2], r_channel, 190, 255, cv::THRESH_BINARY);
        cv::bitwise_or(g_channel, r_channel, g_channel);
        cv::bitwise_not(g_channel, g_channel);
        cv::bitwise_and(b_channel, g_channel, output);
    } else if (_enemy_color == SelfInfo::RED) {
        cv::threshold(channels[0], b_channel, 220, 255, cv::THRESH_BINARY);
        cv::threshold(channels[1], g_channel, 220, 255, cv::THRESH_BINARY);
        cv::threshold(channels[2], r_channel, 220, 255, cv::THRESH_BINARY);
        cv::bitwise_or(g_channel, b_channel, g_channel);
        cv::bitwise_not(g_channel, g_channel);
        cv::bitwise_and(r_channel, g_channel, output);
    }
    return output;
}

cv::Point2f cvtPoint(cv::Mat &_src, cv::Mat &_bin_mat, cv::Point2f _input) {
    float x_ratio = static_cast<float>(_src.cols) / static_cast<float>(_bin_mat.cols);
    float y_ratio = static_cast<float>(_src.rows) / static_cast<float>(_bin_mat.rows);

    return cv::Point2f(_input.x * x_ratio, _input.y * y_ratio);
}

EnergySearch::EnergySearch() {
#ifdef CURRENT_PKG_DIR
    this->model_path_ = std::string(CURRENT_PKG_DIR) + "/models/leaf416_planc.xml";
    this->nanodet_detector_ptr_ = std::make_unique<NanodetOpenVino>(this->model_path_.c_str());
    this->input_height_ = nanodet_detector_ptr_->input_size[0];
    this->input_width_ = nanodet_detector_ptr_->input_size[1];
#else
#error "Macro CURRENT_PKG_DIR not defined"
#endif
    this->erode_struct = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(10, 10));
    this->dilate_struct = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(12, 12));

    this->object_detect_image = this->debug_image = cv::Mat::zeros(640, 480, CV_8UC3);
}

bool EnergySearch::searchLeafPos(cv::Mat &_frame, cv::Mat &_fan_roi, std::vector<cv::Point2f> &_output_pnp_pos, uint8_t _enemy_color) {
    const int WIDTH = _fan_roi.cols * 2;
    const int HEIGHT = _fan_roi.rows * 2;

    cv::Mat bin_channel, resized_bin_channel;
    cv::threshold(_fan_roi, bin_channel, 157, 255, cv::THRESH_BINARY);
    cv::resize(bin_channel, resized_bin_channel, cv::Size(0, 0), WIDTH / _fan_roi.cols, HEIGHT / _fan_roi.rows);

    cv::Mat dilate_mat = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
    cv::dilate(resized_bin_channel, resized_bin_channel, dilate_mat, cv::Point(-1, -1), 3);

    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    findContours(resized_bin_channel, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_NONE);

    for (size_t i = 0; i < hierarchy.size(); ++i) {
        // 同层级不含有
        if (!(hierarchy[i][0] >= 0 || hierarchy[i][1] >= 0) || !(hierarchy[i][2] == -1 && hierarchy[i][3] == -1)) {
            return false;
        }
    }

    size_t leaf_top_index = -1, leaf_body_index = -1;
    cv::RotatedRect leaf_top_rect, leaf_body_rect;
    for (size_t i = 0; i < contours.size(); i++) {
        cv::RotatedRect rect = cv::minAreaRect(contours[i]);

        if (getRotatedRectArea(rect) < (resized_bin_channel.rows * resized_bin_channel.cols / 8.0)) {
            continue;
        }

        if (1.0 < getRotatedRectRatio(rect) && getRotatedRectRatio(rect) < 2) {
            leaf_body_index = i;
            leaf_body_rect = rect;
        } else if (2.0 < getRotatedRectRatio(rect) && getRotatedRectRatio(rect) < 2.8) {
            leaf_top_index = i;
            leaf_top_rect = rect;
        }
    }

    if (leaf_body_index < 0 || leaf_top_index < 0) {
        return false;
    }

    // 优先搜索旋转矩形的两个点
    cv::Point2f top_rect_points[4];
    leaf_top_rect.points(top_rect_points);
    auto leaf_top_close_points = getClosestPoints(std::vector<cv::Point2f>(top_rect_points, top_rect_points + 4), leaf_body_rect.center, 2);

    cv::Point2f body_rect_points[4];
    leaf_body_rect.points(body_rect_points);
    auto leaf_body_close_points =
        getClosestPoints(std::vector<cv::Point2f>(body_rect_points, body_rect_points + 4), leaf_top_rect.center, 2);

    // 拟合多边形搜索最近的点
    if (!(0 <= leaf_top_index && leaf_top_index < contours.size()) || !(0 <= leaf_body_index && leaf_body_index < contours.size())) {
        return false;
    }
    std::vector<cv::Point2f> leaf_body_ploy;
    cv::approxPolyDP(contours[leaf_body_index], leaf_body_ploy, 30, true);
    auto real_leaf_body_point1 = getClosestPoints(leaf_body_ploy, leaf_body_close_points.at(0), 1);
    auto real_leaf_body_point2 = getClosestPoints(leaf_body_ploy, leaf_body_close_points.at(1), 1);

    std::vector<cv::Point2f> leaf_top_ploy;
    cv::approxPolyDP(contours[leaf_top_index], leaf_top_ploy, 30, true);
    auto real_leaf_top_point1 = getClosestPoints(leaf_top_ploy, leaf_top_close_points.at(0), 1);
    auto real_leaf_top_point2 = getClosestPoints(leaf_top_ploy, leaf_top_close_points.at(1), 1);

    cv::Point2f ul, ur, bl, br;
    double leaf_angle =
        asin((leaf_top_rect.center.x - leaf_body_rect.center.x) / getTwoPointDistance(leaf_body_rect.center, leaf_top_rect.center));
    leaf_angle += CV_PI;

    if ((CV_PI / 4.0) < leaf_angle && (CV_PI * 3.0 / 4.0) >= leaf_angle) {
        ul = real_leaf_top_point1.at(0).x > real_leaf_top_point2.at(0).x ? real_leaf_top_point1.at(0) : real_leaf_top_point2.at(0);
        ur = real_leaf_top_point1.at(0).x < real_leaf_top_point2.at(0).x ? real_leaf_top_point1.at(0) : real_leaf_top_point2.at(0);
        bl = real_leaf_body_point1.at(0).x > real_leaf_body_point2.at(0).x ? real_leaf_body_point1.at(0) : real_leaf_body_point2.at(0);
        br = real_leaf_body_point1.at(0).x < real_leaf_body_point2.at(0).x ? real_leaf_body_point1.at(0) : real_leaf_body_point2.at(0);
    } else if ((CV_PI * 3.0 / 4.0) < leaf_angle && (CV_PI * 5.0 / 4.0) >= leaf_angle) {
        ul = real_leaf_top_point1.at(0).y > real_leaf_top_point2.at(0).y ? real_leaf_top_point1.at(0) : real_leaf_top_point2.at(0);
        ur = real_leaf_top_point1.at(0).y < real_leaf_top_point2.at(0).y ? real_leaf_top_point1.at(0) : real_leaf_top_point2.at(0);
        bl = real_leaf_body_point1.at(0).y > real_leaf_body_point2.at(0).y ? real_leaf_body_point1.at(0) : real_leaf_body_point2.at(0);
        br = real_leaf_body_point1.at(0).y < real_leaf_body_point2.at(0).y ? real_leaf_body_point1.at(0) : real_leaf_body_point2.at(0);
    } else if ((CV_PI * 5.0 / 4.0) < leaf_angle && (CV_PI * 7.0 / 4.0) >= leaf_angle) {
        ul = real_leaf_top_point1.at(0).x < real_leaf_top_point2.at(0).x ? real_leaf_top_point1.at(0) : real_leaf_top_point2.at(0);
        ur = real_leaf_top_point1.at(0).x > real_leaf_top_point2.at(0).x ? real_leaf_top_point1.at(0) : real_leaf_top_point2.at(0);
        bl = real_leaf_body_point1.at(0).x < real_leaf_body_point2.at(0).x ? real_leaf_body_point1.at(0) : real_leaf_body_point2.at(0);
        br = real_leaf_body_point1.at(0).x > real_leaf_body_point2.at(0).x ? real_leaf_body_point1.at(0) : real_leaf_body_point2.at(0);
    } else if ((CV_PI * 7.0 / 4.0) < leaf_angle && (CV_PI * 3.0 / 4.0) >= leaf_angle) {
        ul = real_leaf_top_point1.at(0).y < real_leaf_top_point2.at(0).y ? real_leaf_top_point1.at(0) : real_leaf_top_point2.at(0);
        ur = real_leaf_top_point1.at(0).y > real_leaf_top_point2.at(0).y ? real_leaf_top_point1.at(0) : real_leaf_top_point2.at(0);
        bl = real_leaf_body_point1.at(0).y < real_leaf_body_point2.at(0).y ? real_leaf_body_point1.at(0) : real_leaf_body_point2.at(0);
        br = real_leaf_body_point1.at(0).y > real_leaf_body_point2.at(0).y ? real_leaf_body_point1.at(0) : real_leaf_body_point2.at(0);
    }

    ur = cvtPoint(_fan_roi, resized_bin_channel, ur);
    ul = cvtPoint(_fan_roi, resized_bin_channel, ul);
    br = cvtPoint(_fan_roi, resized_bin_channel, br);
    bl = cvtPoint(_fan_roi, resized_bin_channel, bl);

    std::vector<cv::Point2f> pnp_points = {ul, ur, bl, br};

    return true;
}

bool EnergySearch::searchLeafPos(cv::Mat &_frame, cv::Rect2d &_leaf_rect, cv::Rect2d &_leaf_body_rect,
                                 std::vector<cv::Point2f> &_output_pnp_pos, uint8_t _enemy_color) {
    // 放大处理
    // TODO: 如果要修复风车缺陷产生的检测错误需要在这里修改
    // double x_scalar, y_scalar, fx_scalar, fy_scalar;
    // auto leaf_body_related_rect = getRelativeRect(_leaf_rect, _leaf_body_rect);
    // cv::Rect2d leaf_related_rect = cv::Rect2d(cv::Point2f(0, 0), cv::Point2f(_leaf_rect.width, _leaf_rect.height));
    // x_scalar = leaf_body_related_rect.x / leaf_related_rect.width;
    // y_scalar = leaf_body_related_rect.y / leaf_related_rect.height;
    // fx_scalar = (leaf_body_related_rect.x + leaf_body_related_rect.width) / leaf_related_rect.width;
    // fy_scalar = (leaf_body_related_rect.y + leaf_body_related_rect.height) / leaf_related_rect.height;

    cv::Mat frame = _frame.clone();
    cv::Mat leaf_roi = frame(_leaf_rect), resized_leaf_roi;

    cv::resize(leaf_roi, resized_leaf_roi, cv::Size(-1, -1), 6, 6);
    // cv::Rect2d resized_leaf_body_rect = cv::Rect2d(cv::Point2d(x_scalar * resized_leaf_roi.cols, y_scalar * resized_leaf_roi.rows),
    //                                               cv::Point2d(fx_scalar * resized_leaf_roi.cols, fy_scalar * resized_leaf_roi.rows));
    double cols_x_rows = leaf_roi.cols * leaf_roi.rows;

    // 图像预先处理
    auto resized_bin_mat = eraseWhitePollution(resized_leaf_roi, _enemy_color);

    cv::dilate(resized_bin_mat, resized_bin_mat, this->dilate_struct, cv::Point(-1, -1), 5);
    cv::erode(resized_bin_mat, resized_bin_mat, this->dilate_struct, cv::Point(-1, -1), 5);
    cv::erode(resized_bin_mat, resized_bin_mat, this->erode_struct, cv::Point(-1, -1));
    cv::dilate(resized_bin_mat, resized_bin_mat, this->erode_struct, cv::Point(-1, -1));

    cv::Mat resized_colored_bin_mat = cv::Mat(resized_bin_mat.size(), CV_8UC3);
    cv::cvtColor(resized_bin_mat, resized_colored_bin_mat, cv::COLOR_GRAY2BGR);

    // 搜索轮廓
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::RotatedRect leaf_body_rect, leaf_top_rect;
    int leaf_body_index = -1, leaf_top_index = -1;

    cv::findContours(resized_bin_mat, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_NONE);

    for (size_t i = 0; i < contours.size(); ++i) {
        auto rect = cv::minAreaRect(contours[i]);
        //            cv::approxPolyDP(contours[i], leaf_body_ploy, 30, false);
        if (getRotatedRectArea(rect) < (cols_x_rows * 0.9)) {
            continue;
        }
        drawRotatedRect(resized_colored_bin_mat, resized_colored_bin_mat, rect);
        if (1.0 < getRotatedRectRatio(rect) && getRotatedRectRatio(rect) < 2) {
            leaf_body_index = i;
            leaf_body_rect = rect;
            //                std::cout << "body:" << getRotatedRectRatio(rect) << std::endl;
            continue;
        } else if (2.0 < getRotatedRectRatio(rect) && getRotatedRectRatio(rect) < 3.5) {
            leaf_top_index = i;
            leaf_top_rect = rect;
            //                std::cout << "top:" << getRotatedRectRatio(rect) << std::endl;
            continue;
        }
        //            std::cout << "others:" << getRotatedRectRatio(rect) << std::endl;
    }
    //        std::cout << "---" << std::endl;

    /*
     * 假设风车有缺陷的检测方法
     * 计算图像中心与leaf_body之间的角度
     * 计算缺陷和leaf_body之间的角度
     * 检查两者之间的差
     * 合并符合的带缺陷的元素
     * */
    if (leaf_body_index < 0 || leaf_top_index < 0) {
        return false;
    }

    // 搜索符合PNP的四个点

    // // 搜索Rect中离Top中心最近的两个点
    std::array<cv::Point2f, 4> body_points, top_points;
    std::vector<cv::Point2f> vec_body_points, vec_top_points;
    leaf_body_rect.points(body_points.begin());
    vec_body_points.insert(vec_body_points.begin(), body_points.begin(), body_points.end());
    leaf_top_rect.points(top_points.begin());
    vec_top_points.insert(vec_top_points.begin(), top_points.begin(), top_points.end());

    auto top_point_selected = getClosestPoints(vec_top_points, leaf_body_rect.center, 2);
    auto body_point_selected = getClosestPoints(vec_body_points, leaf_top_rect.center, 2);

    cv::line(resized_colored_bin_mat, top_point_selected[0], top_point_selected[1], cv::Scalar(255, 255, 0), 2);
    cv::line(resized_colored_bin_mat, body_point_selected[0], body_point_selected[1], cv::Scalar(255, 255, 0), 2);

    double angle, xx, yy;
    xx = leaf_top_rect.center.x - leaf_body_rect.center.x;
    yy = leaf_top_rect.center.y - leaf_body_rect.center.y;
    if (xx == 0.0) {
        angle = CV_PI / 2.0;
    } else {
        angle = atan(fabs(yy / xx));
    }

    if ((xx < 0.0) && (yy >= 0.0)) {
        angle = CV_PI - angle;
    } else if ((xx < 0.0) && (yy < 0.0)) {
        angle = CV_PI + angle;
    } else if ((xx >= 0.0) && (yy < 0.0)) {
        angle = CV_PI * 2 - angle;
    }

    angle = angle / CV_PI * 360.0 / 2;

    cv::Point2f above_left, above_right, below_left, below_right;
    if (45.0 < angle && angle <= 135.0) {
        if (top_point_selected[0].x < top_point_selected[1].x) {
            above_left = top_point_selected[1];
            above_right = top_point_selected[0];
        } else {
            above_right = top_point_selected[1];
            above_left = top_point_selected[0];
        }

        if (body_point_selected[0].x < body_point_selected[1].x) {
            below_left = body_point_selected[1];
            below_right = body_point_selected[0];
        } else {
            below_right = body_point_selected[1];
            below_left = body_point_selected[0];
        }
    } else if (135.0 < angle && angle <= 225.0) {
        if (top_point_selected[0].y < top_point_selected[1].y) {
            above_left = top_point_selected[1];
            above_right = top_point_selected[0];
        } else {
            above_right = top_point_selected[1];
            above_left = top_point_selected[0];
        }

        if (body_point_selected[0].y < body_point_selected[1].y) {
            below_left = body_point_selected[1];
            below_right = body_point_selected[0];
        } else {
            below_right = body_point_selected[1];
            below_left = body_point_selected[0];
        }
    } else if (225.0 < angle && angle <= 315.0) {
        if (top_point_selected[0].x > top_point_selected[1].x) {
            above_left = top_point_selected[1];
            above_right = top_point_selected[0];
        } else {
            above_right = top_point_selected[1];
            above_left = top_point_selected[0];
        }

        if (body_point_selected[0].x > body_point_selected[1].x) {
            below_left = body_point_selected[1];
            below_right = body_point_selected[0];
        } else {
            below_right = body_point_selected[1];
            below_left = body_point_selected[0];
        }
    } else if ((315.0 < angle && angle <= 360.0) || (0 <= angle && angle <= 45)) {
        if (top_point_selected[0].y > top_point_selected[1].y) {
            above_left = top_point_selected[1];
            above_right = top_point_selected[0];
        } else {
            above_right = top_point_selected[1];
            above_left = top_point_selected[0];
        }

        if (body_point_selected[0].y > body_point_selected[1].y) {
            below_left = body_point_selected[1];
            below_right = body_point_selected[0];
        } else {
            below_right = body_point_selected[1];
            below_left = body_point_selected[0];
        }
    }

#ifndef NDEBUG
    cv::Point2f angle_pos(30, 80);
    drawText(resized_colored_bin_mat, resized_colored_bin_mat, angle_pos, std::to_string(angle));
    drawText(resized_colored_bin_mat, resized_colored_bin_mat, below_left, "BL");
    drawText(resized_colored_bin_mat, resized_colored_bin_mat, below_right, "BR");
    drawText(resized_colored_bin_mat, resized_colored_bin_mat, above_left, "AL");
    drawText(resized_colored_bin_mat, resized_colored_bin_mat, above_right, "AR");
#endif

    cv::Point2f leaf_rect_tl_point(_leaf_rect.x, _leaf_rect.y);
    auto al = cvtPoint(leaf_roi, resized_bin_mat, above_left) + leaf_rect_tl_point;
    auto ar = cvtPoint(leaf_roi, resized_bin_mat, above_right) + leaf_rect_tl_point;
    auto bl = cvtPoint(leaf_roi, resized_bin_mat, below_left) + leaf_rect_tl_point;
    auto br = cvtPoint(leaf_roi, resized_bin_mat, below_right) + leaf_rect_tl_point;

    _output_pnp_pos.emplace_back(al);
    _output_pnp_pos.emplace_back(ar);
    _output_pnp_pos.emplace_back(bl);
    _output_pnp_pos.emplace_back(br);

    return false;
}

// 弃用
bool EnergySearch::leafCoarseSearching(cv::Mat &_frame, std::vector<cv::Rect2d> &_output, uint8_t _enemy_color) {
    std::vector<cv::Mat> img_channels;
    cv::split(_frame, img_channels);
    std::vector<energy_detector::BoxInfo> search_result;  // 粗略搜索结果会放到这里
    object_rect effect_roi;
    // 蓝色
    if (_enemy_color == serial_interface::msg::SelfInfo::BLUE) {
        cv::Mat b_channel = img_channels.at(0);
        cv::Mat b_channel_color, b_channel_resized;
        cv::cvtColor(b_channel, b_channel_color, cv::COLOR_GRAY2BGR);

        this->nanodet_detector_ptr_->resize_uniform(b_channel_color, b_channel_resized, cv::Size(this->input_width_, this->input_height_),
                                                    effect_roi);

        this->debug_image = b_channel_resized.clone();
        search_result = this->nanodet_detector_ptr_->detect(b_channel_resized, 0.6, 0.5);

    }
    // 红色
    else if (_enemy_color == serial_interface::msg::SelfInfo::RED) {
        cv::Mat r_channel = img_channels.at(2);
        cv::Mat r_channel_color, r_channel_resized;
        cv::cvtColor(r_channel, r_channel_color, cv::COLOR_GRAY2BGR);

        this->nanodet_detector_ptr_->resize_uniform(r_channel_color, r_channel_resized, cv::Size(this->input_width_, this->input_height_),
                                                    effect_roi);
        search_result = this->nanodet_detector_ptr_->detect(r_channel_resized, 0.6, 0.5);
    }

    if (search_result.empty()) {
        return false;
    }

    cv::Mat image = _frame.clone();
    this->object_detect_image = this->nanodet_detector_ptr_->draw_bboxes(image, search_result, effect_roi);

    static const char *class_names[] = {"leaf", "leaf_top", "leaf_body"};
    int src_w = image.cols;
    int src_h = image.rows;
    int dst_w = effect_roi.width;
    int dst_h = effect_roi.height;
    float width_ratio = (float)src_w / (float)dst_w;
    float height_ratio = (float)src_h / (float)dst_h;

    bool is_leaf_found = false, is_leaf_body_found = false;
    cv::Rect2d leaf_rect, leaf_body_rect;

    for (size_t i = 0; i < search_result.size(); i++) {
        const BoxInfo &bbox = search_result[i];
        cv::Rect2d rect = cv::Rect2d(cv::Point((bbox.x1 - effect_roi.x) * width_ratio, (bbox.y1 - effect_roi.y) * height_ratio),
                                     cv::Point((bbox.x2 - effect_roi.x) * width_ratio, (bbox.y2 - effect_roi.y) * height_ratio));

        std::string label = std::string(class_names[bbox.label]);

        if (label == "leaf") {
            is_leaf_found = true;
            leaf_rect = rect;
        } else if (label == "leaf_body") {
            is_leaf_body_found = true;
            leaf_body_rect = rect;
        }
    }

    if (!is_leaf_body_found || !is_leaf_found) {
        return false;
    }

    _output.emplace_back(leaf_rect);
    _output.emplace_back(leaf_body_rect);

    //        _output.emplace_back(rect);
    return true;
}

bool EnergySearch::leafDetailedSearch(cv::Mat &_frame, target_msg::msg::Targets &_output, uint8_t _enemy_color) {
    std::vector<cv::Rect2d> nanodet_output;
    std::vector<cv::Point2f> pnp_points_output;
    cv::Mat frame = _frame.clone();
    cv::Mat fan_roi;
    cv::Rect fan_roi_rect;
    cv::Rect2d leaf_rect, leaf_body_rect;
    std::vector<cv::Mat> channels;
    cv::split(frame, channels);

    int cols_4 = _frame.cols / 4;
    int rows_4 = _frame.rows / 4;

    cv::Mat energy_roi = _frame(cv::Rect(cols_4, rows_4, cols_4 * 2, rows_4 * 2));

    if (!this->leafCoarseSearching(energy_roi, nanodet_output, _enemy_color)) {
        return false;
    }

    leaf_rect = nanodet_output[0];
    leaf_body_rect = nanodet_output[1];

    //     this->searchLeafPos(_frame, leaf_rect, leaf_body_rect, pnp_points_output, _enemy_color);

    //     if (pnp_points_output.empty()) {
    //         return false;
    //     }

    //     for (size_t i = 0; i < pnp_points_output.size(); i++) {
    //         pnp_points_output[i].x += fan_roi_rect.x;
    //         pnp_points_output[i].y += fan_roi_rect.y;
    //     }

    // #ifndef NDEBUG
    //     drawText(this->debug_image, pnp_points_output[0], "UL");
    //     drawText(this->debug_image, pnp_points_output[1], "UR");
    //     drawText(this->debug_image, pnp_points_output[2], "BL");
    //     drawText(this->debug_image, pnp_points_output[3], "BR");
    // #endif

    //     target_msg::msg::Target target;
    //     target.above_left.x = pnp_points_output[0].x;
    //     target.above_left.y = pnp_points_output[0].y;
    //     target.above_right.x = pnp_points_output[1].x;
    //     target.above_right.y = pnp_points_output[1].y;
    //     target.below_left.x = pnp_points_output[2].x;
    //     target.below_left.y = pnp_points_output[2].y;
    //     target.below_right.x = pnp_points_output[3].x;
    //     target.below_right.y = pnp_points_output[3].y;
    //     target.color = _enemy_color;
    //     target.id = 0;
    //     target.type = 3;

    //     _output.targets.emplace_back(target);
    return true;
}
}  // namespace energy_detector
