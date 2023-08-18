uint8 id
uint8 color # 颜色
uint8 type  # 类型:1-装甲板小 2-装甲板大 3-风车叶片
uint8 priority # 优先级(暂时想不到在哪里用)

uint8 ARMOR_SMALL=1
uint8 ARMOR_BIG=2
uint8 ENERGY_LEAF=3

float32 distance_to_image_center

# 四个点按照图像划分
Point2D above_left  # 左上
Point2D above_right  # 右上
Point2D below_left  # 左下
Point2D below_right  # 右下

Point2D armor_center

geometry_msgs/Pose pose
