std_msgs/Header header

uint8 BLUE=0
uint8 RED=1

uint8 TOTAL_MODES=4
uint8 NO_MODE=0
uint8 ARMOR_AUTO_AIM=1
uint8 ENERGY_PREPARING=2
uint8 ENERGY_AIMING=3

uint8 sof 0xa5
uint8 robot_mode 0 # 0:no_mode-1:armor_auto_aim-2:fan_wait_aim-3:fan_aiming
uint8 robot_color 0 # 0:blue-1:red
uint8[] raw_info    # 原始数据

float64 cloud_pitch 0.0
float64 cloud_yaw 0.0
int16 forward 0
float64 bullet_speed 30