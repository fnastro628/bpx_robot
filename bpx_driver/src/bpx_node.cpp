#include "bpx_driver/bpx_node.hpp"
#include "motion_types.h"

#include <chrono>
#include <functional>

namespace bpx_driver {

// ── Construction / destruction ───────────────────────────────────────────────

BpxNode::BpxNode(const rclcpp::NodeOptions& opts)
: Node("bpx_node", opts)
{
    declare_parameter("robot_ip",          "10.21.20.1");
    declare_parameter("state_port",        9873);
    declare_parameter("state_rate_hz",     100);
    declare_parameter("command_rate_hz",   50);
    declare_parameter("publish_rate_hz",   100);

    const auto ip        = get_parameter("robot_ip").as_string();
    const auto s_port    = static_cast<uint16_t>(get_parameter("state_port").as_int());
    const auto s_rate    = static_cast<uint16_t>(get_parameter("state_rate_hz").as_int());
    const auto cmd_rate  = static_cast<uint16_t>(get_parameter("command_rate_hz").as_int());
    const int  pub_rate  = get_parameter("publish_rate_hz").as_int();

    joint_names_ = {
        "fl_hip_roll_joint",  "fl_hip_pitch_joint",  "fl_knee_joint",
        "fr_hip_roll_joint",  "fr_hip_pitch_joint",  "fr_knee_joint",
        "hl_hip_roll_joint",  "hl_hip_pitch_joint",  "hl_knee_joint",
        "hr_hip_roll_joint",  "hr_hip_pitch_joint",  "hr_knee_joint"
    };

    // ── SDK setup ────────────────────────────────────────────────────────────
    robot_.setRobotIp(ip.c_str());
    robot_.setRobotStateUploadPort(s_port);
    robot_.setRobotStateUploadRate(s_rate);
    robot_.setMotionCommandRate(cmd_rate);
    robot_.setVelocityControlFlag(true);

    if (!robot_.connect()) {
        RCLCPP_FATAL(get_logger(), "Could not connect to BPX at %s", ip.c_str());
    } else {
        RCLCPP_INFO(get_logger(), "Connected to BPX at %s", ip.c_str());
    }

    // ── Publishers ───────────────────────────────────────────────────────────
    joint_pub_        = create_publisher<sensor_msgs::msg::JointState>("/bpx/joint_states", 10);
    imu_pub_          = create_publisher<sensor_msgs::msg::Imu>("/bpx/imu", 10);
    battery_pub_      = create_publisher<sensor_msgs::msg::BatteryState>("/bpx/battery", 10);
    motion_state_pub_ = create_publisher<std_msgs::msg::UInt8>("/bpx/motion_state", 10);
    odom_pub_         = create_publisher<nav_msgs::msg::Odometry>("/bpx/odometry", 10);

    // ── Subscriber ───────────────────────────────────────────────────────────
    cmd_vel_sub_ = create_subscription<geometry_msgs::msg::Twist>(
        "/cmd_vel", 10,
        std::bind(&BpxNode::onCmdVel, this, std::placeholders::_1));

    // ── Services ─────────────────────────────────────────────────────────────
    stand_srv_    = create_service<std_srvs::srv::Trigger>("/bpx/stand",
        std::bind(&BpxNode::srvStand,        this, std::placeholders::_1, std::placeholders::_2));
    sit_srv_      = create_service<std_srvs::srv::Trigger>("/bpx/sit",
        std::bind(&BpxNode::srvSit,          this, std::placeholders::_1, std::placeholders::_2));
    damp_srv_     = create_service<std_srvs::srv::Trigger>("/bpx/damp",
        std::bind(&BpxNode::srvDamp,         this, std::placeholders::_1, std::placeholders::_2));
    zero_pos_srv_ = create_service<std_srvs::srv::Trigger>("/bpx/zero_position",
        std::bind(&BpxNode::srvZeroPosition, this, std::placeholders::_1, std::placeholders::_2));

    // ── State timer ──────────────────────────────────────────────────────────
    state_timer_ = create_wall_timer(
        std::chrono::milliseconds(1000 / pub_rate),
        std::bind(&BpxNode::publishState, this));

    RCLCPP_INFO(get_logger(), "BpxNode ready @ %d Hz pub", pub_rate);
}

BpxNode::~BpxNode() {
    robot_.setDamping();
    robot_.disconnect();
}

// ── State publish ────────────────────────────────────────────────────────────

void BpxNode::publishState() {
    const auto now = get_clock()->now();

    // Joint states
    float pos[kNumJoints], vel[kNumJoints], tau[kNumJoints];
    robot_.getJointPosition(pos);
    robot_.getJointVelocity(vel);
    robot_.getJointTorque(tau);

    auto js = sensor_msgs::msg::JointState();
    js.header.stamp    = now;
    js.header.frame_id = "base_link";
    js.name.assign(joint_names_.begin(), joint_names_.end());
    js.position.assign(pos, pos + kNumJoints);
    js.velocity.assign(vel, vel + kNumJoints);
    js.effort.assign(tau,   tau + kNumJoints);
    joint_pub_->publish(js);

    // IMU
    float rpy[3], quat[4], acc[3], omega[3];
    robot_.getImuRpy(rpy);
    robot_.getImuQuat(quat);        // x, y, z, w
    robot_.getImuAcc(acc);
    robot_.getImuOmega(omega);

    auto imu = sensor_msgs::msg::Imu();
    imu.header.stamp    = now;
    imu.header.frame_id = "imu_link";
    imu.orientation.x = quat[0]; imu.orientation.y = quat[1];
    imu.orientation.z = quat[2]; imu.orientation.w = quat[3];
    imu.linear_acceleration.x = acc[0];
    imu.linear_acceleration.y = acc[1];
    imu.linear_acceleration.z = acc[2];
    imu.angular_velocity.x = omega[0];
    imu.angular_velocity.y = omega[1];
    imu.angular_velocity.z = omega[2];
    // Covariances unknown — leave as zero matrices
    imu_pub_->publish(imu);

    // Battery
    uint8_t pct; float amps;
    robot_.getBatteryLevel(&pct);
    robot_.getBatteryCurrent(&amps);

    auto bat = sensor_msgs::msg::BatteryState();
    bat.header.stamp = now;
    bat.percentage   = static_cast<float>(pct) / 100.0f;
    bat.current      = amps;
    bat.power_supply_status =
        sensor_msgs::msg::BatteryState::POWER_SUPPLY_STATUS_DISCHARGING;
    battery_pub_->publish(bat);

    // Motion state
    uint8_t state = 0;
    robot_.getCurrentMotionState(&state);
    auto sm = std_msgs::msg::UInt8();
    sm.data = state;
    motion_state_pub_->publish(sm);

    // Odometry
    float bv[3], odom[3];
    robot_.getCurrentVelocityBody(bv);
    robot_.getLegOdom(odom);

    auto od = nav_msgs::msg::Odometry();
    od.header.stamp      = now;
    od.header.frame_id   = "odom";
    od.child_frame_id    = "base_link";
    od.pose.pose.position.x = odom[0];
    od.pose.pose.position.y = odom[1];
    od.pose.pose.position.z = odom[2];
    od.twist.twist.linear.x  = bv[0];
    od.twist.twist.linear.y  = bv[1];
    od.twist.twist.angular.z = bv[2];
    odom_pub_->publish(od);
}

// ── Subscriptions ────────────────────────────────────────────────────────────

void BpxNode::onCmdVel(geometry_msgs::msg::Twist::SharedPtr msg) {
    robot_.setVelocity(
        static_cast<float>(msg->linear.x),
        static_cast<float>(msg->linear.y),
        static_cast<float>(msg->angular.z));
}

// ── Services ─────────────────────────────────────────────────────────────────

void BpxNode::srvStand(TrigReq, TrigRes res) {
    robot_.setStandUp();
    res->success = true;
    res->message = "stand command sent";
}

void BpxNode::srvSit(TrigReq, TrigRes res) {
    robot_.setSitDown();
    res->success = true;
    res->message = "sit command sent";
}

void BpxNode::srvDamp(TrigReq, TrigRes res) {
    robot_.setDamping();
    res->success = true;
    res->message = "damp command sent";
}

void BpxNode::srvZeroPosition(TrigReq, TrigRes res) {
    robot_.setZeroPositionsFlag();
    res->success = true;
    res->message = "zero position sent — ensure all feet are on the ground";
}

} // namespace bpx_driver

// ── Entry point ───────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<bpx_driver::BpxNode>());
    rclcpp::shutdown();
    return 0;
}
