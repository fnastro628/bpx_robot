#pragma once

#include <array>
#include <string>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/battery_state.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <std_msgs/msg/u_int8.hpp>
#include <std_srvs/srv/trigger.hpp>

#include "motion_level_control.h"

namespace bpx_driver {

constexpr int kNumJoints = 12;

class BpxNode : public rclcpp::Node {
public:
    explicit BpxNode(const rclcpp::NodeOptions& opts = rclcpp::NodeOptions());
    ~BpxNode();

private:
    // ── Periodic state publish ───────────────────────────────────────────────
    void publishState();

    // ── Subscriptions ────────────────────────────────────────────────────────
    void onCmdVel(geometry_msgs::msg::Twist::SharedPtr msg);

    // ── Service handlers ─────────────────────────────────────────────────────
    using TrigReq = std_srvs::srv::Trigger::Request::SharedPtr;
    using TrigRes = std_srvs::srv::Trigger::Response::SharedPtr;
    void srvStand(TrigReq, TrigRes res);
    void srvSit(TrigReq, TrigRes res);
    void srvDamp(TrigReq, TrigRes res);
    void srvZeroPosition(TrigReq, TrigRes res);

    // ── SDK ──────────────────────────────────────────────────────────────────
    bpx_sdk::MotionLevelControl robot_;

    std::array<std::string, kNumJoints> joint_names_;

    // ── Publishers ───────────────────────────────────────────────────────────
    rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr   joint_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Imu>::SharedPtr          imu_pub_;
    rclcpp::Publisher<sensor_msgs::msg::BatteryState>::SharedPtr battery_pub_;
    rclcpp::Publisher<std_msgs::msg::UInt8>::SharedPtr           motion_state_pub_;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr        odom_pub_;

    // ── Subscribers ──────────────────────────────────────────────────────────
    rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr cmd_vel_sub_;

    // ── Services ─────────────────────────────────────────────────────────────
    rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr stand_srv_;
    rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr sit_srv_;
    rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr damp_srv_;
    rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr zero_pos_srv_;

    // ── Timer ────────────────────────────────────────────────────────────────
    rclcpp::TimerBase::SharedPtr state_timer_;
};

} // namespace bpx_driver
