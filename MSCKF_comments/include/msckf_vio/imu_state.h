/*
 * COPYRIGHT AND PERMISSION NOTICE
 * Penn Software MSCKF_VIO
 * Copyright (C) 2017 The Trustees of the University of Pennsylvania
 * All rights reserved.
 */

#ifndef MSCKF_VIO_IMU_STATE_H
#define MSCKF_VIO_IMU_STATE_H

#include <map>
#include <vector>
#include <Eigen/Dense>
#include <Eigen/Geometry>

#define GRAVITY_ACCELERATION 9.81

namespace msckf_vio
{

/*
 * @brief IMUState State for IMU
 */
struct IMUState
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    typedef long long int StateIDType;

    // An unique identifier for the IMU state.
    StateIDType id;

    // id for next IMU state
    static StateIDType next_id;

    // Time when the state is recorded
    double time;

    // Orientation
    // Take a vector from the world frame to
    // the IMU (body) frame.
    // Rbw
    Eigen::Vector4d orientation;

    //2023-04-30 对状态新增姿态角的记录
    Eigen::Vector3d AttN;

    // Position of the IMU (body) frame
    // in the world frame.
    // twb
    Eigen::Vector3d position;

    // Velocity of the IMU (body) frame
    // in the world frame.
    Eigen::Vector3d velocity;

    // Bias for measured angular velocity
    // and acceleration.
    Eigen::Vector3d gyro_bias;
    Eigen::Vector3d acc_bias;
    // 2023-05-01 新增符合松组合的零偏项（3项）  // 增加gyrosOffset：载体在静止情况下的角速度误差估计
    Eigen::Vector3d gyros_bias;
    Eigen::Vector3d gyros_markov_bias;
    Eigen::Vector3d gyrosOffset;
    Eigen::Vector3d acc_markov_bias;

    // Transformation between the IMU and the
    Eigen::Matrix3d R_imu_cam0;
    Eigen::Vector3d t_cam0_imu;

    // These three variables should have the same physical
    // interpretation with `orientation`, `position`, and
    // `velocity`. There three variables are used to modify
    // the transition matrices to make the observability matrix
    // have proper null space.
    // 用于使可观测性矩阵具有适当的零空间
    Eigen::Vector4d orientation_null;
    Eigen::Vector3d position_null;
    Eigen::Vector3d velocity_null;
    Eigen::Vector3d AttN_null;// 新增姿态的零空间数据设置

    // Process noise
    static double gyro_noise;
    static double acc_noise;
    static double gyro_bias_noise;
    static double acc_bias_noise;

    // 2023-05-02 新增初始参数的构建，便于数据导入
    Eigen::Vector3d attitude_uncertainty;       // 姿态不确定性(deg)   3,3,3
    Eigen::Vector3d velocity_uncertainty;       // 速度不确定性(m/s) 1E-7
    Eigen::Vector3d position_uncertainty; // 位置不确定性(m)  10
    Eigen::Vector3d gyros_uncertainty;       // 陀螺零偏不确定性(deg/s)  1E-7
    Eigen::Vector3d gyros_markov_uncertainty;// 陀螺一阶马尔可夫不确定性(deg/s)  0.01
    Eigen::Vector3d accel_markov_uncertainty;// 加速度计一阶马尔可夫不确定性(m/s^2)  0.000001
    Eigen::Vector3d gyros_noise_std;            // 陀螺零偏噪声密度(deg/s^{3/2})    1E-7
    Eigen::Vector3d gyros_markov_noise_std;     // 陀螺一阶马尔可夫标噪声准差(deg/s)   1E-6
    Eigen::Vector3d accel_markov_noise_std;     // 加速度计一阶马尔可夫噪声标准差(m/s^2)   0.000001
    Eigen::Vector3d Ta;                      // 加速度 一阶马尔可夫相关时间（与IMU仿真同）  10
    Eigen::Vector3d Tg;                      // 陀螺   一阶马尔可夫相关时间（与IMU仿真同）  10
    Eigen::Vector3d posN_meas_noise_std;        // GPS位置观测不确定性(m)  5,5,0.0002

    // 2023-05-04 新增初始位置的记录，用于后续经纬高到ENU位置的转换
    Eigen::Vector3d init_att_618Dpro;
    Eigen::Vector3d init_vel_618Dpro;
    Eigen::Vector3d init_pos_618Dpro;


    // Gravity vector in the world frame
    static Eigen::Vector3d gravity;

    // Transformation offset from the IMU frame to
    // the body frame. The transformation takes a
    // vector from the IMU frame to the body frame.
    // The z axis of the body frame should point upwards.
    // Normally, this transform should be identity.
    static Eigen::Isometry3d T_imu_body;

    IMUState() 
        : id(0), time(0),
        orientation(Eigen::Vector4d(0, 0, 0, 1)),
        position(Eigen::Vector3d::Zero()),
        velocity(Eigen::Vector3d::Zero()),
        gyro_bias(Eigen::Vector3d::Zero()),
        acc_bias(Eigen::Vector3d::Zero()),
        orientation_null(Eigen::Vector4d(0, 0, 0, 1)),
        position_null(Eigen::Vector3d::Zero()),
        velocity_null(Eigen::Vector3d::Zero()),
        AttN(Eigen::Vector3d::Zero()),         // 2023-04-30 新增框架内容
        AttN_null(Eigen::Vector3d::Zero()) {}

    IMUState(const StateIDType &new_id)
        : id(new_id), time(0),
        orientation(Eigen::Vector4d(0, 0, 0, 1)),
        position(Eigen::Vector3d::Zero()),
        velocity(Eigen::Vector3d::Zero()),
        gyro_bias(Eigen::Vector3d::Zero()),
        acc_bias(Eigen::Vector3d::Zero()),
        orientation_null(Eigen::Vector4d(0, 0, 0, 1)), //零空间的参数
        position_null(Eigen::Vector3d::Zero()),
        velocity_null(Eigen::Vector3d::Zero()),
        AttN(Eigen::Vector3d::Zero()),          // 2023-04-30 新增框架内容
        AttN_null(Eigen::Vector3d::Zero()) {} 
};

typedef IMUState::StateIDType StateIDType;

} // namespace msckf_vio

#endif // MSCKF_VIO_IMU_STATE_H
