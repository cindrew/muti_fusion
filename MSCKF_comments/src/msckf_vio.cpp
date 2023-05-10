/*
 * COPYRIGHT AND PERMISSION NOTICE
 * Penn Software MSCKF_VIO
 * Copyright (C) 2017 The Trustees of the University of Pennsylvania
 * All rights reserved.
 */

#include <iostream>
#include <iomanip>
#include <cmath>
#include <iterator>
#include <algorithm>

#include <Eigen/SVD>
#include <Eigen/QR>
#include <Eigen/SparseCore>
#include <Eigen/SPQRSupport>
#include <boost/math/distributions/chi_squared.hpp>
#include <sensor_msgs/NavSatFix.h>
#include <eigen_conversions/eigen_msg.h>
#include <tf_conversions/tf_eigen.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>

#include <nav_msgs/Path.h>

#include <msckf_vio/msckf_vio.h>
#include <msckf_vio/math_utils.hpp>
#include <msckf_vio/utils.h>

using namespace std;
using namespace Eigen;

namespace msckf_vio
{
// Static member variables in IMUState class.
StateIDType IMUState::next_id = 0;
double IMUState::gyro_noise = 0.001;
double IMUState::acc_noise = 0.01;
double IMUState::gyro_bias_noise = 0.001;
double IMUState::acc_bias_noise = 0.01;
Vector3d IMUState::gravity = Vector3d(0, 0, -GRAVITY_ACCELERATION);
Isometry3d IMUState::T_imu_body = Isometry3d::Identity();

// Static member variables in CAMState class.
Isometry3d CAMState::T_cam0_cam1 = Isometry3d::Identity();

// Static member variables in Feature class.
FeatureIDType Feature::next_id = 0;
double Feature::observation_noise = 0.01;
Feature::OptimizationConfig Feature::optimization_config;

map<int, double> MsckfVio::chi_squared_test_table;

// MsckfVio::MsckfVio(ros::NodeHandle &pnh) : is_gravity_set(false), is_first_img(true), nh(pnh)
// {
//     return;
// }

// 2023-04-30 重新设定参数初始化架构
MsckfVio::MsckfVio(ros::NodeHandle& pnh):
    is_gravity_set(false),
    is_first_img(true),
    // is_aligned(false),
    // last_check_GPS_time(0),
    // 2023-04-30 新增判断是否初始化完成（也就是参考值是否赋值作为初值）
    is_state_initial(false),
    nh(pnh) {
    return;
}

/**
 * @brief 导入各种参数，包括阈值、传感器误差标准差等
 */
bool MsckfVio::loadParameters()
{
    // Frame id
    // 坐标系名字
    nh.param<string>("fixed_frame_id", fixed_frame_id, "world");
    nh.param<string>("child_frame_id", child_frame_id, "robot");

    nh.param<bool>("publish_tf", publish_tf, true);
    nh.param<double>("frame_rate", frame_rate, 40.0);
    // 用于判断状态是否发散
    nh.param<double>("position_std_threshold", position_std_threshold, 8.0);

    // 判断是否删除状态
    nh.param<double>("rotation_threshold", rotation_threshold, 0.2618);
    nh.param<double>("translation_threshold", translation_threshold, 0.4);
    nh.param<double>("tracking_rate_threshold", tracking_rate_threshold, 0.5);

    // Feature optimization parameters
    // 判断点是否能够做三角化，这个参数用的非常精彩
    nh.param<double>(
        "feature/config/translation_threshold",
        Feature::optimization_config.translation_threshold, 0.2);

    // Noise related parameters
    // imu参数
    nh.param<double>("noise/gyro", IMUState::gyro_noise, 0.001);
    nh.param<double>("noise/acc", IMUState::acc_noise, 0.01);
    nh.param<double>("noise/gyro_bias", IMUState::gyro_bias_noise, 0.001);
    nh.param<double>("noise/acc_bias", IMUState::acc_bias_noise, 0.01);
    // 特征的噪声
    nh.param<double>("noise/feature", Feature::observation_noise, 0.01);

    // Use variance instead of standard deviation.
    // 方差
    IMUState::gyro_noise *= IMUState::gyro_noise;
    IMUState::acc_noise *= IMUState::acc_noise;
    IMUState::gyro_bias_noise *= IMUState::gyro_bias_noise;
    IMUState::acc_bias_noise *= IMUState::acc_bias_noise;
    Feature::observation_noise *= Feature::observation_noise;

    // Set the initial IMU state.
    // The intial orientation and position will be set to the origin
    // implicitly. But the initial velocity and bias can be
    // set by parameters.
    // TODO: is it reasonable to set the initial bias to 0?
    // 设置初始化速度为0，那必须一开始处于静止了，也符合msckf的静态初始化了
    nh.param<double>("initial_state/velocity/x", state_server.imu_state.velocity(0), 0.0);
    nh.param<double>("initial_state/velocity/y", state_server.imu_state.velocity(1), 0.0);
    nh.param<double>("initial_state/velocity/z", state_server.imu_state.velocity(2), 0.0);

    // The initial covariance of orientation and position can be
    // set to 0. But for velocity, bias and extrinsic parameters,
    // there should be nontrivial uncertainty.
    // 初始协方差的赋值（误差状态的协方差）
    // 为什么旋转平移就可以是0？因为在正式开始之前我们通过初始化找好了重力方向，确定了第一帧的位姿
    double gyro_bias_cov, acc_bias_cov, velocity_cov;
    nh.param<double>("initial_covariance/velocity", velocity_cov, 0.25);
    nh.param<double>("initial_covariance/gyro_bias", gyro_bias_cov, 1e-4);
    nh.param<double>("initial_covariance/acc_bias", acc_bias_cov, 1e-2);

    double extrinsic_rotation_cov, extrinsic_translation_cov;
    nh.param<double>("initial_covariance/extrinsic_rotation_cov", extrinsic_rotation_cov, 3.0462e-4);
    nh.param<double>("initial_covariance/extrinsic_translation_cov", extrinsic_translation_cov, 1e-4);

    // 2023-04-30 使用组合的参数
    // 松组合系统参数设定
    state_server.imu_state.attitude_uncertainty<<1E-1,1E-1,1E-1;       // 姿态不确定性(deg)   3,3,3
    state_server.imu_state.velocity_uncertainty<<1E-2,1E-2,1E-2;       // 速度不确定性(m/s) 1E-7
    state_server.imu_state.position_uncertainty<<1E0,1E0,1E0; // 位置不确定性(m)  10
    state_server.imu_state.gyros_uncertainty<<1E-7,1E-7,1E-7;       // 陀螺零偏不确定性(deg/s)  1E-7
    state_server.imu_state.gyros_markov_uncertainty<<0.01,0.01,0.01;// 陀螺一阶马尔可夫不确定性(deg/s)  0.01
    state_server.imu_state.accel_markov_uncertainty<<1E-7,1E-7,1E-7;// 加速度计一阶马尔可夫不确定性(m/s^2)  0.000001
    state_server.imu_state.gyros_noise_std<<1E-7,1E-7,1E-7;            // 陀螺零偏噪声密度(deg/s^{3/2})    1E-7
    state_server.imu_state.gyros_markov_noise_std<<1E-5,1E-5,1E-5;     // 陀螺一阶马尔可夫标噪声准差(deg/s)   1E-6
    state_server.imu_state.accel_markov_noise_std<<1E-5,1E-5,1E-5;     // 加速度计一阶马尔可夫噪声标准差(m/s^2)   0.000001
    state_server.imu_state.Ta<<1800,1800,1800;                      // 加速度 一阶马尔可夫相关时间（与IMU仿真同）  10
    state_server.imu_state.Tg<<3600,3600,3600;                      // 陀螺   一阶马尔可夫相关时间（与IMU仿真同）  10
    state_server.imu_state.posN_meas_noise_std<<20,20,0.0002;        // GPS位置观测不确定性(m)  5,5,0.0002

    double attitude_uncertainty_num, velocity_uncertainty_num, position_uncertainty_num;
    double gyros_uncertainty_num, gyros_markov_uncertainty_num, accel_markov_uncertainty_num;   // 注意这是用于修改kalman中对应imu噪声阵Q阵的参数，后面需要和真实器件参数统一
    double gyros_noise_std_num, gyros_markov_noise_std_num, accel_markov_noise_std_num;
    double Ta_num, Tg_num, posN_meas_noise_std_x_num, posN_meas_noise_std_y_num, posN_meas_noise_std_z_num;

    // 赋值
    nh.param<double>("initial_covariance/attitude_loose", attitude_uncertainty_num, state_server.imu_state.attitude_uncertainty[0]);
    nh.param<double>("initial_covariance/velocity_loose", velocity_uncertainty_num, state_server.imu_state.velocity_uncertainty[0]);
    nh.param<double>("initial_covariance/position_loose", position_uncertainty_num, state_server.imu_state.position_uncertainty[0]);

    nh.param<double>("initial_covariance/gyros_uncertainty_loose", gyros_uncertainty_num, state_server.imu_state.gyros_uncertainty[0]);
    nh.param<double>("initial_covariance/gyros_markov_uncertainty_loose", gyros_markov_uncertainty_num, state_server.imu_state.gyros_markov_uncertainty[0]);
    nh.param<double>("initial_covariance/accel_markov_uncertainty_loose", accel_markov_uncertainty_num, state_server.imu_state.accel_markov_uncertainty[0]);

    nh.param<double>("initial_covariance/gyros_uncertainty_loose", gyros_noise_std_num, state_server.imu_state.gyros_noise_std[0]);
    nh.param<double>("initial_covariance/gyros_markov_uncertainty_loose", gyros_markov_noise_std_num, state_server.imu_state.gyros_markov_noise_std[0]);
    nh.param<double>("initial_covariance/accel_markov_uncertainty_loose", accel_markov_noise_std_num, state_server.imu_state.accel_markov_noise_std[0]);

    nh.param<double>("initial_covariance/Ta_loose", Ta_num, state_server.imu_state.Ta[0]);
    nh.param<double>("initial_covariance/Tg_loose", Tg_num, state_server.imu_state.Tg[0]);
    nh.param<double>("initial_covariance/posN_meas_noise_std_loose/x", posN_meas_noise_std_x_num, state_server.imu_state.posN_meas_noise_std[0]);
    nh.param<double>("initial_covariance/posN_meas_noise_std_loose/y", posN_meas_noise_std_y_num, state_server.imu_state.posN_meas_noise_std[1]);
    nh.param<double>("initial_covariance/posN_meas_noise_std_loose/z", posN_meas_noise_std_z_num, state_server.imu_state.posN_meas_noise_std[2]);


    // // 0~3 角度
    // // 3~6 陀螺仪偏置
    // // 6~9 速度 
    // // 9~12 加速度计偏置
    // // 12~15 位移
    // // 15~18 外参旋转
    // // 18~21 外参平移
    // state_server.state_cov = MatrixXd::Zero(21, 21);
    // for (int i = 3; i < 6; ++i)
    //     state_server.state_cov(i, i) = gyro_bias_cov;
    // for (int i = 6; i < 9; ++i)
    //     state_server.state_cov(i, i) = velocity_cov;
    // for (int i = 9; i < 12; ++i)
    //     state_server.state_cov(i, i) = acc_bias_cov;
    // for (int i = 15; i < 18; ++i)
    //     state_server.state_cov(i, i) = extrinsic_rotation_cov;
    // for (int i = 18; i < 21; ++i)
    //     state_server.state_cov(i, i) = extrinsic_translation_cov;

    // 2023-04-30 重新构建18维状态量用于IMU/GPS松组合处理
    // 0~3 角度(deg) 
    // 3~6 速度(m/s)
    // 6~9 位置(m) 
    // 9~12 陀螺零偏不确定性(deg/s)
    // 12~15 陀螺一阶马尔可夫不确定性(deg/s)
    // 15~18 加速度计一阶马尔可夫不确定性
    state_server.state_cov = MatrixXd::Zero(18, 18);
    for (int i = 0; i < 3; ++i)
        state_server.state_cov(i, i) = attitude_uncertainty_num;
    for (int i = 3; i < 6; ++i)
        state_server.state_cov(i, i) = velocity_uncertainty_num;
    for (int i = 6; i < 9; ++i)
        state_server.state_cov(i, i) = position_uncertainty_num;
    for (int i = 9; i < 12; ++i)
        state_server.state_cov(i, i) = gyros_uncertainty_num;
    for (int i = 12; i < 15; ++i)
        state_server.state_cov(i, i) = gyros_markov_uncertainty_num;
    for (int i = 15; i < 18; ++i)
        state_server.state_cov(i, i) = accel_markov_uncertainty_num;


    std::stringstream ss_R0;
    ss_R0 << state_server.state_cov;
    ROS_INFO("=====================state_server.state_cov========================");
    ROS_INFO("Matrix:\n%s", ss_R0.str().c_str());
    ROS_INFO("==========================over================================");


    // Transformation offsets between the frames involved.
    // 外参注意还是从左往右那么看
    Isometry3d T_imu_cam0 = utils::getTransformEigen(nh, "cam0/T_cam_imu");
    Isometry3d T_cam0_imu = T_imu_cam0.inverse();

    // 关于外参状态的初始值设置
    state_server.imu_state.R_imu_cam0 = T_cam0_imu.linear().transpose();
    state_server.imu_state.t_cam0_imu = T_cam0_imu.translation();

    // 一些其他外参
    CAMState::T_cam0_cam1 =
        utils::getTransformEigen(nh, "cam1/T_cn_cnm1");
    IMUState::T_imu_body =
        utils::getTransformEigen(nh, "T_imu_body").inverse();

    // Maximum number of camera states to be stored
    nh.param<int>("max_cam_state_size", max_cam_state_size, 30);

    // 剩下的都是打印的东西了
    ROS_INFO("===========================================");
    ROS_INFO("fixed frame id: %s", fixed_frame_id.c_str());
    ROS_INFO("child frame id: %s", child_frame_id.c_str());
    ROS_INFO("publish tf: %d", publish_tf);
    ROS_INFO("frame rate: %f", frame_rate);
    ROS_INFO("position std threshold: %f", position_std_threshold);
    ROS_INFO("Keyframe rotation threshold: %f", rotation_threshold);
    ROS_INFO("Keyframe translation threshold: %f", translation_threshold);
    ROS_INFO("Keyframe tracking rate threshold: %f", tracking_rate_threshold);
    ROS_INFO("gyro noise: %.10f", IMUState::gyro_noise);
    ROS_INFO("gyro bias noise: %.10f", IMUState::gyro_bias_noise);
    ROS_INFO("acc noise: %.10f", IMUState::acc_noise);
    ROS_INFO("acc bias noise: %.10f", IMUState::acc_bias_noise);
    ROS_INFO("observation noise: %.10f", Feature::observation_noise);
    ROS_INFO("initial velocity: %f, %f, %f",
        state_server.imu_state.velocity(0),
        state_server.imu_state.velocity(1),
        state_server.imu_state.velocity(2));
    ROS_INFO("initial gyro bias cov: %f", gyro_bias_cov);
    ROS_INFO("initial acc bias cov: %f", acc_bias_cov);
    ROS_INFO("initial velocity cov: %f", velocity_cov);
    ROS_INFO("initial extrinsic rotation cov: %f", extrinsic_rotation_cov);
    ROS_INFO("initial extrinsic translation cov: %f", extrinsic_translation_cov);

    cout << T_imu_cam0.linear() << endl;
    cout << T_imu_cam0.translation().transpose() << endl;

    ROS_INFO("max camera state #: %d", max_cam_state_size);
    ROS_INFO("===========================================");

    // 加入松组合参数的打印
    ROS_INFO("====================GPS/IMU loosely coupled=======================");
    ROS_INFO("initial attitude cov: %f", attitude_uncertainty_num);
    ROS_INFO("initial velocity cov: %f", velocity_uncertainty_num);
    ROS_INFO("initial position cov: %f", position_uncertainty_num);
    ROS_INFO("initial gyro bias cov: %f", gyros_uncertainty_num);
    ROS_INFO("initial gyro markov bias cov: %f", gyros_markov_uncertainty_num);
    ROS_INFO("initial acc markov cov: %f", accel_markov_uncertainty_num);
    ROS_INFO("==========================over================================");


    return true;
}

bool MsckfVio::createRosIO()
{
    // 发送位姿信息与三维点
    odom_pub = nh.advertise<nav_msgs::Odometry>("odom", 10);
    feature_pub = nh.advertise<sensor_msgs::PointCloud2>("feature_point_cloud", 10);

    // 重置
    reset_srv = nh.advertiseService("reset", &MsckfVio::resetCallback, this);

    // 接收imu数据与前端跟踪的特征
    imu_sub = nh.subscribe("imu", 100, &MsckfVio::imuCallback, this);
    feature_sub = nh.subscribe("features", 40, &MsckfVio::featureCallback, this);

    // 接受真值，动作捕捉发来的
    mocap_odom_sub = nh.subscribe("mocap_odom", 10, &MsckfVio::mocapOdomCallback, this);
    mocap_odom_pub = nh.advertise<nav_msgs::Odometry>("gt_odom", 1);

    // 2023-04-26 接受GPS的真值数据
    gps_sub=nh.subscribe("gps_position", 10, & MsckfVio::GPSCallback, this);
    // gps_vel_sub=nh.subscribe("gps_velocity", 10, & MsckfVio::GPSVelCallback, this);

        // 2023-04-27 定义好pub，用于数据的输出
    // gps_pub=nh.advertise<sensor_msgs::NavSatFix>("gps_pos", 10);
    // gps_vel_pub=nh.advertise<forsense_msg::RtkVelocity>("gps_vel", 10);

    // 2023-04-29 基于旧的消息重新定义好数据的发布
    gps_pub=nh.advertise<forsense_msg::Forsense>("gps_pos", 10000);
    gps_vel_pub=nh.advertise<forsense_msg::Forsense>("gps_vel", 10000);

    // 2023-04-29 基于旧的消息加入新的参考值的数据发布
    // Tru_pub = nh.advertise<forsense_msg::Forsense>("Tru_pso",10000);
    Tru_vel_pub = nh.advertise<forsense_msg::Forsense>("Tru_vel",10000);

    // 2023-05-02 将松组合解算的轨迹进行输出
    path_pub = nh.advertise<nav_msgs::Path>("path", 1000);
    // Tru_pub = nh.advertise<nav_msgs::Odometry>("Tru_path",10000);
    Tru_pub = nh.advertise<nav_msgs::Path>("Tru_path",10000);

    return true;
}


bool MsckfVio::initialize()
{
    // 1. 加载参数
    if (!loadParameters())
        return false;
    ROS_INFO("Finish loading ROS parameters...");

    // Initialize state server
    // imu观测的协方差     
    // To do list1:后续需要改这里的IMU传递噪声阵
    state_server.continuous_noise_cov =
        Matrix<double, 12, 12>::Zero();
    state_server.continuous_noise_cov.block<3, 3>(0, 0) =
        Matrix3d::Identity() * IMUState::gyro_noise;
    state_server.continuous_noise_cov.block<3, 3>(3, 3) =
        Matrix3d::Identity() * IMUState::gyro_bias_noise;
    state_server.continuous_noise_cov.block<3, 3>(6, 6) =
        Matrix3d::Identity() * IMUState::acc_noise;
    state_server.continuous_noise_cov.block<3, 3>(9, 9) =
        Matrix3d::Identity() * IMUState::acc_bias_noise;

    kalmanFilterInitializaion();                    // 第0步：kalman滤波器的构建


    // 卡方检验表
    // Initialize the chi squared test table with confidence
    // level 0.95.
    for (int i = 1; i < 100; ++i)
    {
        boost::math::chi_squared chi_squared_dist(i);
        chi_squared_test_table[i] =
            boost::math::quantile(chi_squared_dist, 0.05);
    }

    // 2. 接收与发布
    if (!createRosIO())
        return false;
    ROS_INFO("Finish creating ROS IO...");

    return true;
}

// 2023-04-26 基于新消息类型定义下的GPS数据提取
// // GPS fusion part
// void MsckfVio::GPSCallback(const sensor_msgs::NavSatFix::Ptr & gps_msg)
// {
//     ROS_INFO("START GPS DATA...");
//     double time = gps_msg->header.stamp.toSec();
//     sensor_msgs::NavSatFix GNSSposi;
//     GNSSposi.longitude=gps_msg->longitude;
//     GNSSposi.latitude=gps_msg->latitude;
//     GNSSposi.altitude=gps_msg->altitude;
//     // auto msg_modified = msg;
//     // msg_modified.altitude = 21.0;
//     // GPS_buffer.push_back(make_pair(time,msg_modified));
//     GPS_buffer.push_back(make_pair(time,GNSSposi));
//     // state_server.imu_state.R_n_e = geo_utils::CoordTrans::getRne(Vector3d(msg_modified.latitude,
//     //                                                                       msg_modified.longitude,
//     //                                                                       msg_modified.altitude)).transpose();
//     /*
//     ofsgps<<setprecision(20)<<time<<" "<<
//             setprecision(15)<<msg_modified.latitude<<" "<<
//             setprecision(15)<<msg_modified.longitude<<" "<<
//             setprecision(15)<<msg_modified.altitude<<" "<<endl;
//             */
//     gps_pub.publish(GNSSposi);
//     ROS_INFO("SUCCESS GPS OUTPUT...");

// }
// 
// // void MsckfVio::GPSVelCallback(const geometry_msgs::Twist::Ptr & velo_msg)
// void MsckfVio::GPSVelCallback(const forsense_msg::RtkVelocity::Ptr & velo_msg)
// {    
//     double time = velo_msg->header.stamp.toSec();
//     forsense_msg::RtkVelocity GNSSvelo;
//     GNSSvelo.twist.linear.x = velo_msg->twist.linear.x;
//     GNSSvelo.twist.linear.y = velo_msg->twist.linear.y;
//     GNSSvelo.twist.linear.z = velo_msg->twist.linear.z;

//     GPS_vel_buffer.push_back(make_pair(time,GNSSvelo));

//     gps_vel_pub.publish(GNSSvelo);
//     ROS_INFO("SUCCESS GPS OUTPUT...");

// }


// 2023-04-29 基于旧的消息类型定义下的GPS数据提取代码
// GPS fusion part
void MsckfVio::GPSCallback(const forsense_msg::Forsense::Ptr & gps_msg)
{
    // ROS_INFO("START GPS DATA...");
    // double time = gps_msg->header.stamp.toSec();
    // 2023-04-29 注意由于原有的消息类型中没有加入ROS时间戳的记录，所以只能用自带的UTC/GPS时间进行处理
    unsigned int time = gps_msg->time_gps_ms;
    double time_trans = static_cast<double>(time)/1000.0;

    cout<<"====================Time run here==========="<<__LINE__<<endl;
    ROS_INFO("===================size Zero_Second========================");
    ROS_INFO("my_time show 1: %f ", time_trans);
    ROS_INFO("==========================over================================");

    // ---------------------------------------------------------------------------
    // 2023-04-30 继续进行初始化处理，状态传递，状态观测，修正处理，状态输出等部分内容的处理

    // 1、进行状态初始化处理(参考值传输过来实现)
    forsense_msg::Forsense msg_copy;
    msg_copy=(*gps_msg);
    // 实现对IMU参数的记录
    msg_copy.accel = gps_msg->accel;
    msg_copy.gyro = gps_msg->gyro;

    IMU_couter_buffer.push_back(msg_copy);

    forsense_msg::Forsense msg_copy2;
    msg_copy2.time_gps_ms = gps_msg->time_gps_ms;
    msg_copy2.pos = gps_msg->pos;
    msg_copy2.vel = gps_msg->vel;
    msg_copy2.att = gps_msg->att;

    // cout<<"====================Time run here==========="<<__LINE__<<endl;
    // ROS_INFO("===================size Zero_Second========================");
    // ROS_INFO("my_time show 1.1: %d ", msg_copy2.time_gps_ms);
    // ROS_INFO("==========================over================================");

    Tru_couter_buffer.push_back(msg_copy2);
   
    
    static int dataCount;                                           // 计数，用于观测修正
    if (!is_state_initial) {
        if (IMU_couter_buffer.size() < 50) return;
        // INS_Inialization();
        // 这里是直接将姿态、速度、位置直接拿出来进行处理
        Vector3d atti_conference = Vector3d::Zero();
        Vector3d vel_conference = Vector3d::Zero();
        Vector3d posi_conference = Vector3d::Zero();

        Vector3d init_att = Vector3d::Zero();
        Vector3d init_vel = Vector3d::Zero();
        Vector3d init_pos = Vector3d::Zero();

        double m_time_618Dpro, m_time_618Dpro_s;                            // 原始的618Dpro记载的缺少了ROS时间戳，所以这里是拿GPS时间来处理

        // 所以这里不应该拿IMU的速度，而应该拿参考值部分的消息内容
        for (const auto &gps_msg : Tru_couter_buffer)
        {

            // 使用自定义的消息类型转换（针对618Dpro旧的数据定义）
            MsckfVio::ForsenseMsgAttiToEigen(gps_msg, init_att);
            MsckfVio::ForsenseMsgVeloToEigen(gps_msg, init_vel);
            MsckfVio::ForsenseMsgPosiToEigen(gps_msg, init_pos);

            MsckfVio::ForsenseMsgTimeToDouble(gps_msg, m_time_618Dpro);

        }

        // 得出姿态，速度，位置（相当于最后的参考值作为初始化处理）
        state_server.imu_state.AttN = init_att;
        // state_server.imu_state.AttN << init_att[0],init_att[1], 360.0 - init_att[2]; // 618Dpro定义的航向角符合顺时针为+，所以不需要进行转换  --2023-05-03
        // state_server.imu_state.velocity = init_vel; 
        state_server.imu_state.velocity << init_vel[1],init_vel[0], -init_vel[2]; // 注意618Dpro录制的是ned三方向速度，这里需要改成ENU三方向的速度
        state_server.imu_state.position = init_pos;
        // state_server.imu_state.position << init_pos[1]/1E7,init_pos[0]/1E7,init_pos[2]/1E3; //统一格式为经纬高（deg,deg,m）

        // 2023-05-04 同步将初值进行记录
        state_server.imu_state.init_att_618Dpro = init_att;
        state_server.imu_state.init_vel_618Dpro << init_vel[1],init_vel[0], -init_vel[2];
        // state_server.imu_state.init_pos_618Dpro << init_pos[1]/1E7,init_pos[0]/1E7,init_pos[2]/1E3;
        state_server.imu_state.init_pos_618Dpro << init_pos;

        // 同时给IMU进行时间赋值，这样便于后续的IMU迭代
        // m_time_618Dpro_s = m_time_618Dpro/1000.0;
        m_time_618Dpro_s = m_time_618Dpro; //时间重新在ForsenseMsgTimeToDouble转换完成，所以这里不用转换了
        state_server.imu_state.time = m_time_618Dpro_s; // 统一单位为s
        cout<<"====================Time run here==========="<<__LINE__<<endl;
        ROS_INFO("===================size Zero_Second========================");
        ROS_INFO("my_time show 2: %f ", m_time_618Dpro_s);
        ROS_INFO("==========================over================================");

        ROS_INFO("===============START IMU INIT========================");

        // 由于显示精度的问题，这里用cout
        std::cout<<setprecision(20)<<"initial attitude : attix="<<state_server.imu_state.AttN[0]<<",attiy="<<state_server.imu_state.AttN[1]<<",attiz="<<state_server.imu_state.AttN[2]<<std::endl;  
        std::cout<<setprecision(20)<<"initial velocity : velox="<<state_server.imu_state.velocity[0]<<",veloy="<<state_server.imu_state.velocity[1]<<",veloz="<<state_server.imu_state.velocity[2]<<std::endl;  
        std::cout<<setprecision(20)<<"initial position : posix="<<state_server.imu_state.position[0]<<",posiy="<<state_server.imu_state.position[1]<<",posiz="<<state_server.imu_state.position[2]<<std::endl;  
        std::cout<<setprecision(20)<<"initial time : time="<<state_server.imu_state.time<<std::endl;  
        
        ROS_INFO("=====================end=============================");
        is_state_initial = true;

        dataCount = 0;                                           
    }
    else
    {

        // 618Dpro中消息的读取
        MsckfVio::ForsenseMsgGyroToEigen(msg_copy, m_gyro_618Dpro);
        MsckfVio::ForsenseMsgAccToEigen(msg_copy, m_acc_618Dpro);
        MsckfVio::ForsenseMsgTimeToDouble(msg_copy, m_time_618Dpro);
        // 将时间也记录下来
        // m_time_618Dpro_s = m_time_618Dpro/1000.0;
        m_time_618Dpro_s = m_time_618Dpro; //时间重新在ForsenseMsgTimeToDouble转换完成，所以这里不用转换了

        cout<<"====================Time run here==========="<<__LINE__<<endl;
        ROS_INFO("===================size Zero_Second========================");
        ROS_INFO("my_time show 3: %f ", m_time_618Dpro_s);
        ROS_INFO("last_my_time show 3: %f ", state_server.imu_state.time);
        ROS_INFO("==========================over================================");

        double g=9.7803698;         // 重力加速度
        // 注意原有算法中角速度和加速度三轴是符合FLU的顺序，这里需要改成符合松组合算法的RFU三轴定义，所以对角速度和加速度进行转换处理
        m_gyro_618Dpro_RFU << -m_gyro_618Dpro[1], m_gyro_618Dpro[0], m_gyro_618Dpro[2];  // （rad/s）注意这里的z轴旋转角速度是要符合逆时针旋转为+的顺序（与航向角符合顺时针为+的顺序不同）,且符合RFU三轴的顺序定义
        m_acc_618Dpro_RFU << -m_acc_618Dpro[1], m_acc_618Dpro[0], m_acc_618Dpro[2];

        m_gyro_618Dpro_RFU_deg = m_gyro_618Dpro_RFU * rad2deg;  // （deg/s）
        m_acc_618Dpro_RFU_mss = m_acc_618Dpro_RFU * g;  // （m/s/s）
        
        // 这里是实现地理系下的IMU的参数迭代处理，主要包括IMU的零偏修正，状态更新
        // Remove the bias from the measured gyro and acceleration
        // IMUState& imu_state = state_server.imu_state;
        my_gyro = m_gyro_618Dpro_RFU_deg - state_server.imu_state.gyros_bias - state_server.imu_state.gyros_markov_bias;
        my_acc = m_acc_618Dpro_RFU_mss - state_server.imu_state.acc_markov_bias;
        my_dtime = m_time_618Dpro_s - state_server.imu_state.time;

        // Update the state info
        state_server.imu_state.time = m_time_618Dpro_s;

        cout<<"====================Time run here==========="<<__LINE__<<endl;
        ROS_INFO("===================size Zero_Second========================");
        ROS_INFO("my_time update 4: %f ", state_server.imu_state.time);
        ROS_INFO("==========================over================================");
        ROS_INFO("my_dtime: %f ", my_dtime);
        ROS_INFO("==========================over================================");

        ROS_INFO("===============navigation states ========================");
        // 由于显示精度的问题，这里用cout
        std::cout<<setprecision(20)<<"attitude : attix="<<state_server.imu_state.AttN[0]<<",attiy="<<state_server.imu_state.AttN[1]<<",attiz="<<state_server.imu_state.AttN[2]<<std::endl;  
        std::cout<<setprecision(20)<<"velocity : velox="<<state_server.imu_state.velocity[0]<<",veloy="<<state_server.imu_state.velocity[1]<<",veloz="<<state_server.imu_state.velocity[2]<<std::endl;  
        std::cout<<setprecision(20)<<"position : posix="<<state_server.imu_state.position[0]<<",posiy="<<state_server.imu_state.position[1]<<",posiz="<<state_server.imu_state.position[2]<<std::endl;  
        std::cout<<setprecision(20)<<"time : time="<<state_server.imu_state.time<<std::endl;  
        ROS_INFO("=====================end=============================");

        INS_Update();       // 第一步：利用IMU的实现运动状态更新
        
        system_model_cal();           // 第二步：构建运动模型

        //卡尔曼滤波时间更新和状态更新
        my_Xestimated = Eigen::VectorXd::Zero(18);

        ROS_INFO("===================size Zero_One========================");
        // std::cout<<"size of Pestimated: "<<Pestimated.rows()<<std::endl;
        ROS_INFO("state_server.state_cov dimensions: %d x %d", state_server.state_cov.rows(), state_server.state_cov.cols());
        ROS_INFO("==========================over================================");

        my_Pestimated = state_server.state_cov;

        ROS_INFO("===================size Zero_Second========================");
        // std::cout<<"size of Pestimated: "<<Pestimated.rows()<<std::endl;
        ROS_INFO("my_Pestimated dimensions: %d x %d", my_Pestimated.rows(), my_Pestimated.cols());
        ROS_INFO("==========================over================================");


        // ROS_INFO("===================Xestimated========================");
        // ROS_INFO("Xestimated(0): %f", Xestimated(0));
        // ROS_INFO("Xestimated(1): %f", Xestimated(1));
        // ROS_INFO("Xestimated(2): %f", Xestimated(2));
        // ROS_INFO("Xestimated(3): %f", Xestimated(3));
        // ROS_INFO("Xestimated(4): %f", Xestimated(4));
        // ROS_INFO("Xestimated(5): %f", Xestimated(5));
        // ROS_INFO("Xestimated(6): %f", Xestimated(6));
        // ROS_INFO("Xestimated(7): %f", Xestimated(7));
        // ROS_INFO("Xestimated(8): %f", Xestimated(8));
        // ROS_INFO("Xestimated(9): %f", Xestimated(9));
        // ROS_INFO("Xestimated(10): %f", Xestimated(10));
        // ROS_INFO("Xestimated(11): %f", Xestimated(11));
        // ROS_INFO("Xestimated(12): %f", Xestimated(12));
        // ROS_INFO("Xestimated(13): %f", Xestimated(13));
        // ROS_INFO("Xestimated(14): %f", Xestimated(14));
        // ROS_INFO("Xestimated(15): %f", Xestimated(15));
        // ROS_INFO("Xestimated(16): %f", Xestimated(16));
        // ROS_INFO("Xestimated(17): %f", Xestimated(17));
        // ROS_INFO("==========================over================================");

        // KF_time_update(Xestimated,Pestimated,F,Q_noise);                                                           // 第三步：基于模型实现状态,协方差的运动迭代
        
        // Eigen::MatrixXd Temp_0 = F*Pestimated*(F.transpose());
        Eigen::MatrixXd Temp_0 = my_F;

        std::stringstream ss_Temp_0;
        ss_Temp_0 << Temp_0;
        ROS_INFO("=====================ss_Temp_0========================");
        ROS_INFO("Matrix:\n%s", ss_Temp_0.str().c_str());
        ROS_INFO("==========================over================================");
        
        
        my_Xestimated = my_F*my_Xestimated;
        my_Pestimated = my_F*my_Pestimated*(my_F.transpose()) + my_Q;
        ROS_INFO("===================size One========================");
        // std::cout<<"size of Pestimated: "<<Pestimated.rows()<<std::endl;
        ROS_INFO("Pestimated dimensions: %d x %d", my_Pestimated.rows(), my_Pestimated.cols());
        ROS_INFO("==========================over================================");

        dataCount++;
        ROS_INFO("===================Xestimated========================");
        ROS_INFO("Xestimated(0): %f", my_Xestimated(0));
        ROS_INFO("Xestimated(1): %f", my_Xestimated(1));
        ROS_INFO("Xestimated(2): %f", my_Xestimated(2));
        ROS_INFO("Xestimated(3): %f", my_Xestimated(3));
        ROS_INFO("Xestimated(4): %f", my_Xestimated(4));
        ROS_INFO("Xestimated(5): %f", my_Xestimated(5));
        ROS_INFO("Xestimated(6): %f", my_Xestimated(6));
        ROS_INFO("Xestimated(7): %f", my_Xestimated(7));
        ROS_INFO("Xestimated(8): %f", my_Xestimated(8));
        ROS_INFO("Xestimated(9): %f", my_Xestimated(9));
        ROS_INFO("Xestimated(10): %f", my_Xestimated(10));
        ROS_INFO("Xestimated(11): %f", my_Xestimated(11));
        ROS_INFO("Xestimated(12): %f", my_Xestimated(12));
        ROS_INFO("Xestimated(13): %f", my_Xestimated(13));
        ROS_INFO("Xestimated(14): %f", my_Xestimated(14));
        ROS_INFO("Xestimated(15): %f", my_Xestimated(15));
        ROS_INFO("Xestimated(16): %f", my_Xestimated(16));
        ROS_INFO("Xestimated(17): %f", my_Xestimated(17));
        ROS_INFO("==========================over================================");


        if(dataCount>=10) // 100Hz频率下 1s更新一次
        {
            dataCount = 0;
            // 618Dpro中消息的读取（GPS的位置）
            MsckfVio::ForsenseMsgGpsPosiToEigen(msg_copy, m_posi_GPS_618Dpro);
            // 注意由于msg中单位没有转换过来，这里进行转换处理
            Eigen::Vector3d m_posi_GPS(m_posi_GPS_618Dpro[0]/1E7,m_posi_GPS_618Dpro[1]/1E7,m_posi_GPS_618Dpro[2]/1E3); //统一格式为经纬高（deg,deg,m）

            measur_model_cal(m_posi_GPS);                                                            // 第四步：构建观测模型

            std::stringstream ss_Pestimated;
            ss_Pestimated << my_Pestimated;
            ROS_INFO("=====================Pestimated========================");
            ROS_INFO("Matrix:\n%s", ss_Pestimated.str().c_str());
            ROS_INFO("==========================over================================");

            KF_meas_update();                                                           // 第五步：实现观测修正处理

            ROS_INFO("=====================6.1 error correction calculated complete=============================");

            // INS_Correction(PosError,VelError,AttError);
            INS_Correction(); 

            // ROS_INFO("=====================6.2 start INS correct=============================");
            // //IMU修正
            // Eigen::Vector3d PosNError(Xestimated(7)*rad2deg,Xestimated(6)*rad2deg,Xestimated(8));
            // Eigen::Vector3d VelNError = Xestimated.segment(3,3);
            // Eigen::Vector3d AttNError = Xestimated.segment(0,3);

            // ROS_INFO("=====================6.3 error correct get=============================");


            // // 飞行器速度修正
            // state_server.imu_state.velocity = state_server.imu_state.velocity - VelNError;

            // // 飞行器位置修正
            // state_server.imu_state.position = state_server.imu_state.position - PosNError;

            // roll  = state_server.imu_state.AttN(0)*deg2rad;
            // pitch = state_server.imu_state.AttN(1)*deg2rad;
            // head  = state_server.imu_state.AttN(2)*deg2rad;
            // // 坐标系C-->B(机体系－－>计算系)
            // Eigen::Matrix3d Cbc;
            // Cbc  << cos(roll)*cos(head)+sin(roll)*sin(pitch)*sin(head), -cos(roll)*sin(head)+sin(roll)*sin(pitch)*cos(head), -sin(roll)*cos(pitch),
            //         cos(pitch)*sin(head),                               cos(pitch)*cos(head),                                sin(pitch),
            //         sin(roll)*cos(head)-cos(roll)*sin(pitch)*sin(head), -sin(roll)*sin(head)-cos(roll)*sin(pitch)*cos(head), cos(roll)*cos(pitch);

            // // 修正矩阵N-->C(导航坐标系－－>计算坐标系)
            // Eigen::Matrix3d Ccn;
            // Ccn <<             1,     AttNError(2), -AttNError(1),
            //     -AttNError(2),                1,  AttNError(0),
            //         AttNError(1),    -AttNError(0),             1;
            // // 姿态矩阵修正N--->B
            // Eigen::Matrix3d Cbn = (Ccn.transpose() * Cbc.transpose()).transpose();

            // //求姿态(横滚、俯仰、航向）
            // state_server.imu_state.AttN(0) = atan(-Cbn(0,2)/Cbn(2,2))*rad2deg;
            // state_server.imu_state.AttN(1) = atan(Cbn(1,2)/sqrt(Cbn(1,0)*Cbn(1,0)+Cbn(1,1)*Cbn(1,1)))*rad2deg;
            // state_server.imu_state.AttN(2) = atan(Cbn(1,0)/Cbn(1,1))*rad2deg;

            // // 象限判断
            // if((Cbn(1,1)<0))
            //     state_server.imu_state.AttN(2) += 180.0;
            // else if(Cbn(1,0)<0)
            //     state_server.imu_state.AttN(2) += 360.0;

            // if((Cbn(2,2)<0))
            // {
            //     if(Cbn(0,2)>0)
            //         state_server.imu_state.AttN(0) = 180.0 - state_server.imu_state.AttN(0);
            //     if(Cbn(0,2)<0)
            //         state_server.imu_state.AttN(0) = -(180.0 + state_server.imu_state.AttN(0));
            // }
            // ROS_INFO("=====================6.4 INS correct finish=============================");


        }

        publish_loose(msg_copy2);                                                                                   // 第六步：相关消息的发布

    }

}


/**
 * @brief kalman滤波器初始化构建
 */
// void MsckfVio::kalmanFilterInitializaion(Eigen::MatrixXd Pestimated)
void MsckfVio::kalmanFilterInitializaion()
{
    
    // 初始状态协方差计算
    double Re  = 6378137.0;        //地球半径（米）
    double f   = 1/298.257;        //地球的椭圆率

    // 飞行器位置
    double latitude = state_server.imu_state.position(1)*deg2rad;
    double height   = state_server.imu_state.position(2);

    double Rm          = Re*( 1 - 2*f + 3*f*sin(latitude)*sin(latitude) );
    double Rn          = Re*( 1 + f*sin(latitude)*sin(latitude) );

    Eigen::Array3d position_m2rad; position_m2rad << Rm+height,(Rn+height)/cos(latitude),1;

    Eigen::VectorXd Xerr_initial = Eigen::VectorXd::Zero(18);
    Xerr_initial << state_server.imu_state.attitude_uncertainty*deg2rad,  state_server.imu_state.velocity_uncertainty,            (state_server.imu_state.position_uncertainty.array()/position_m2rad).matrix(),
                    state_server.imu_state.gyros_uncertainty*deg2rad,     state_server.imu_state.gyros_markov_uncertainty*deg2rad, state_server.imu_state.accel_markov_uncertainty;
//    double pi = 3.14159265358979;
//    Xerr_initial<<1000.0*pi/(3600.0*180.0), 1000.0*pi/(3600.0*180.0),     5*pi/180.0,
//                  0.6,                      0.6,                          0.6,
//                  20.0/(Rm+height),         20.0/(Rn+height)/cos(height), 50.0,
//                  10*pi/(3600.0*180.0),     10*pi/(3600.0*180.0),         10*pi/(3600.0*180.0),
//                  10*pi/(3600.0*180.0),     10*pi/(3600.0*180.0),         10*pi/(3600.0*180.0),
//                  0.0001,                   0.0001,                       0.0001;

    my_Pestimated = Xerr_initial.array().pow(2).matrix().asDiagonal();
    ROS_INFO("===================size Zero========================");
    // std::cout<<"size of Pestimated: "<<Pestimated.rows()<<std::endl;
    ROS_INFO("Pestimated dimensions: %d x %d", my_Pestimated.rows(), my_Pestimated.cols());
    ROS_INFO("==========================over================================");



    state_server.state_cov  = my_Pestimated;   // 初始化状态协方差

    // 观测噪声标准差（m->rad）
    state_server.imu_state.posN_meas_noise_std = (state_server.imu_state.posN_meas_noise_std.array()/position_m2rad).matrix();
    ROS_INFO("=====================0 kalman filter initialization complete=============================");
}


/**
 * @brief imu初始化，这里相当于是将初始静止下的参考值拿出来作为初值进行状态迭代
 */
void MsckfVio::INS_Inialization()
{

    // 这里是直接将姿态、速度、位置直接拿出来进行处理
    Vector3d atti_conference = Vector3d::Zero();
    Vector3d vel_conference = Vector3d::Zero();
    Vector3d posi_conference = Vector3d::Zero();

    Vector3d init_att = Vector3d::Zero();
    Vector3d init_vel = Vector3d::Zero();
    Vector3d init_pos = Vector3d::Zero();

    double m_time_618Dpro, m_time_618Dpro_s;                            // 原始的618Dpro记载的缺少了ROS时间戳，所以这里是拿GPS时间来处理

    // 所以这里不应该拿IMU的速度，而应该拿参考值部分的消息内容
    for (const auto &gps_msg : Tru_couter_buffer)
    {

        // 使用自定义的消息类型转换（针对618Dpro旧的数据定义）
        MsckfVio::ForsenseMsgAttiToEigen(gps_msg, init_att);
        MsckfVio::ForsenseMsgVeloToEigen(gps_msg, init_vel);
        MsckfVio::ForsenseMsgPosiToEigen(gps_msg, init_pos);

        MsckfVio::ForsenseMsgTimeToDouble(gps_msg, m_time_618Dpro);

    }

    // 得出姿态，速度，位置（相当于最后的参考值作为初始化处理）
    state_server.imu_state.AttN = init_att;
    // state_server.imu_state.AttN << init_att[0],init_att[1], 360.0 - init_att[2]; // 618Dpro定义的航向角符合顺时针为+，所以不需要进行转换  --2023-05-03
    // state_server.imu_state.velocity = init_vel; 
    state_server.imu_state.velocity << init_vel[1],init_vel[0], -init_vel[2]; // 注意618Dpro录制的是ned三方向速度，这里需要改成ENU三方向的速度
    state_server.imu_state.position = init_pos;
    // state_server.imu_state.position << init_pos[1]/1E7,init_pos[0]/1E7,init_pos[2]/1E3; //统一格式为经纬高（deg,deg,m）

    // 2023-05-04 同步将初值进行记录
    state_server.imu_state.init_att_618Dpro = init_att;
    state_server.imu_state.init_vel_618Dpro << init_vel[1],init_vel[0], -init_vel[2];
    // state_server.imu_state.init_pos_618Dpro << init_pos[1]/1E7,init_pos[0]/1E7,init_pos[2]/1E3;
    state_server.imu_state.init_pos_618Dpro << init_pos;

    // 同时给IMU进行时间赋值，这样便于后续的IMU迭代
    // m_time_618Dpro_s = m_time_618Dpro/1000.0;
    m_time_618Dpro_s = m_time_618Dpro;   // 已经在ForsenseMsgTimeToDouble实现了时间的转换(ms-->s)
    state_server.imu_state.time = m_time_618Dpro_s; // 统一单位为s

    ROS_INFO("===============START IMU INIT========================");
    // ROS_INFO("initial attitude : attix=%f,attiy=%f,attiz=%f ", state_server.imu_state.AttN[0],state_server.imu_state.AttN[1],state_server.imu_state.AttN[2]);
    // ROS_INFO("initial velocity : velox=%f,veloy=%f,veloz=%f ", state_server.imu_state.velocity[0], state_server.imu_state.velocity[1], state_server.imu_state.velocity[2] );
    // ROS_INFO("initial position : posix=%f,posiy=%f,posiz=%f ", state_server.imu_state.position[0], state_server.imu_state.position[1], state_server.imu_state.position[2]);
    // ROS_INFO("initial time : time=%f ", state_server.imu_state.time);

    // 由于显示精度的问题，这里用cout
    std::cout<<setprecision(20)<<"initial attitude : attix="<<state_server.imu_state.AttN[0]<<",attiy="<<state_server.imu_state.AttN[1]<<",attiz="<<state_server.imu_state.AttN[2]<<std::endl;  
    std::cout<<setprecision(20)<<"initial velocity : velox="<<state_server.imu_state.velocity[0]<<",veloy="<<state_server.imu_state.velocity[1]<<",veloz="<<state_server.imu_state.velocity[2]<<std::endl;  
    std::cout<<setprecision(20)<<"initial position : posix="<<state_server.imu_state.position[0]<<",posiy="<<state_server.imu_state.position[1]<<",posiz="<<state_server.imu_state.position[2]<<std::endl;  
    std::cout<<setprecision(20)<<"initial time : time="<<state_server.imu_state.time<<std::endl;  
    
    ROS_INFO("=====================end=============================");
    return;
}

/**
 * @brief 进行INS状态的迭代（姿态、速度、位置）
 */
// void MsckfVio::INS_Update(const double& time,
//                           const Vector3d& m_gyro,
//                           const Vector3d& m_acc,
//                           double dtime,
//                           Vector3d gyro,
//                           Vector3d acc )
void MsckfVio::INS_Update()
{
    // 这里是实现地理系下的IMU的参数迭代处理，主要包括IMU的零偏修正，状态更新
    // Remove the bias from the measured gyro and acceleration
    // IMUState& imu_state = state_server.imu_state;
    // my_gyro = m_gyro - imu_state.gyros_bias - imu_state.gyros_markov_bias;
    // my_acc = m_acc - imu_state.acc_markov_bias;
    // dtime = time;
    // 设置一个中间量用于状态的传递
    // Vector3d AttN_loose, Velo_loose, Pose_loose;

    // // Update the state info
    // state_server.imu_state.time = my_dtime;

    ROS_INFO("===============START INS ========================");
    // ===============================================1、姿态角的更新========================================
    // INS_Attitude_Update();
    static Eigen::Vector3d WnbbA_old = Eigen::Vector3d::Zero();

    double Re=6378137.0;        // 地球半径（米）
    double f=1/298.257;         //地球的椭圆率
    double Wie=7.292115147e-5;  //地球自转角速度
    double g=9.7803698;         // 重力加速度
    // cout<<"run here"<<__LINE__<<endl;
    // 飞行器位置
    double latitude = state_server.imu_state.position(1)*deg2rad;
    double height   = state_server.imu_state.position(2);

    // 地球曲率半径求解
    double Rm       = Re*(1-2*f+3*f*sin(latitude)*sin(latitude));
    double Rn       = Re*(1+f*sin(latitude)*sin(latitude));

    // 坐标系N-->B
    double roll  = state_server.imu_state.AttN(0)*deg2rad;
    double pitch = state_server.imu_state.AttN(1)*deg2rad;
    double head  = state_server.imu_state.AttN(2)*deg2rad;
    Eigen::Matrix3d Cbn;
    Cbn  << cos(roll)*cos(head)+sin(roll)*sin(pitch)*sin(head), -cos(roll)*sin(head)+sin(roll)*sin(pitch)*cos(head), -sin(roll)*cos(pitch),
            cos(pitch)*sin(head),                               cos(pitch)*cos(head),                                sin(pitch),
            sin(roll)*cos(head)-cos(roll)*sin(pitch)*sin(head), -sin(roll)*sin(head)-cos(roll)*sin(pitch)*cos(head), cos(roll)*cos(pitch);
    // 飞行器速度
    double Ve = state_server.imu_state.velocity(0);
    double Vn = state_server.imu_state.velocity(1);

    Eigen::Vector3d Wenn(-Vn/(Rm+height), Ve/(Rn+height),  Ve/(Rn+height)*tan(latitude));
    Eigen::Vector3d Wien(0,               Wie*cos(latitude),           Wie*sin(latitude));
        // 单位：弧度/秒
    Eigen::Vector3d Wnbb  = my_gyro *deg2rad-Cbn*(Wien+Wenn);
        // 单位：弧度/秒

    //  %%%%%%%%%%%%%%四元数法求姿态矩阵%%%%%%%%%%%%%%%
    Eigen::Vector4d Q;
    Q    << cos(head/2)*cos(pitch/2)*cos(roll/2)+sin(head/2)*sin(pitch/2)*sin(roll/2),
            cos(head/2)*sin(pitch/2)*cos(roll/2)+sin(head/2)*cos(pitch/2)*sin(roll/2),
            cos(head/2)*cos(pitch/2)*sin(roll/2)-sin(head/2)*sin(pitch/2)*cos(roll/2),
            -1.0*sin(head/2)*cos(pitch/2)*cos(roll/2)+cos(head/2)*sin(pitch/2)*sin(roll/2);

    Eigen::Vector3d WnbbA  = Wnbb* my_dtime ;
    double WnbbA0 = sqrt(WnbbA(0)*WnbbA(0)+WnbbA(1)*WnbbA(1)+WnbbA(2)*WnbbA(2));

    Eigen::Matrix4d WnbbX;
    WnbbX << 0,        -WnbbA(0),   -WnbbA(1), -WnbbA(2),
             WnbbA(0),  0,           WnbbA(2), -WnbbA(1),
             WnbbA(1), -WnbbA(2),   0,          WnbbA(0),
             WnbbA(2),  WnbbA(1),  -WnbbA(0),   0       ;

    double c_q    = cos(WnbbA0/2);

    double d_q;
    if( WnbbA0<=1.0e-15 )
        d_q = 0.5;
    else
        d_q = sin(WnbbA0/2)/WnbbA0;

    // %%%%%%%%%%%等效转动矢量修正算法%%%%%%%%%%%%%%%
    Eigen::Vector3d WnbbA_e = WnbbA_old.cross(WnbbA);

    Eigen::Matrix4d WnbbX_e;
    WnbbX_e << 0,          -WnbbA_e(0), -WnbbA_e(1), -WnbbA_e(2),
               WnbbA_e(0),  0,           WnbbA_e(2), -WnbbA_e(1),
               WnbbA_e(1), -WnbbA_e(2),   0,          WnbbA_e(0),
               WnbbA_e(2),  WnbbA_e(1),  -WnbbA_e(0),   0       ;

    Eigen::Matrix4d eye_4   = Eigen::Matrix4d::Identity();
    Q       = ( c_q*eye_4+d_q*(WnbbX+1/12*WnbbX_e) )*Q; //修正算法

    WnbbA_old = WnbbA;


    // %%%%%%%%%%四元数规范化%%%%%%%%%
    double tmp_Q = sqrt(Q(0)*Q(0)+Q(1)*Q(1)+Q(2)*Q(2)+Q(3)*Q(3));

    for (int i=0; i<4; i++ )
        Q(i)=Q(i)/tmp_Q;


    // %%%%%%%%%%获取姿态矩阵%%%%%%%%%
    Cbn << Q(1)*Q(1)+Q(0)*Q(0)-Q(3)*Q(3)-Q(2)*Q(2), 2*(Q(1)*Q(2)+Q(0)*Q(3)), 2*(Q(1)*Q(3)-Q(0)*Q(2)),
            2*(Q(1)*Q(2)-Q(0)*Q(3)), Q(2)*Q(2)-Q(3)*Q(3)+Q(0)*Q(0)-Q(1)*Q(1),  2*(Q(2)*Q(3)+Q(0)*Q(1)),
            2*(Q(1)*Q(3)+Q(0)*Q(2)), 2*(Q(2)*Q(3)-Q(0)*Q(1)), Q(3)*Q(3)-Q(2)*Q(2)-Q(1)*Q(1)+Q(0)*Q(0);


    //求姿态(横滚、俯仰、航向） (还需要对分母为零的情况进行处理！！！2005年4月3日，目前还没有修改)
    state_server.imu_state.AttN(0) = atan(-Cbn(0,2)/Cbn(2,2))*rad2deg;
    state_server.imu_state.AttN(1) = atan(Cbn(1,2)/sqrt(Cbn(1,0)*Cbn(1,0)+Cbn(1,1)*Cbn(1,1)))*rad2deg;
    state_server.imu_state.AttN(2) = atan(Cbn(1,0)/Cbn(1,1))*rad2deg;
        // 单位：度

    //象限判断
    if(Cbn(1,1)<0)
        state_server.imu_state.AttN(2)=180.0+state_server.imu_state.AttN(2);
    else if(Cbn(1,0)<0)
        state_server.imu_state.AttN(2)=360.0+state_server.imu_state.AttN(2);
        // 航向角度（单位：度）
    if(Cbn(2,2)<0)
    {
        if(Cbn(0,2)>0)
            state_server.imu_state.AttN(0)=-(180.0-state_server.imu_state.AttN(0));
        if(Cbn(0,2)<0)
            state_server.imu_state.AttN(0)=180.0+state_server.imu_state.AttN(0);
    }
        // 横滚角度（单位：度）

    // // 姿态更新
    // state_server.imu_state.AttN = AttN_loose;
    // =================================================================================

    // =================================2、速度更新=======================================
    // INS_Velocity_Update();
    // // 提取前一时刻下计算的速度
    // Velo_loose = state_server.imu_state.velocity;

    // 坐标系N-->B
    roll  = state_server.imu_state.AttN(0)*deg2rad;
    pitch = state_server.imu_state.AttN(1)*deg2rad;
    head  = state_server.imu_state.AttN(2)*deg2rad;
    // Eigen::Matrix3d Cbn;
    Cbn  << cos(roll)*cos(head)+sin(roll)*sin(pitch)*sin(head), -cos(roll)*sin(head)+sin(roll)*sin(pitch)*cos(head), -sin(roll)*cos(pitch),
            cos(pitch)*sin(head),                               cos(pitch)*cos(head),                                sin(pitch),
            sin(roll)*cos(head)-cos(roll)*sin(pitch)*sin(head), -sin(roll)*sin(head)-cos(roll)*sin(pitch)*cos(head), cos(roll)*cos(pitch);

    Eigen::Vector3d Fn = Cbn.transpose()* my_acc;

    // 飞行器速度
    Ve = state_server.imu_state.velocity(0);
    Vn = state_server.imu_state.velocity(1);

    Wenn << -Vn/(Rm+height), Ve/(Rn+height),  Ve/(Rn+height)*tan(latitude);
    Wien << 0,               Wie*cos(latitude),           Wie*sin(latitude);

    // 一阶龙格库塔法解微分方程
    Eigen::Vector3d g_matrix(0,0,-g);
    state_server.imu_state.velocity = state_server.imu_state.velocity + my_dtime*(Fn - (2*Wien+Wenn).cross(state_server.imu_state.velocity)+g_matrix);

    // // 速度更新
    // state_server.imu_state.velocity = Velo_loose;

    // =================================================================================

    // =================================3、位置更新=======================================
    // INS_Position_Update();
    // 飞行器位置
    double longitude = state_server.imu_state.position(0)*deg2rad;
    latitude  = state_server.imu_state.position(1)*deg2rad;
    height    = state_server.imu_state.position(2);

    height    = height    + my_dtime*state_server.imu_state.velocity(2);
    latitude  = latitude  + my_dtime*(state_server.imu_state.velocity(1)/(Rm+height));
    longitude = longitude + my_dtime*(state_server.imu_state.velocity(0)/((Rn+height)*cos(latitude)));

    state_server.imu_state.position(0) = longitude*rad2deg; //单位：度
    state_server.imu_state.position(1) = latitude *rad2deg; //单位：度
    state_server.imu_state.position(2) = height;

    cout<<"run here"<<__LINE__<<endl;
    ROS_INFO("=====================d_time========================");
    ROS_INFO("my_dtime: %f", my_dtime);
    ROS_INFO("==========================over================================");

    // // 位置更新
    // state_server.imu_state.position = Pose_loose;

    // =================================================================================
    ROS_INFO("=============== 1 INS finish========================");

    return;
}


/**
 * @brief 进行系统模型的状态迭代处理
 */
// void MsckfVio::system_model_cal(double dtime,
//                                 Vector3d gyro,
//                                 Vector3d acc,
//                                 Eigen::MatrixXd F, 
//                                 Eigen::MatrixXd Q) 
void MsckfVio::system_model_cal() 
{
    // IMUState& imu_state = state_server.imu_state;
    // Vector3d gyro = m_gyro - imu_state.gyros_bias - imu_state.gyros_markov_bias;
    // Vector3d acc = m_acc - imu_state.acc_markov_bias;
    Eigen::MatrixXd I18  =  Eigen::MatrixXd::Identity(18,18);
    //地球模型参数
    double Re=6378137.0;        // 地球半径（米）
    double f=1.0/298.257;       // 地球椭圆率
    double Wie=7.292115147e-5;  // 地球自转角速度

    // Vector3d Ta(1800.0,1800.0,1800.0);                      // 加速度 一阶马尔可夫相关时间（与IMU仿真同）  10
    // Vector3d Tg(3600.0,3600.0,3600.0);                      // 陀螺   一阶马尔可夫相关时间（与IMU仿真同）  10
    // Vector3d gyros_noise_std(1E-7,1E-7,1E-7); 
    // Vector3d gyros_markov_noise_std(1E-5,1E-5,1E-5); 
    // Vector3d accel_markov_noise_std(1E-5,1E-5,1E-5);

    //飞行器位置
    double latitude  = state_server.imu_state.position(1)*deg2rad;
    double height    = state_server.imu_state.position(2);

    //地球曲率半径求解
    double Rm = Re *( 1 - 2*f + 3*f*sin(latitude)*sin(latitude) );
    double Rn = Re *( 1 + f*sin(latitude)*sin(latitude) );

    //坐标系N-->B
    double roll  = state_server.imu_state.AttN(0)*deg2rad;
    double pitch = state_server.imu_state.AttN(1)*deg2rad;
    double head  = state_server.imu_state.AttN(2)*deg2rad;

    Eigen::Matrix3d Cbn;
    Cbn << cos(roll)*cos(head)+sin(roll)*sin(pitch)*sin(head), -cos(roll)*sin(head)+sin(roll)*sin(pitch)*cos(head), -sin(roll)*cos(pitch),
           cos(pitch)*sin(head),                               cos(pitch)*cos(head),                                sin(pitch),
           sin(roll)*cos(head)-cos(roll)*sin(pitch)*sin(head), -sin(roll)*sin(head)-cos(roll)*sin(pitch)*cos(head), cos(roll)*cos(pitch);

    Eigen::Vector3d accel_n; accel_n = Cbn.transpose()*my_acc;

    //姿态、速度转换
    double velocityE = state_server.imu_state.velocity(0);
    double velocityN = state_server.imu_state.velocity(1);
    double velocityU = state_server.imu_state.velocity(2);

    // ***********************************求解状态转移矩阵*************************************//
    // Step 1: 位置速度姿态误差 - 位置速度姿态误差 微分方程
    Eigen::MatrixXd FN = Eigen::MatrixXd::Zero(9,9);
    FN(0,1) =    Wie*sin(latitude)  + velocityE/(Rn+height)*tan(latitude);
    FN(0,2) = - (Wie*cos(latitude)  + velocityE/(Rn+height));
    FN(0,4) = -  1/(Rm+height);
    FN(1,0) = - (Wie*sin(latitude)  + velocityE/(Rn+height)*tan(latitude));
    FN(1,2) = -  velocityN/(Rm+height);
    FN(1,3) =    1/(Rn+height);
    FN(1,6) = -  Wie*sin(latitude);
    FN(2,0) =    Wie*cos(latitude)  + velocityE/(Rn+height);
    FN(2,1) =    velocityN/(Rm+height);
    FN(2,3) =    1/(Rn+height)*tan(latitude);
    FN(2,6)=     Wie*cos(latitude)  + velocityE/(Rn+height)*(1.0/cos(latitude))* (1.0/cos(latitude));
    FN(3,1) = -  accel_n(2);
    FN(3,2) =    accel_n(1);
    FN(3,3) =    velocityN/(Rm+height)*tan(latitude) - velocityU/(Rm+height);
    FN(3,4) =    2*Wie*sin(latitude) + velocityE/(Rn+height)*tan(latitude);
    FN(3,5) = - (2*Wie*cos(latitude) + velocityE/(Rn+height));
    FN(3,6) =    2*Wie*cos(latitude)*velocityN + velocityE*velocityN/(Rn+height)*(1.0/cos(latitude))* (1.0/cos(latitude)) + 2*Wie*sin(latitude)*velocityU;
    FN(4,0) =    accel_n(2);
    FN(4,2) = -  accel_n(0);
    FN(4,3) = -  2*(Wie*sin(latitude)+ velocityE/(Rn+height)*tan(latitude));
    FN(4,4) = -  velocityU/(Rm+height);
    FN(4,5) = -  velocityN/(Rm+height);
    FN(4,6) = - (2*Wie*cos(latitude) + velocityE/(Rn+height)*(1.0/cos(latitude))* (1.0/cos(latitude)) )*velocityE;
    FN(5,0) = -  accel_n(1);
    FN(5,1) =    accel_n(0);
    FN(5,3) =    2*(Wie*cos(latitude)+ velocityE/(Rn+height));
    FN(5,4) =    2*velocityN/(Rm+height);
    FN(5,6) = -  2*velocityE*Wie*sin(latitude);
    FN(6,4) =    1/(Rm+height);
    FN(7,3) =    (1.0/cos(latitude))/(Rn+height);
    FN(7,6) =    velocityE/(Rn+height)*(1/cos(latitude))*tan(latitude);
    FN(8,5) =    1;

    // Step 2: 位置速度姿态误差 - 加速度计和陀螺噪声 微分方程
    Eigen::MatrixXd FS = Eigen::MatrixXd::Zero(9,9);
    FS << Cbn.transpose(),  Cbn.transpose(),  Eigen::Matrix3d::Zero(),
          Eigen::Matrix3d::Zero(), Eigen::Matrix3d::Zero(), Cbn.transpose(),
          Eigen::Matrix3d::Zero(), Eigen::Matrix3d::Zero(), Eigen::Matrix3d::Zero();

    // Step 3: 加速度计和陀螺噪声 - 加速度计和陀螺噪声 微分方程
    Eigen::VectorXd FM_Vector = Eigen::VectorXd::Zero(9); FM_Vector <<  Eigen::Vector3d::Zero(), (-1.0/state_server.imu_state.Tg.array()).matrix(), (-1.0/state_server.imu_state.Ta.array()).matrix();
    Eigen::MatrixXd FM = FM_Vector.asDiagonal();

    // Step 4: 计算状态转移 微分方程
    Eigen::MatrixXd FI  = Eigen::MatrixXd::Zero(18,18);
    FI <<   FN,FS,
            Eigen::MatrixXd::Zero(9,9),FM;

    cout<<"run here"<<__LINE__<<endl;
    std::stringstream ss_Temp_0FI;
    ss_Temp_0FI << FI;
    ROS_INFO("=====================ss_Temp_0FI========================");
    ROS_INFO("Matrix:\n%s", ss_Temp_0FI.str().c_str());
    ROS_INFO("==========================over================================");

    // Step 5: 状态转移微分方程离散化 ----> 状态转移矩阵 

    my_F =  I18+ FI*my_dtime + FI*(FI/2.0)*my_dtime*my_dtime;

    Eigen::MatrixXd Temp_0F = my_F;


    std::stringstream ss_Temp_0F;
    ss_Temp_0F << Temp_0F;
    ROS_INFO("=====================ss_Temp_0F========================");
    ROS_INFO("Matrix:\n%s", ss_Temp_0F.str().c_str());
    ROS_INFO("==========================over================================");

    cout<<"run here"<<__LINE__<<endl;
    ROS_INFO("=====================d_time========================");
    ROS_INFO("my_dtime: %f", my_dtime);
    ROS_INFO("==========================over================================");

    // ***********************************求解系统噪声协方差矩阵*************************************//
    // Step 1: 连续系统噪声分布微分方程
    Eigen::MatrixXd GI = Eigen::MatrixXd::Zero(18,9);
    GI.block(0,0,3,3) = Cbn.transpose();
    GI.block(12,3,6,6) = Eigen::MatrixXd::Identity(6,6);

    // Step 2: 连续系统噪声分布微分方程离散化 ----> 系统噪声分布矩阵
    Eigen::MatrixXd  GL = (I18+FI/2.0*my_dtime+FI*FI/6.0*my_dtime*my_dtime)*GI*my_dtime;

    // Step 3: 系统噪声向量
    Eigen::VectorXd W = Eigen::VectorXd::Zero(9);
    W << state_server.imu_state.gyros_noise_std*deg2rad, state_server.imu_state.gyros_markov_noise_std*deg2rad, state_server.imu_state.accel_markov_noise_std;
//    double pi = 3.14159265358979;
//    W << 208.0*pi/(3600.0*180.0),                               178.0*pi/(3600.0*180.0),                               70.0*pi/(3600.0*180.0),
//         sqrt(2.0*INS->IMU->tor/Tg(0))*208.0*pi/(3600.0*180.0), sqrt(2.0*INS->IMU->tor/Tg(1))*178.0*pi/(3600.0*180.0), sqrt(2.0*INS->IMU->tor/Tg(2))*70.0*pi/(3600.0*180.0),
//         sqrt(2.0*INS->IMU->tor/Ta(0))*0.0015,                  sqrt(2.0*INS->IMU->tor/Ta(0))*0.0015,                  sqrt(2.0*INS->IMU->tor/Ta(0))*0.002;

    // Step 4: 系统噪声协方差矩阵
    Eigen::MatrixXd W_Matrix = Eigen::MatrixXd::Zero(9,9); W_Matrix = W.array().pow(2).matrix().asDiagonal();
    my_Q = GL* W_Matrix*(GL.transpose()) ;

    ROS_INFO("=====================2 kalman process model finish=============================");

    return;
}

/**
 * @brief 进行系统模型的状态迭代后对状态量的迭代处理
 */
// void MsckfVio::KF_time_update(  Eigen::VectorXd Xestimated,
//                                 Eigen::MatrixXd Pestimated,
//                                 Eigen::MatrixXd F, 
//                                 Eigen::MatrixXd Q)
void MsckfVio::KF_time_update()
{
    my_Xestimated = my_F*my_Xestimated;
    my_Pestimated = my_F*my_Pestimated*(my_F.transpose()) + my_Q;
    ROS_INFO("=====================3 kalman prediction finish=============================");
    return;
}

/**
 * @brief 松组合观测模型的构建
 */
void MsckfVio::measur_model_cal(Eigen::Vector3d m_posi_GPS)
{
    double fe = 1.0/298.257; //地球的椭圆率
    double Re = 6378137.0; //地球半径（米）

    //飞行器位置
    double latitude  = state_server.imu_state.position(1)*deg2rad;
    double height    = state_server.imu_state.position(2);

    double Rm = Re*(1.0-2.0*fe+3.0*fe*sin(latitude)*sin(latitude));
    double Rn = Re*(1.0+fe*sin(latitude)*sin(latitude));

    // GPS/INS位置量测矩阵
    my_H = Eigen::MatrixXd::Zero(3,18);
    Eigen::Matrix3d H_Pos = Eigen::Vector3d(Rm,Rn*cos(latitude),1).asDiagonal();
    my_H << Eigen::Matrix<double,3,6>::Zero(),H_Pos,Eigen::Matrix<double,3,9>::Zero();

    //位置观测标准差
    Eigen::Vector3d V = state_server.imu_state.posN_meas_noise_std;

    my_R = (V.array().pow(2).matrix().asDiagonal());

   // 位置观测向量
    my_Z = Eigen::Vector3d( (state_server.imu_state.position(1)-m_posi_GPS(1))*deg2rad*(Rm+height),
                 (state_server.imu_state.position(0)-m_posi_GPS(0))*deg2rad*(Rn+height)*cos(latitude),
                  state_server.imu_state.position(2)-m_posi_GPS(2) );

    // 用于直接显示位置上的误差（经纬高）
    Eigen::Vector3d Err_z = Eigen::Vector3d( (state_server.imu_state.position(1)-m_posi_GPS(1)),
                                    (state_server.imu_state.position(0)-m_posi_GPS(0)),
                                    state_server.imu_state.position(2)-m_posi_GPS(2) );

    ROS_INFO("===================INS position========================");
    // ROS_INFO("state_server.imu_state.position(0): %f", state_server.imu_state.position(0));
    // ROS_INFO("state_server.imu_state.position(1): %f", state_server.imu_state.position(1));
    // ROS_INFO("state_server.imu_state.position(2): %f", state_server.imu_state.position(2));

    std::cout<<setprecision(20)<<"state_server.imu_state.position(0)="<<state_server.imu_state.position[0]<<std::endl;  
    std::cout<<setprecision(20)<<"state_server.imu_state.position(1)="<<state_server.imu_state.position[1]<<std::endl;  
    std::cout<<setprecision(20)<<"state_server.imu_state.position(2)="<<state_server.imu_state.position[2]<<std::endl;   

    ROS_INFO("===================GPS position========================");
    // ROS_INFO("GPS_position(0): %f", m_posi_GPS(0));
    // ROS_INFO("GPS_position(1): %f", m_posi_GPS(1));
    // ROS_INFO("GPS_position(2): %f", m_posi_GPS(2));
    std::cout<<setprecision(20)<<"m_posi_GPS(0)="<<m_posi_GPS[0]<<std::endl;  
    std::cout<<setprecision(20)<<"m_posi_GPS(1)="<<m_posi_GPS[1]<<std::endl;  
    std::cout<<setprecision(20)<<"m_posi_GPS(2)="<<m_posi_GPS[2]<<std::endl;   

    ROS_INFO("===================Z========================");
    ROS_INFO("measurement_Z(0): %f", my_Z(0));
    ROS_INFO("measurement_Z(1): %f", my_Z(1));
    ROS_INFO("measurement_Z(2): %f", my_Z(2));
    ROS_INFO("===================Position_Z========================");
    ROS_INFO("Err_z(0): %f", Err_z(0));
    ROS_INFO("Err_z(1): %f", Err_z(1));
    ROS_INFO("Err_z(2): %f", Err_z(2));
    ROS_INFO("==========================over================================");

    ROS_INFO("=====================4 GPS measurement model finish=============================");

    return;
}

/**
 * @brief kalman滤波器修正迭代
 */
// void MsckfVio::KF_meas_update(Eigen::VectorXd Xestimated,
//                               Eigen::MatrixXd Pestimated,
//                               Eigen::MatrixXd H,
//                               Eigen::MatrixXd R,
//                               Eigen::Vector3d Z)
void MsckfVio::KF_meas_update()
{
    Eigen::MatrixXd Ppropagate = my_Pestimated;
    Eigen::MatrixXd K = Ppropagate*my_H.transpose()*(my_H*Ppropagate*my_H.transpose()+my_R).inverse();

    Eigen::VectorXd Xpropagate = my_Xestimated;
    my_Xestimated = Xpropagate + K*(my_Z - my_H*Xpropagate);

    std::stringstream ss_my_Xestimated;
    ss_my_Xestimated << my_Xestimated;
    cout<<"===================run here======================"<<__LINE__<<endl;
    ROS_INFO("=====================ss_my_Xestimated(update)========================");
    ROS_INFO("Matrix:\n%s", ss_my_Xestimated.str().c_str());
    ROS_INFO("==========================over================================");

    Eigen::MatrixXd I = Eigen::MatrixXd::Identity(my_Pestimated.rows(),my_Pestimated.cols());
    my_Pestimated = (I - K*my_H)*Ppropagate;

    // 第六步：传感器状态参数更新
    state_server.imu_state.acc_markov_bias = my_Xestimated.segment(15,3)*rad2deg;
    state_server.imu_state.gyros_markov_bias = my_Xestimated.segment(12,3)*rad2deg;
    state_server.imu_state.gyros_bias   = my_Xestimated.segment(9,3)*rad2deg;

    std::stringstream ss_Ppropagate;
    ss_Ppropagate << Ppropagate;
    ROS_INFO("=====================Ppropagate========================");
    ROS_INFO("Matrix:\n%s", ss_Ppropagate.str().c_str());
    ROS_INFO("==========================over================================");

    Eigen::MatrixXd Temp = (my_H*Ppropagate*my_H.transpose()+my_R).inverse();

    std::stringstream ss_Temp;
    ss_Temp << Temp;
    ROS_INFO("=====================ss_Temp========================");
    ROS_INFO("Matrix:\n%s", ss_Temp.str().c_str());
    ROS_INFO("==========================over================================");

    Eigen::MatrixXd Temp2 = my_H.transpose();

    std::stringstream ss_Temp2;
    ss_Temp2 << Temp2;
    ROS_INFO("=====================ss_Temp2========================");
    ROS_INFO("Matrix:\n%s", ss_Temp2.str().c_str());
    ROS_INFO("==========================over================================");

    // Eigen::MatrixXd K = Ppropagate*H.transpose()*(H*Ppropagate*H.transpose()+R).inverse();

    std::stringstream ss_K;
    ss_K << K;
    ROS_INFO("=====================K========================");
    ROS_INFO("Matrix:\n%s", ss_K.str().c_str());
    ROS_INFO("==========================over================================");

    ROS_INFO("=====================5 GPS measurement update finish=============================");
    return;
}                               

/**
 * @brief 基于kalman估计误差对状态进行修正处理
 */
// void MsckfVio::INS_Correction(Eigen::Vector3d PosNError,
//                               Eigen::Vector3d VelNError,
//                               Eigen::Vector3d AttNError) 
// void MsckfVio::INS_Correction(Eigen::VectorXd Xestimated) 
void MsckfVio::INS_Correction() 
{

    ROS_INFO("=====================6.2 start INS correct=============================");
    //IMU修正
    Eigen::Vector3d PosNError(my_Xestimated(7)*rad2deg,my_Xestimated(6)*rad2deg,my_Xestimated(8));
    Eigen::Vector3d VelNError = my_Xestimated.segment(3,3);
    Eigen::Vector3d AttNError = my_Xestimated.segment(0,3);

    ROS_INFO("=====================6.3 error correct get=============================");

    // 飞行器速度修正
    state_server.imu_state.velocity = state_server.imu_state.velocity - VelNError;

    // 飞行器位置修正
    state_server.imu_state.position = state_server.imu_state.position - PosNError;

    double roll  = state_server.imu_state.AttN(0)*deg2rad;
    double pitch = state_server.imu_state.AttN(1)*deg2rad;
    double head  = state_server.imu_state.AttN(2)*deg2rad;
    // 坐标系C-->B(机体系－－>计算系)
    Eigen::Matrix3d Cbc;
    Cbc  << cos(roll)*cos(head)+sin(roll)*sin(pitch)*sin(head), -cos(roll)*sin(head)+sin(roll)*sin(pitch)*cos(head), -sin(roll)*cos(pitch),
            cos(pitch)*sin(head),                               cos(pitch)*cos(head),                                sin(pitch),
            sin(roll)*cos(head)-cos(roll)*sin(pitch)*sin(head), -sin(roll)*sin(head)-cos(roll)*sin(pitch)*cos(head), cos(roll)*cos(pitch);

    // 修正矩阵N-->C(导航坐标系－－>计算坐标系)
    Eigen::Matrix3d Ccn;
    Ccn <<             1,     AttNError(2), -AttNError(1),
           -AttNError(2),                1,  AttNError(0),
            AttNError(1),    -AttNError(0),             1;
    // 姿态矩阵修正N--->B
    Eigen::Matrix3d Cbn = (Ccn.transpose() * Cbc.transpose()).transpose();

    //求姿态(横滚、俯仰、航向）
    state_server.imu_state.AttN(0) = atan(-Cbn(0,2)/Cbn(2,2))*rad2deg;
    state_server.imu_state.AttN(1) = atan(Cbn(1,2)/sqrt(Cbn(1,0)*Cbn(1,0)+Cbn(1,1)*Cbn(1,1)))*rad2deg;
    state_server.imu_state.AttN(2) = atan(Cbn(1,0)/Cbn(1,1))*rad2deg;

    // 象限判断
    if((Cbn(1,1)<0))
        state_server.imu_state.AttN(2) += 180.0;
    else if(Cbn(1,0)<0)
        state_server.imu_state.AttN(2) += 360.0;

    if((Cbn(2,2)<0))
    {
        if(Cbn(0,2)>0)
            state_server.imu_state.AttN(0) = 180.0 - state_server.imu_state.AttN(0);
        if(Cbn(0,2)<0)
            state_server.imu_state.AttN(0) = -(180.0 + state_server.imu_state.AttN(0));
    }

    cout<<"====================run here===================="<<__LINE__<<endl;
    ROS_INFO("===================corrected INS position========================");
    // ROS_INFO("state_server.imu_state.position(0): %f", state_server.imu_state.position(0));
    // ROS_INFO("state_server.imu_state.position(1): %f", state_server.imu_state.position(1));
    // ROS_INFO("state_server.imu_state.position(2): %f", state_server.imu_state.position(2));
    std::cout<<setprecision(20)<<"state_server.imu_state.position(0)="<<state_server.imu_state.position[0]<<std::endl;  
    std::cout<<setprecision(20)<<"state_server.imu_state.position(1)="<<state_server.imu_state.position[1]<<std::endl;  
    std::cout<<setprecision(20)<<"state_server.imu_state.position(2)="<<state_server.imu_state.position[2]<<std::endl;   

    ROS_INFO("===================corrected INS velocity========================");
    ROS_INFO("state_server.imu_state.velocity(0): %f", state_server.imu_state.velocity(0));
    ROS_INFO("state_server.imu_state.velocity(1): %f", state_server.imu_state.velocity(1));
    ROS_INFO("state_server.imu_state.velocity(2): %f", state_server.imu_state.velocity(2));
    ROS_INFO("===================corrected INS attitude========================");
    ROS_INFO("state_server.imu_state.AttN(0): %f", state_server.imu_state.AttN(0));
    ROS_INFO("state_server.imu_state.AttN(1): %f", state_server.imu_state.AttN(1));
    ROS_INFO("state_server.imu_state.AttN(2): %f", state_server.imu_state.AttN(2));
    ROS_INFO("=====================6.4 INS correct finish=============================");
    return;
}

/**
 * @brief 解算的轨迹等消息发布
 */
// void MsckfVio::publish_loose()
void MsckfVio::publish_loose(forsense_msg::Forsense msg_copy2)
{


    // // Publish tf
    // if (publish_tf) {
    //     tf::Transform T_b_w_tf(
    //     quaternionToRotation(
    //   state_server.imu_state.orientation)
    //     );
    //     tf::transformEigenToTF(T_b_w, T_b_w_tf);
    //     tf_pub.sendTransform(tf::StampedTransform(
    //         T_b_w_tf, time, "world", "odom"));
    // }

    // 2023-05-04 注意由于RVIZ不支持经纬高的直接显示，所以还是需要将经纬高转换到ENU系下。（暂定）
    // 分为以下两步：第一步，初始GPS位置的确定， 第二步，将解算的经纬高位置转换到ENU系下。

    // 获取初值
    Vector3d init_posi_first = state_server.imu_state.init_pos_618Dpro; 

    Vector3d PosiN_index = state_server.imu_state.position;
    //将单位进行转换： deg-->rad -->m
    double lati=PosiN_index[1]*deg2rad;
    double heig=PosiN_index[2];

    //地球模型参数
    double Re=6378137.0;        // 地球半径（米）
    double f=1.0/298.257;       // 地球椭圆率
    double Wie=7.292115147e-5;  // 地球自转角速度

    //地球曲率半径求解
    double Rm = Re *( 1 - 2*f + 3*f*sin(lati)*sin(lati) );
    double Rn = Re *( 1 + f*sin(lati)*sin(lati) );

    //位置转换(ENU系下)
    double PosiWi_m_lon = (PosiN_index[0] - init_posi_first[0]) * deg2rad * (Rn + heig) * cos(lati); //经度(m) E
    double PosiWi_m_lat = (PosiN_index[1] - init_posi_first[1]) * deg2rad * (Rm + heig); //纬度 N
    double PosiWi_m_h = PosiN_index[2] - init_posi_first[2]; //高度（m）U

    path_msg.header.stamp = ros::Time::now();
    path_msg.header.frame_id = "world";

    geometry_msgs::PoseStamped pose_stamped;
    // unsigned long time_second = state_server.imu_state.time;
    // pose_stamped.header.stamp = ros::Time((*it)->time_stamp);
    // pose_stamped.header.stamp = state_server.imu_state.time;
    pose_stamped.header.stamp = ros::Time::now();   // 由于记录中缺少数据录制时的ROS时间，所以这里拿现在的ROS时间来处理了
    pose_stamped.header.frame_id = "world";
    // pose_stamped.header.child_frame_id = "robot";

    // pose_stamped.pose.position.x = state_server.imu_state.position[0];
    // pose_stamped.pose.position.y = state_server.imu_state.position[1];
    // pose_stamped.pose.position.z = state_server.imu_state.position[2];
    // 用ENU系下的位置进行输入
    pose_stamped.pose.position.x = PosiWi_m_lon;
    pose_stamped.pose.position.y = PosiWi_m_lat;
    pose_stamped.pose.position.z = PosiWi_m_h;

    double roll  = state_server.imu_state.AttN(0)*deg2rad;
    double pitch = state_server.imu_state.AttN(1)*deg2rad;
    double head  = state_server.imu_state.AttN(2)*deg2rad;
    tf::Quaternion Q = tf::createQuaternionFromRPY(roll, pitch, head);  // 注意这里是rad而不是deg。

    pose_stamped.pose.orientation.x = Q.x();
    pose_stamped.pose.orientation.y = Q.y();
    pose_stamped.pose.orientation.z = Q.z();
    pose_stamped.pose.orientation.w = Q.w();

    // Create a path message
    // nav_msgs::Path path_msg;
    path_msg.poses.push_back(pose_stamped);

    path_pub.publish(path_msg);
    ROS_INFO("=====================7 path out=============================");

    // nav_msgs::Odometry odo;
    Vector3d Tru_pos_618Dpro;
    MsckfVio::ForsenseMsgPosiToEigen( msg_copy2, Tru_pos_618Dpro);
    // double Tru_pos_618Dpro = gps_msg->pos(0);

    // 同样也需要进行位置转换(ENU系下)  //统一格式为经纬高（deg,deg,m）
    //将单位进行转换： deg-->rad -->m
    // double lati_Tru=Tru_pos_618Dpro[0]/1E7*deg2rad;  // 注意原始真值是 纬经高
    // double heig_Tru=Tru_pos_618Dpro[2]/1E3;
    double lati_Tru=Tru_pos_618Dpro[1]*deg2rad;  // 注意原始真值已经在消息读取中改成了 经纬高
    double heig_Tru=Tru_pos_618Dpro[2];

    // double PosiWi_m_lon_Tru = (Tru_pos_618Dpro[1]/1E7 - init_posi_first[0]) * deg2rad * (Rn + heig_Tru) * cos(lati_Tru); //经度(m) E
    // double PosiWi_m_lat_Tru = (Tru_pos_618Dpro[0]/1E7 - init_posi_first[1]) * deg2rad * (Rm + heig_Tru); //纬度 N
    double PosiWi_m_lon_Tru = (Tru_pos_618Dpro[0] - init_posi_first[0]) * deg2rad * (Rn + heig_Tru) * cos(lati_Tru); //经度(m) E
    double PosiWi_m_lat_Tru = (Tru_pos_618Dpro[1] - init_posi_first[1]) * deg2rad * (Rm + heig_Tru); //纬度 N
    double PosiWi_m_h_Tru = Tru_pos_618Dpro[2] - init_posi_first[2]; //高度（m）U
    ROS_INFO("===================Xestimated========================");
    ROS_INFO("Tru_pos_618Dpro(0): %f", Tru_pos_618Dpro[0]);    // 经度 long
    ROS_INFO("Tru_pos_618Dpro(1): %f", Tru_pos_618Dpro[1]);    // 纬度 lati  
    ROS_INFO("Tru_pos_618Dpro(2): %f", Tru_pos_618Dpro[2]);
    ROS_INFO("==========================over================================");

    // // 先尝试将odo的信息输出看是否可以//可以
    // Tru_msg.header.frame_id ="world";
    // Tru_msg.header.stamp = ros::Time::now();   // 由于记录中缺少数据录制时的ROS时间，所以这里拿现在的ROS时间来处理了

    // Tru_msg.pose.pose.position.x = PosiWi_m_lon_Tru;
    // Tru_msg.pose.pose.position.y = PosiWi_m_lat_Tru;
    // Tru_msg.pose.pose.position.z = PosiWi_m_h_Tru;   // 将真值进行输入

    Tru_msg.header.stamp = ros::Time::now();
    Tru_msg.header.frame_id = "world";

    geometry_msgs::PoseStamped pose_stamped_Tru;
    pose_stamped_Tru.header.stamp = ros::Time::now();   // 由于记录中缺少数据录制时的ROS时间，所以这里拿现在的ROS时间来处理了
    pose_stamped_Tru.header.frame_id = "world";

    // 用ENU系下的位置进行输入
    pose_stamped_Tru.pose.position.x = PosiWi_m_lon_Tru;
    pose_stamped_Tru.pose.position.y = PosiWi_m_lat_Tru;
    pose_stamped_Tru.pose.position.z = PosiWi_m_h_Tru;

    Tru_msg.poses.push_back(pose_stamped_Tru);

    // Tru_msg_ALL.pose.pose.push_back();

    Tru_pub.publish(Tru_msg);
    ROS_INFO("=====================7.1 Tru path pub out=============================");

    return;

}                             

// void MsckfVio::INS_Attitude_Update()
// {
//     // 姿态迭代处理
// }

// void MsckfVio::INS_Velocity_Update()
// {
//     // 速度迭代处理
// }

// void MsckfVio::INS_Position_Update()
// {
//     // 位置迭代处理
// }

void MsckfVio::imuCallback(const sensor_msgs::ImuConstPtr &msg)
{
    // ROS_INFO("START IMU DATA...");

    // IMU msgs are pushed backed into a buffer instead of
    // being processed immediately. The IMU msgs are processed
    // when the next image is available, in which way, we can
    // easily handle the transfer delay.
    // 1. 存放imu数据
    imu_msg_buffer.push_back(*msg);

    // 2. 用200个imu数据做静止初始化，不够则不做
    if (!is_gravity_set)
    {

        if (imu_msg_buffer.size() < 200)
            return;
        // if (imu_msg_buffer.size() < 10) return;
        // imu初始化，200个数据必须都是静止时采集的
        // 这里面没有判断是否成功，也就是一开始如果运动会导致轨迹飘
        initializeGravityAndBias();
        is_gravity_set = true;
    }
    // ROS_INFO("START IMU OUTPUT...");

    return;
}

/**
 * @brief imu初始化，计算陀螺仪偏置，重力方向以及初始姿态，必须都是静止，且不做加速度计的偏置估计
 */
void MsckfVio::initializeGravityAndBias()
{

    // Initialize gravity and gyro bias.
    // 1. 角速度与加速度的和
    Vector3d sum_angular_vel = Vector3d::Zero();
    Vector3d sum_linear_acc = Vector3d::Zero();

    for (const auto &imu_msg : imu_msg_buffer)
    {
        Vector3d angular_vel = Vector3d::Zero();
        Vector3d linear_acc = Vector3d::Zero();

        tf::vectorMsgToEigen(imu_msg.angular_velocity, angular_vel);
        tf::vectorMsgToEigen(imu_msg.linear_acceleration, linear_acc);

        sum_angular_vel += angular_vel;
        sum_linear_acc += linear_acc;
    }

    // 2. 因为假设静止的，因此陀螺仪理论应该都是0，额外读数包括偏置+噪声，但是噪声属于高斯分布
    // 因此这一段相加噪声被认为互相抵消了，所以剩下的均值被认为是陀螺仪的初始偏置
    state_server.imu_state.gyro_bias =
        sum_angular_vel / imu_msg_buffer.size();
    // IMUState::gravity =
    //   -sum_linear_acc / imu_msg_buffer.size();
    //  This is the gravity in the IMU frame.
    // 3. 计算重力，忽略加速度计的偏置，剩下的就只有重力了
    Vector3d gravity_imu =
        sum_linear_acc / imu_msg_buffer.size();

    // Initialize the initial orientation, so that the estimation
    // is consistent with the inertial frame.
    // 重力本来的方向
    double gravity_norm = gravity_imu.norm();
    IMUState::gravity = Vector3d(0.0, 0.0, -gravity_norm);

    // 求出当前imu状态的重力方向与实际重力方向的旋转 tosee 查看谁到谁的
    Quaterniond q0_i_w = Quaterniond::FromTwoVectors(
        gravity_imu, -IMUState::gravity);
    // 得出姿态
    state_server.imu_state.orientation =
        rotationToQuaternion(q0_i_w.toRotationMatrix().transpose());

    ROS_INFO("===============START IMU INIT========================");
    return;
}

/**
 * @brief 重置
 */
bool MsckfVio::resetCallback(
    std_srvs::Trigger::Request &req,
    std_srvs::Trigger::Response &res)
{

    ROS_WARN("Start resetting msckf vio...");
    // Temporarily shutdown the subscribers to prevent the
    // state from updating.
    feature_sub.shutdown();
    imu_sub.shutdown();

    // Reset the IMU state.
    IMUState &imu_state = state_server.imu_state;
    imu_state.time = 0.0;
    imu_state.orientation = Vector4d(0.0, 0.0, 0.0, 1.0);
    imu_state.position = Vector3d::Zero();
    imu_state.velocity = Vector3d::Zero();
    imu_state.gyro_bias = Vector3d::Zero();
    imu_state.acc_bias = Vector3d::Zero();
    imu_state.orientation_null = Vector4d(0.0, 0.0, 0.0, 1.0);
    imu_state.position_null = Vector3d::Zero();
    imu_state.velocity_null = Vector3d::Zero();

    // Remove all existing camera states.
    state_server.cam_states.clear();

    // Reset the state covariance.
    double gyro_bias_cov, acc_bias_cov, velocity_cov;
    nh.param<double>("initial_covariance/velocity", velocity_cov, 0.25);
    nh.param<double>("initial_covariance/gyro_bias", gyro_bias_cov, 1e-4);
    nh.param<double>("initial_covariance/acc_bias", acc_bias_cov, 1e-2);

    double extrinsic_rotation_cov, extrinsic_translation_cov;
    nh.param<double>("initial_covariance/extrinsic_rotation_cov", extrinsic_rotation_cov, 3.0462e-4);
    nh.param<double>("initial_covariance/extrinsic_translation_cov", extrinsic_translation_cov, 1e-4);

    state_server.state_cov = MatrixXd::Zero(21, 21);
    for (int i = 3; i < 6; ++i)
        state_server.state_cov(i, i) = gyro_bias_cov;
    for (int i = 6; i < 9; ++i)
        state_server.state_cov(i, i) = velocity_cov;
    for (int i = 9; i < 12; ++i)
        state_server.state_cov(i, i) = acc_bias_cov;
    for (int i = 15; i < 18; ++i)
        state_server.state_cov(i, i) = extrinsic_rotation_cov;
    for (int i = 18; i < 21; ++i)
        state_server.state_cov(i, i) = extrinsic_translation_cov;

    // Clear all exsiting features in the map.
    map_server.clear();

    // Clear the IMU msg buffer.
    imu_msg_buffer.clear();

    // Reset the starting flags.
    is_gravity_set = false;
    is_first_img = true;

    // Restart the subscribers.
    imu_sub = nh.subscribe("imu", 100, &MsckfVio::imuCallback, this);
    feature_sub = nh.subscribe("features", 40, &MsckfVio::featureCallback, this);

    // TODO: When can the reset fail?
    res.success = true;
    ROS_WARN("Resetting msckf vio completed...");
    return true;
}

/**
 * @brief 后端主要函数，处理新来的数据
 */
void MsckfVio::featureCallback(const CameraMeasurementConstPtr &msg)
{

    ROS_INFO("START vio PROCESS...");
    // Return if the gravity vector has not been set.
    // 1. 必须经过imu初始化
    if (!is_gravity_set)
        return;

    // Start the system if the first image is received.
    // The frame where the first image is received will be
    // the origin.
    // 开始递推状态的第一个时刻为初始化后的第一帧特征
    if (is_first_img)
    {
        is_first_img = false;
        state_server.imu_state.time = msg->header.stamp.toSec();
    }

    // 调试使用
    static double max_processing_time = 0.0;
    static int critical_time_cntr = 0;
    double processing_start_time = ros::Time::now().toSec();

    // Propogate the IMU state.
    // that are received before the image msg.
    // 2. imu积分
    ros::Time start_time = ros::Time::now();
    batchImuProcessing(msg->header.stamp.toSec());
    double imu_processing_time = (ros::Time::now() - start_time).toSec();

    // Augment the state vector.
    // 3. 根据imu积分递推出的状态计算相机状态，更新协方差矩阵
    start_time = ros::Time::now();
    stateAugmentation(msg->header.stamp.toSec());
    double state_augmentation_time = (ros::Time::now() - start_time).toSec();

    // Add new observations for existing features or new
    // features in the map server.
    // 4. 添加新的观测
    start_time = ros::Time::now();
    addFeatureObservations(msg);
    double add_observations_time = (ros::Time::now() - start_time).toSec();

    // Perform measurement update if necessary.
    // 5. 使用不再跟踪上的点来更新
    start_time = ros::Time::now();
    removeLostFeatures();
    double remove_lost_features_time = (ros::Time::now() - start_time).toSec();

    // 6. 当cam状态数达到最大值时，挑出若干cam状态待删除
    // 并基于能被2帧以上这些cam观测到的feature进行MSCKF测量更新
    start_time = ros::Time::now();
    pruneCamStateBuffer();
    double prune_cam_states_time = (ros::Time::now() - start_time).toSec();

    // Publish the odometry.
    // 7. 发布位姿
    start_time = ros::Time::now();
    publish(msg->header.stamp);
    double publish_time = (ros::Time::now() - start_time).toSec();

    // Reset the system if necessary.
    // 8. 根据IMU状态位置协方差判断是否重置整个系统
    onlineReset();

    // 一些调试数据
    double processing_end_time = ros::Time::now().toSec();
    double processing_time =
        processing_end_time - processing_start_time;
    if (processing_time > 1.0 / frame_rate)
    {
        ++critical_time_cntr;
        ROS_INFO("\033[1;31mTotal processing time %f/%d...\033[0m",
                    processing_time, critical_time_cntr);
        // printf("IMU processing time: %f/%f\n",
        //     imu_processing_time, imu_processing_time/processing_time);
        // printf("State augmentation time: %f/%f\n",
        //     state_augmentation_time, state_augmentation_time/processing_time);
        // printf("Add observations time: %f/%f\n",
        //     add_observations_time, add_observations_time/processing_time);
        printf("Remove lost features time: %f/%f\n",
                remove_lost_features_time, remove_lost_features_time / processing_time);
        printf("Remove camera states time: %f/%f\n",
                prune_cam_states_time, prune_cam_states_time / processing_time);
        // printf("Publish time: %f/%f\n",
        //     publish_time, publish_time/processing_time);
    }

    return;
}

/**
 * @brief 没用，暂时不看
 */
void MsckfVio::mocapOdomCallback(
    const nav_msgs::OdometryConstPtr &msg)
{
    static bool first_mocap_odom_msg = true;

    // If this is the first mocap odometry messsage, set
    // the initial frame.
    if (first_mocap_odom_msg)
    {
        Quaterniond orientation;
        Vector3d translation;
        tf::pointMsgToEigen(
            msg->pose.pose.position, translation);
        tf::quaternionMsgToEigen(
            msg->pose.pose.orientation, orientation);
        // tf::vectorMsgToEigen(
        //     msg->transform.translation, translation);
        // tf::quaternionMsgToEigen(
        //     msg->transform.rotation, orientation);
        mocap_initial_frame.linear() = orientation.toRotationMatrix();
        mocap_initial_frame.translation() = translation;
        first_mocap_odom_msg = false;
    }

    // Transform the ground truth.
    Quaterniond orientation;
    Vector3d translation;
    // tf::vectorMsgToEigen(
    //     msg->transform.translation, translation);
    // tf::quaternionMsgToEigen(
    //     msg->transform.rotation, orientation);
    tf::pointMsgToEigen(
        msg->pose.pose.position, translation);
    tf::quaternionMsgToEigen(
        msg->pose.pose.orientation, orientation);

    Eigen::Isometry3d T_b_v_gt;
    T_b_v_gt.linear() = orientation.toRotationMatrix();
    T_b_v_gt.translation() = translation;
    Eigen::Isometry3d T_b_w_gt = mocap_initial_frame.inverse() * T_b_v_gt;

    // Eigen::Vector3d body_velocity_gt;
    // tf::vectorMsgToEigen(msg->twist.twist.linear, body_velocity_gt);
    // body_velocity_gt = mocap_initial_frame.linear().transpose() *
    //   body_velocity_gt;

    // Ground truth tf.
    if (publish_tf)
    {
        tf::Transform T_b_w_gt_tf;
        tf::transformEigenToTF(T_b_w_gt, T_b_w_gt_tf);
        tf_pub.sendTransform(tf::StampedTransform(
            T_b_w_gt_tf, msg->header.stamp, fixed_frame_id, child_frame_id + "_mocap"));
    }

    // Ground truth odometry.
    nav_msgs::Odometry mocap_odom_msg;
    mocap_odom_msg.header.stamp = msg->header.stamp;
    mocap_odom_msg.header.frame_id = fixed_frame_id;
    mocap_odom_msg.child_frame_id = child_frame_id + "_mocap";

    tf::poseEigenToMsg(T_b_w_gt, mocap_odom_msg.pose.pose);
    // tf::vectorEigenToMsg(body_velocity_gt,
    //     mocap_odom_msg.twist.twist.linear);

    mocap_odom_pub.publish(mocap_odom_msg);
    return;
}

/**
 * @brief imu积分，批量处理imu数据
 * @param  time_bound 处理到这个时间
 */
void MsckfVio::batchImuProcessing(const double &time_bound)
{
    // Counter how many IMU msgs in the buffer are used.
    int used_imu_msg_cntr = 0;

    // 取出两帧之间的imu数据去递推位姿
    // 这里有个细节问题，time_bound表示新图片的时间戳，
    // 但是IMU就积分到了距time_bound最近的一个，导致时间会差一点点
    for (const auto &imu_msg : imu_msg_buffer)
    {
        double imu_time = imu_msg.header.stamp.toSec();
        // 小于，说明这个数据比较旧，因为state_server.imu_state.time代表已经处理过的imu数据的时间
        if (imu_time < state_server.imu_state.time)
        {
            ++used_imu_msg_cntr;
            continue;
        }
        // 超过的供下次使用
        if (imu_time > time_bound)
            break;

        // Convert the msgs.
        Vector3d m_gyro, m_acc;
        tf::vectorMsgToEigen(imu_msg.angular_velocity, m_gyro);
        tf::vectorMsgToEigen(imu_msg.linear_acceleration, m_acc);

        // Execute process model.
        // 递推位姿，核心函数
        processModel(imu_time, m_gyro, m_acc);
        ++used_imu_msg_cntr;
    }

    // Set the state ID for the new IMU state.
    // 新的状态，更新id，相机状态的id也根据这个赋值
    state_server.imu_state.id = IMUState::next_id++;

    // 删掉已经用过
    // Remove all used IMU msgs.
    imu_msg_buffer.erase(
        imu_msg_buffer.begin(),
        imu_msg_buffer.begin() + used_imu_msg_cntr);

    return;
}

/**
 * @brief 来一个新的imu数据更新协方差矩阵与状态积分
 * @param  time 新数据时间戳
 * @param  m_gyro 角速度
 * @param  m_acc 加速度
 */
void MsckfVio::processModel(
    const double &time, const Vector3d &m_gyro, const Vector3d &m_acc)
{

    // Remove the bias from the measured gyro and acceleration
    // 以引用的方式取出
    IMUState &imu_state = state_server.imu_state;

    // 1. imu读数减掉偏置
    Vector3d gyro = m_gyro - imu_state.gyro_bias;
    Vector3d acc = m_acc - imu_state.acc_bias;  // acc_bias 初始值是0
    double dtime = time - imu_state.time;

    // 2. 计算F G矩阵
    // Compute discrete transition and noise covariance matrix
    Matrix<double, 21, 21> F = Matrix<double, 21, 21>::Zero();
    Matrix<double, 21, 12> G = Matrix<double, 21, 12>::Zero();

    // 误差为真值（观测） - 预测
    // F矩阵表示的是误差的导数的微分方程，其实就是想求δ的递推公式
    // δ`= F · δ + G    δ表示误差
    // δn+1 = (I + F·dt)·δn + G·dt·Q    Q表示imu噪声

    // 两种推法，一种是通过论文中四元数的推，这个过程不再重复
    // 下面给出李代数推法，我们知道msckf的四元数使用的是反人类的jpl
    // 也就是同一个数值的旋转四元数经过两种不同定义得到的旋转是互为转置的
    // 这里面的四元数转成的旋转表示的是Riw，所以要以李代数推的话也要按照Riw推
    // 按照下面的旋转更新方式为左乘，因此李代数也要用左乘，且jpl模式下左乘一个δq = Exp(-δθ)
    // δQj * Qjw = Exp(-(w - b) * t) * δQi * Qiw
    // Exp(-δθj) * Qjw = Exp(-(w - b) * t) * Exp(-δθi) * Qiw
    // 其中Qjw = Exp(-(w - b) * t) * Qiw
    // 两边除得到 Exp(-δθj) = Exp(-(w - b) * t) * Exp(-δθi) * Exp(-(w - b) * t).t()
    // -δθj = - Exp(-(w - b) * t) * δθi
    // δθj = (I - (w - b)^ * t)  * δθi  得证

    // 关于偏置一样可以这么算，只不过式子变成了
    // δQj * Qjw = Exp(-(w - b - δb) * t) * Qiw
    // 上式使用bch近似公式可以得到 δθj = -t * δb
    // 其他也可以通过这个方法推出，是正确的

    F.block<3, 3>(0, 0) = -skewSymmetric(gyro);
    F.block<3, 3>(0, 3) = -Matrix3d::Identity();


    F.block<3, 3>(6, 0) =
        -quaternionToRotation(imu_state.orientation).transpose() * skewSymmetric(acc);
    F.block<3, 3>(6, 9) = -quaternionToRotation(imu_state.orientation).transpose();
    F.block<3, 3>(12, 6) = Matrix3d::Identity();

    G.block<3, 3>(0, 0) = -Matrix3d::Identity();
    G.block<3, 3>(3, 3) = Matrix3d::Identity();
    G.block<3, 3>(6, 6) = -quaternionToRotation(imu_state.orientation).transpose();
    G.block<3, 3>(9, 9) = Matrix3d::Identity();

    // Approximate matrix exponential to the 3rd order,
    // which can be considered to be accurate enough assuming
    // dtime is within 0.01s.
    Matrix<double, 21, 21> Fdt = F * dtime;
    Matrix<double, 21, 21> Fdt_square = Fdt * Fdt;
    Matrix<double, 21, 21> Fdt_cube = Fdt_square * Fdt;

    // 3. 计算转移矩阵Phi矩阵
    // 论文Indirect Kalman Filter for 3D Attitude Estimation 中 公式164
    // 其实就是泰勒展开
    Matrix<double, 21, 21> Phi =
        Matrix<double, 21, 21>::Identity() + Fdt +
        0.5 * Fdt_square + (1.0 / 6.0) * Fdt_cube;

    // Propogate the state using 4th order Runge-Kutta
    // 4. 四阶龙格库塔积分预测状态
    predictNewState(dtime, gyro, acc);

    // 5. Observability-constrained VINS 可观性约束
    // Modify the transition matrix
    // 5.1 修改phi_11
    // imu_state.orientation_null为上一个imu数据递推后保存的
    // 这块可能会有疑问，因为当上一个imu假如被观测更新了，
    // 导致当前的imu状态是由更新后的上一个imu状态递推而来，但是这里的值是没有更新的，这个有影响吗
    // 答案是没有的，因为我们更改了phi矩阵，保证了零空间
    // 并且这里必须这么处理，因为如果使用更新后的上一个imu状态构建上一时刻的零空间
    // 就破坏了上上一个跟上一个imu状态之间的0空间
    // Ni-1 = phi_[i-2] * Ni-2
    // Ni = phi_[i-1] * Ni-1^
    // 如果像上面这样约束，那么中间的0空间就“崩了”
    Matrix3d R_kk_1 = quaternionToRotation(imu_state.orientation_null);
    Phi.block<3, 3>(0, 0) =
        quaternionToRotation(imu_state.orientation) * R_kk_1.transpose();

    // 5.2 修改phi_31
    Vector3d u = R_kk_1 * IMUState::gravity;
    RowVector3d s = (u.transpose() * u).inverse() * u.transpose();

    Matrix3d A1 = Phi.block<3, 3>(6, 0);
    Vector3d w1 =
        skewSymmetric(imu_state.velocity_null - imu_state.velocity) * IMUState::gravity;
    Phi.block<3, 3>(6, 0) = A1 - (A1 * u - w1) * s;

    // 5.3 修改phi_51
    Matrix3d A2 = Phi.block<3, 3>(12, 0);
    Vector3d w2 =
        skewSymmetric(
            dtime * imu_state.velocity_null + imu_state.position_null -
            imu_state.position) *
        IMUState::gravity;
    Phi.block<3, 3>(12, 0) = A2 - (A2 * u - w2) * s;

    // Propogate the state covariance matrix.
    // 6. 使用0空间约束后的phi计算积分后的新的协方差矩阵
    Matrix<double, 21, 21> Q =
        Phi * G * state_server.continuous_noise_cov * G.transpose() * Phi.transpose() * dtime;
    state_server.state_cov.block<21, 21>(0, 0) =
        Phi * state_server.state_cov.block<21, 21>(0, 0) * Phi.transpose() + Q;

    // 7. 如果有相机状态量，那么更新imu状态量与相机状态量交叉的部分
    if (state_server.cam_states.size() > 0)
    {
        // 起点是0 21  然后是21行 state_server.state_cov.cols() - 21 列的矩阵
        // 也就是整个协方差矩阵的右上角，这部分说白了就是imu状态量与相机状态量的协方差，imu更新了，这部分也需要更新
        state_server.state_cov.block(0, 21, 21, state_server.state_cov.cols() - 21) =
            Phi * state_server.state_cov.block(0, 21, 21, state_server.state_cov.cols() - 21);

        // 同理，这个是左下角
        state_server.state_cov.block(21, 0, state_server.state_cov.rows() - 21, 21) =
            state_server.state_cov.block(21, 0, state_server.state_cov.rows() - 21, 21) *
            Phi.transpose();
    }

    // 8. 强制对称，因为协方差矩阵就是对称的
    MatrixXd state_cov_fixed = 
        (state_server.state_cov + state_server.state_cov.transpose()) / 2.0;
    state_server.state_cov = state_cov_fixed;

    // Update the state correspondes to null space.
    // 9. 更新零空间，供下个IMU来了使用
    imu_state.orientation_null = imu_state.orientation;
    imu_state.position_null = imu_state.position;
    imu_state.velocity_null = imu_state.velocity;

    // Update the state info
    state_server.imu_state.time = time;
    return;
}

/**
 * @brief 来一个新的imu数据做积分，应用四阶龙哥库塔法
 * @param  dt 相对上一个数据的间隔时间
 * @param  gyro 角速度减去偏置后的
 * @param  acc 加速度减去偏置后的
 */
void MsckfVio::predictNewState(
    const double &dt, const Vector3d &gyro, const Vector3d &acc)
{

    // TODO: Will performing the forward integration using
    //    the inverse of the quaternion give better accuracy?

    // 角速度，标量
    double gyro_norm = gyro.norm();
    Matrix4d Omega = Matrix4d::Zero();
    Omega.block<3, 3>(0, 0) = -skewSymmetric(gyro);
    Omega.block<3, 1>(0, 3) = gyro;
    Omega.block<1, 3>(3, 0) = -gyro;

    Vector4d &q = state_server.imu_state.orientation;
    Vector3d &v = state_server.imu_state.velocity;
    Vector3d &p = state_server.imu_state.position;

    // Some pre-calculation
    // dq_dt表示积分n到n+1
    // dq_dt2表示积分n到n+0.5 算龙哥库塔用的
    Vector4d dq_dt, dq_dt2;
    if (gyro_norm > 1e-5)
    {
        dq_dt =
            (cos(gyro_norm * dt * 0.5) * Matrix4d::Identity() +
            1 / gyro_norm * sin(gyro_norm * dt * 0.5) * Omega) * q;
        dq_dt2 =
            (cos(gyro_norm * dt * 0.25) * Matrix4d::Identity() +
            1 / gyro_norm * sin(gyro_norm * dt * 0.25) * Omega) * q;
    }
    else
    {
        // 当角增量很小时的近似
        dq_dt = (Matrix4d::Identity() + 0.5 * dt * Omega) * cos(gyro_norm * dt * 0.5) * q;
        dq_dt2 = (Matrix4d::Identity() + 0.25 * dt * Omega) * cos(gyro_norm * dt * 0.25) * q;
    }
    // Rwi
    Matrix3d dR_dt_transpose = quaternionToRotation(dq_dt).transpose();
    Matrix3d dR_dt2_transpose = quaternionToRotation(dq_dt2).transpose();

    // k1 = f(tn, yn)
    Vector3d k1_v_dot = quaternionToRotation(q).transpose() * acc + IMUState::gravity;
    Vector3d k1_p_dot = v;

    // k2 = f(tn+dt/2, yn+k1*dt/2)
    // 这里的4阶LK法用了匀加速度假设，即认为前一时刻的加速度和当前时刻相等
    Vector3d k1_v = v + k1_v_dot * dt / 2;
    Vector3d k2_v_dot = dR_dt2_transpose * acc + IMUState::gravity;
    Vector3d k2_p_dot = k1_v;

    // k3 = f(tn+dt/2, yn+k2*dt/2)
    Vector3d k2_v = v + k2_v_dot * dt / 2;
    Vector3d k3_v_dot = dR_dt2_transpose * acc + IMUState::gravity;
    Vector3d k3_p_dot = k2_v;

    // k4 = f(tn+dt, yn+k3*dt)
    Vector3d k3_v = v + k3_v_dot * dt;
    Vector3d k4_v_dot = dR_dt_transpose * acc + IMUState::gravity;
    Vector3d k4_p_dot = k3_v;

    // yn+1 = yn + dt/6*(k1+2*k2+2*k3+k4)
    q = dq_dt;
    quaternionNormalize(q);
    v = v + dt / 6 * (k1_v_dot + 2 * k2_v_dot + 2 * k3_v_dot + k4_v_dot);
    p = p + dt / 6 * (k1_p_dot + 2 * k2_p_dot + 2 * k3_p_dot + k4_p_dot);

    return;
}

/**
 * @brief 根据时间分裂出相机状态
 * @param  time 图片的时间戳
 */
void MsckfVio::stateAugmentation(const double &time)
{
    // 1. 取出当前更新好的imu状态量
    // 1.1 取出状态量中的外参，这个东西是参与滤波的
    // 老规矩R_i_c 按照常理我们应该叫他 Rci imu到cam0的旋转
    const Matrix3d &R_i_c = state_server.imu_state.R_imu_cam0;
    const Vector3d &t_c_i = state_server.imu_state.t_cam0_imu;

    // Add a new camera state to the state server.
    // 1.2 取出imu旋转平移，按照外参，将这个时刻cam0的位姿算出来
    Matrix3d R_w_i = quaternionToRotation(
        state_server.imu_state.orientation);
    Matrix3d R_w_c = R_i_c * R_w_i;
    Vector3d t_c_w = state_server.imu_state.position +
                        R_w_i.transpose() * t_c_i;

    // 2. 注册新的相机状态到状态库中
    // 嗯。。。说人话就是找个记录的，不然咋更新
    state_server.cam_states[state_server.imu_state.id] =
        CAMState(state_server.imu_state.id);
    CAMState &cam_state = state_server.cam_states[state_server.imu_state.id];

    // 严格上讲这个时间不对，但是几乎没影响
    cam_state.time = time;
    cam_state.orientation = rotationToQuaternion(R_w_c);
    cam_state.position = t_c_w;

    // 记录第一次被估计的数据，不能被改变，因为改变了就破坏了之前的0空间
    cam_state.orientation_null = cam_state.orientation;
    cam_state.position_null = cam_state.position;

    // Update the covariance matrix of the state.
    // To simplify computation, the matrix J below is the nontrivial block
    // in Equation (16) in "A Multi-State Constraint Kalman Filter for Vision
    // -aided Inertial Navigation".


    // 3. 这个雅可比可以认为是cam0位姿相对于imu的状态量的求偏导
    // 此时我们首先要知道相机位姿是 Rcw  twc
    // Rcw = Rci * Riw   twc = twi + Rwi * tic
    Matrix<double, 6, 21> J = Matrix<double, 6, 21>::Zero();
    // Rcw对Riw的左扰动导数
    J.block<3, 3>(0, 0) = R_i_c;
    // Rcw对Rci的左扰动导数
    J.block<3, 3>(0, 15) = Matrix3d::Identity();

    // twc对Riw的左扰动导数
    // twc = twi + Rwi * Exp(φ) * tic
    //     = twi + Rwi * (I + φ^) * tic
    //     = twi + Rwi * tic + Rwi * φ^ * tic
    //     = twi + Rwi * tic - Rwi * tic^ * φ
    // 这部分的偏导为 -Rwi * tic^     与论文一致
    // TODO 试一下 -R_w_i.transpose() * skewSymmetric(t_c_i)
    // 其实这里可以反过来推一下当给的扰动是
    // twc = twi + Exp(-φ) * Rwi * tic
    //     = twi + (I - φ^) * Rwi * tic
    //     = twi + Rwi * tic - φ^ * Rwi * tic
    //     = twi + Rwi * tic + (Rwi * tic)^ * φ
    // 这样跟代码就一样了，但是上下定义的扰动方式就不同了
    J.block<3, 3>(3, 0) = skewSymmetric(R_w_i.transpose() * t_c_i);
    // 下面是代码里自带的，论文中给出的也是下面的结果
    // J.block<3, 3>(3, 0) = -R_w_i.transpose()*skewSymmetric(t_c_i);

    // twc对twi的左扰动导数
    J.block<3, 3>(3, 12) = Matrix3d::Identity();
    // twc对tic的左扰动导数
    J.block<3, 3>(3, 18) = R_w_i.transpose();

    // 4. 增广协方差矩阵
    // 简单地说就是原来的协方差是 21 + 6n 维的，现在新来了一个伙计，维度要扩了
    // 并且对应位置的值要根据雅可比跟这个时刻（也就是最新时刻）的imu协方差计算
    // 4.1 扩展矩阵大小 conservativeResize函数不改变原矩阵对应位置的数值
    // Resize the state covariance matrix.
    size_t old_rows = state_server.state_cov.rows();
    size_t old_cols = state_server.state_cov.cols();
    state_server.state_cov.conservativeResize(old_rows + 6, old_cols + 6);

    // Rename some matrix blocks for convenience.
    // imu的协方差矩阵
    const Matrix<double, 21, 21> &P11 =
        state_server.state_cov.block<21, 21>(0, 0);

    // imu相对于各个相机状态量的协方差矩阵（不包括最新的）
    const MatrixXd &P12 =
        state_server.state_cov.block(0, 21, 21, old_cols - 21);

    // Fill in the augmented state covariance.
    // 4.2 计算协方差矩阵
    // 左下角
    state_server.state_cov.block(old_rows, 0, 6, old_cols) << J * P11, J * P12;

    // 右上角
    state_server.state_cov.block(0, old_cols, old_rows, 6) =
        state_server.state_cov.block(old_rows, 0, 6, old_cols).transpose();

    // 右下角，关于相机部分的J都是0所以省略了
    state_server.state_cov.block<6, 6>(old_rows, old_cols) =
        J * P11 * J.transpose();

    // Fix the covariance to be symmetric
    // 强制对称
    MatrixXd state_cov_fixed = (state_server.state_cov +
                                state_server.state_cov.transpose()) /
                                2.0;
    state_server.state_cov = state_cov_fixed;

    return;
}

/**
 * @brief 添加特征点观测
 * @param  msg 前端发来的特征点信息，里面包含了时间，左右目上的角点及其id（严格意义上不能说是特征点）
 */
void MsckfVio::addFeatureObservations(
    const CameraMeasurementConstPtr &msg)
{
    // 这是个long long int 嗯。。。。直接当作int理解吧
    // 这个id会在 batchImuProcessing 更新
    StateIDType state_id = state_server.imu_state.id;

    // 1. 获取当前窗口内特征点数量
    int curr_feature_num = map_server.size();
    int tracked_feature_num = 0;

    // Add new observations for existing features or new
    // features in the map server.
    // 2. 添加新来的点，做的花里胡哨，其实就是在现有的特征管理里面找，
    // id已存在说明是跟踪的点，在已有的上面更新
    // id不存在说明新来的点，那么就新添加一个
    for (const auto &feature : msg->features)
    {
        if (map_server.find(feature.id) == map_server.end())
        {
            // This is a new feature.
            map_server[feature.id] = Feature(feature.id);
            map_server[feature.id].observations[state_id] =
                Vector4d(feature.u0, feature.v0,
                            feature.u1, feature.v1);
        }
        else
        {
            // This is an old feature.
            map_server[feature.id].observations[state_id] =
                Vector4d(feature.u0, feature.v0,
                            feature.u1, feature.v1);
            ++tracked_feature_num;
        }
    }

    // 这个东西计算了当前进来的跟踪的点中在总数里面的占比（进来的点有可能是新提的）
    tracking_rate =
        static_cast<double>(tracked_feature_num) /
        static_cast<double>(curr_feature_num);

    return;
}

/**
 * @brief 使用不再跟踪上的点来更新
 */
void MsckfVio::removeLostFeatures()
{
    // Remove the features that lost track.
    // BTW, find the size the final Jacobian matrix and residual vector.
    int jacobian_row_size = 0;
    // FeatureIDType 这是个long long int 嗯。。。。直接当作int理解吧
    vector<FeatureIDType> invalid_feature_ids(0);  // 无效点，最后要删的
    vector<FeatureIDType> processed_feature_ids(0);  // 待参与更新的点，用完也被无情的删掉

    // 遍历所有特征管理里面的点，包括新进来的
    for (auto iter = map_server.begin();
            iter != map_server.end(); ++iter)
    {
        // Rename the feature to be checked.
        // 引用，改变feature相当于改变iter->second，类似于指针的效果
        auto &feature = iter->second;

        // Pass the features that are still being tracked.
        // 1. 这个点被当前状态观测到，说明这个点后面还有可能被跟踪
        // 跳过这些点
        if (feature.observations.find(state_server.imu_state.id) !=
            feature.observations.end())
            continue;

        // 2. 跟踪小于3帧的点，认为是质量不高的点
        // 也好理解，三角化起码要两个观测，但是只有两个没有其他观测来验证
        if (feature.observations.size() < 3)
        {
            invalid_feature_ids.push_back(feature.id);
            continue;
        }

        // Check if the feature can be initialized if it
        // has not been.
        // 3. 如果这个特征没有被初始化，尝试去初始化
        // 初始化就是三角化
        if (!feature.is_initialized)
        {
            // 3.1 看看运动是否足够，没有足够视差或者平移小旋转多这种不符合三角化
            // 所以就不要这些点了
            if (!feature.checkMotion(state_server.cam_states))
            {
                invalid_feature_ids.push_back(feature.id);
                continue;
            }
            else
            {
                // 3.3 尝试三角化，失败也不要了
                if (!feature.initializePosition(state_server.cam_states))
                {
                    invalid_feature_ids.push_back(feature.id);
                    continue;
                }
            }
        }

        // 4. 到这里表示这个点能用于更新，所以准备下一步计算
        // 一个观测代表一帧，一帧有左右两个观测
        // 也就是算重投影误差时维度将会是4 * feature.observations.size()
        // 这里为什么减3下面会提到
        jacobian_row_size += 4 * feature.observations.size() - 3;
        // 接下来要参与优化的点加入到这个变量中
        processed_feature_ids.push_back(feature.id);
    }

    // cout << "invalid/processed feature #: " <<
    //   invalid_feature_ids.size() << "/" <<
    //   processed_feature_ids.size() << endl;
    // cout << "jacobian row #: " << jacobian_row_size << endl;

    // Remove the features that do not have enough measurements.
    // 5. 删掉非法点
    for (const auto &feature_id : invalid_feature_ids)
        map_server.erase(feature_id);

    // Return if there is no lost feature to be processed.
    if (processed_feature_ids.size() == 0)
        return;

    // 准备好误差相对于状态量的雅可比
    MatrixXd H_x = MatrixXd::Zero(jacobian_row_size,
                                    21 + 6 * state_server.cam_states.size());
    VectorXd r = VectorXd::Zero(jacobian_row_size);
    int stack_cntr = 0;

    // Process the features which lose track.
    // 6. 处理特征点
    for (const auto &feature_id : processed_feature_ids)
    {
        auto &feature = map_server[feature_id];

        vector<StateIDType> cam_state_ids(0);
        for (const auto &measurement : feature.observations)
            cam_state_ids.push_back(measurement.first);

        MatrixXd H_xj;
        VectorXd r_j;
        // 6.1 计算雅可比，计算重投影误差
        featureJacobian(feature.id, cam_state_ids, H_xj, r_j);

        // 6.2 卡方检验，剔除错误点，并不是所有点都用
        if (gatingTest(H_xj, r_j, cam_state_ids.size() - 1))
        {
            H_x.block(stack_cntr, 0, H_xj.rows(), H_xj.cols()) = H_xj;
            r.segment(stack_cntr, r_j.rows()) = r_j;
            stack_cntr += H_xj.rows();
        }

        // Put an upper bound on the row size of measurement Jacobian,
        // which helps guarantee the executation time.
        // 限制最大更新量
        if (stack_cntr > 1500)
            break;
    }

    // resize成实际大小
    H_x.conservativeResize(stack_cntr, H_x.cols());
    r.conservativeResize(stack_cntr);

    // Perform the measurement update step.
    // 7. 使用误差及雅可比更新状态
    measurementUpdate(H_x, r);

    // Remove all processed features from the map.
    // 8. 删除用完的点
    for (const auto &feature_id : processed_feature_ids)
        map_server.erase(feature_id);

    return;
}

/**
 * @brief 更新
 * @param  H 雅可比
 * @param  r 误差
 */
void MsckfVio::measurementUpdate(
    const MatrixXd &H, const VectorXd &r)
{

    if (H.rows() == 0 || r.rows() == 0)
        return;

    // Decompose the final Jacobian matrix to reduce computational
    // complexity as in Equation (28), (29).
    MatrixXd H_thin;
    VectorXd r_thin;

    if (H.rows() > H.cols())
    {
        // Convert H to a sparse matrix.
        SparseMatrix<double> H_sparse = H.sparseView();

        // Perform QR decompostion on H_sparse.
        // 利用H矩阵稀疏性，QR分解
        // 这段结合零空间投影一起理解，主要作用就是降低计算量
        SPQR<SparseMatrix<double>> spqr_helper;
        spqr_helper.setSPQROrdering(SPQR_ORDERING_NATURAL);
        spqr_helper.compute(H_sparse);

        MatrixXd H_temp;
        VectorXd r_temp;
        (spqr_helper.matrixQ().transpose() * H).evalTo(H_temp);
        (spqr_helper.matrixQ().transpose() * r).evalTo(r_temp);

        H_thin = H_temp.topRows(21 + state_server.cam_states.size() * 6);
        r_thin = r_temp.head(21 + state_server.cam_states.size() * 6);

        // HouseholderQR<MatrixXd> qr_helper(H);
        // MatrixXd Q = qr_helper.householderQ();
        // MatrixXd Q1 = Q.leftCols(21+state_server.cam_states.size()*6);

        // H_thin = Q1.transpose() * H;
        // r_thin = Q1.transpose() * r;
    }
    else
    {
        H_thin = H;
        r_thin = r;
    }

    // 2. 标准的卡尔曼计算过程
    // Compute the Kalman gain.
    const MatrixXd &P = state_server.state_cov;
    MatrixXd S = H_thin * P * H_thin.transpose() +
                    Feature::observation_noise * MatrixXd::Identity(
                                                    H_thin.rows(), H_thin.rows());
    // MatrixXd K_transpose = S.fullPivHouseholderQr().solve(H_thin*P);
    MatrixXd K_transpose = S.ldlt().solve(H_thin * P);
    MatrixXd K = K_transpose.transpose();

    // Compute the error of the state.
    VectorXd delta_x = K * r_thin;

    // Update the IMU state.
    const VectorXd &delta_x_imu = delta_x.head<21>();

    if ( // delta_x_imu.segment<3>(0).norm() > 0.15 ||
            // delta_x_imu.segment<3>(3).norm() > 0.15 ||
        delta_x_imu.segment<3>(6).norm() > 0.5 ||
        // delta_x_imu.segment<3>(9).norm() > 0.5 ||
        delta_x_imu.segment<3>(12).norm() > 1.0)
    {
        printf("delta velocity: %f\n", delta_x_imu.segment<3>(6).norm());
        printf("delta position: %f\n", delta_x_imu.segment<3>(12).norm());
        ROS_WARN("Update change is too large.");
        // return;
    }

    // 3. 更新到imu状态量
    const Vector4d dq_imu =
        smallAngleQuaternion(delta_x_imu.head<3>());
    // 相当于左乘dq_imu
    state_server.imu_state.orientation = quaternionMultiplication(
        dq_imu, state_server.imu_state.orientation);
    state_server.imu_state.gyro_bias += delta_x_imu.segment<3>(3);
    state_server.imu_state.velocity += delta_x_imu.segment<3>(6);
    state_server.imu_state.acc_bias += delta_x_imu.segment<3>(9);
    state_server.imu_state.position += delta_x_imu.segment<3>(12);

    // 外参
    const Vector4d dq_extrinsic =
        smallAngleQuaternion(delta_x_imu.segment<3>(15));
    state_server.imu_state.R_imu_cam0 =
        quaternionToRotation(dq_extrinsic) * state_server.imu_state.R_imu_cam0;
    state_server.imu_state.t_cam0_imu += delta_x_imu.segment<3>(18);

    // Update the camera states.
    // 更新相机姿态
    auto cam_state_iter = state_server.cam_states.begin();
    for (int i = 0; i < state_server.cam_states.size(); ++i, ++cam_state_iter)
    {
        const VectorXd &delta_x_cam = delta_x.segment<6>(21 + i * 6);
        const Vector4d dq_cam = smallAngleQuaternion(delta_x_cam.head<3>());
        cam_state_iter->second.orientation = quaternionMultiplication(
            dq_cam, cam_state_iter->second.orientation);
        cam_state_iter->second.position += delta_x_cam.tail<3>();
    }

    // Update state covariance.
    // 4. 更新协方差
    MatrixXd I_KH = MatrixXd::Identity(K.rows(), H_thin.cols()) - K * H_thin;
    // state_server.state_cov = I_KH*state_server.state_cov*I_KH.transpose() +
    //   K*K.transpose()*Feature::observation_noise;
    state_server.state_cov = I_KH * state_server.state_cov;

    // Fix the covariance to be symmetric
    MatrixXd state_cov_fixed = (state_server.state_cov +
                                state_server.state_cov.transpose()) /
                                2.0;
    state_server.state_cov = state_cov_fixed;

    return;
}

/**
 * @brief 计算一个路标点的雅可比
 * @param  feature_id 路标点id
 * @param  cam_state_ids 这个点对应的所有的相机状态id
 * @param  H_x 雅可比
 * @param  r 误差
 */
void MsckfVio::featureJacobian(
    const FeatureIDType &feature_id,
    const std::vector<StateIDType> &cam_state_ids,
    MatrixXd &H_x, VectorXd &r)
{
    // 取出特征
    const auto &feature = map_server[feature_id];

    // Check how many camera states in the provided camera
    // id camera has actually seen this feature.
    // 1. 统计有效观测的相机状态，因为对应的个别状态有可能被滑走了
    vector<StateIDType> valid_cam_state_ids(0);
    for (const auto &cam_id : cam_state_ids)
    {
        if (feature.observations.find(cam_id) ==
            feature.observations.end())
            continue;

        valid_cam_state_ids.push_back(cam_id);
    }

    int jacobian_row_size = 0;
    // 行数等于4*观测数量，一个观测在双目上都有，所以是2*2
    // 此时还没有0空间投影
    jacobian_row_size = 4 * valid_cam_state_ids.size();

    // 误差相对于状态量的雅可比，没有约束列数，因为列数一直是最新的
    MatrixXd H_xj = MatrixXd::Zero(jacobian_row_size,
                                    21 + state_server.cam_states.size() * 6);
    // 误差相对于三维点的雅可比
    MatrixXd H_fj = MatrixXd::Zero(jacobian_row_size, 3);
    // 误差
    VectorXd r_j = VectorXd::Zero(jacobian_row_size);
    int stack_cntr = 0;

    // 2. 计算每一个观测（同一帧左右目这里被叫成一个观测）的雅可比与误差
    for (const auto &cam_id : valid_cam_state_ids)
    {

        Matrix<double, 4, 6> H_xi = Matrix<double, 4, 6>::Zero();
        Matrix<double, 4, 3> H_fi = Matrix<double, 4, 3>::Zero();
        Vector4d r_i = Vector4d::Zero();
        // 2.1 计算一个左右目观测的雅可比
        measurementJacobian(cam_id, feature.id, H_xi, H_fi, r_i);

        // 计算这个cam_id在整个矩阵的列数，因为要在大矩阵里面放
        auto cam_state_iter = state_server.cam_states.find(cam_id);
        int cam_state_cntr = std::distance(
            state_server.cam_states.begin(), cam_state_iter);

        // Stack the Jacobians.
        H_xj.block<4, 6>(stack_cntr, 21 + 6 * cam_state_cntr) = H_xi;
        H_fj.block<4, 3>(stack_cntr, 0) = H_fi;
        r_j.segment<4>(stack_cntr) = r_i;
        stack_cntr += 4;
    }

    // Project the residual and Jacobians onto the nullspace
    // of H_fj.
    // 零空间投影
    JacobiSVD<MatrixXd> svd_helper(H_fj, ComputeFullU | ComputeThinV);
    MatrixXd A = svd_helper.matrixU().rightCols(
        jacobian_row_size - 3);

    // 上面的效果跟QR分解一样，下面的代码可以测试打印对比
    // Eigen::ColPivHouseholderQR<MatrixXd> qr(H_fj);
	// MatrixXd Q = qr.matrixQ();
    // std::cout << "spqr_helper.matrixQ(): " << std::endl << Q << std::endl << std::endl;
    // std::cout << "A: " << std::endl << A << std::endl;

    // 0空间投影
    H_x = A.transpose() * H_xj;
    r = A.transpose() * r_j;

    return;
}

/**
 * @brief 计算一个路标点的雅可比
 * @param  cam_state_id 有效的相机状态id
 * @param  feature_id 路标点id
 * @param  H_x 误差相对于位姿的雅可比
 * @param  H_f 误差相对于三维点的雅可比
 * @param  r 误差
 */
void MsckfVio::measurementJacobian(
    const StateIDType &cam_state_id,
    const FeatureIDType &feature_id,
    Matrix<double, 4, 6> &H_x, Matrix<double, 4, 3> &H_f, Vector4d &r)
{

    // Prepare all the required data.
    // 1. 取出相机状态与特征
    const CAMState &cam_state = state_server.cam_states[cam_state_id];
    const Feature &feature = map_server[feature_id];

    // 2. 取出左目位姿，根据外参计算右目位姿
    // Cam0 pose.
    Matrix3d R_w_c0 = quaternionToRotation(cam_state.orientation);
    const Vector3d &t_c0_w = cam_state.position;

    // Cam1 pose.
    Matrix3d R_c0_c1 = CAMState::T_cam0_cam1.linear();
    Matrix3d R_w_c1 = CAMState::T_cam0_cam1.linear() * R_w_c0;
    Vector3d t_c1_w = t_c0_w - R_w_c1.transpose() * CAMState::T_cam0_cam1.translation();

    // 3. 取出三维点坐标与归一化的坐标点，因为前端发来的是归一化坐标的
    // 3d feature position in the world frame.
    // And its observation with the stereo cameras.
    const Vector3d &p_w = feature.position;
    const Vector4d &z = feature.observations.find(cam_state_id)->second;

    // 4. 转到左右目相机坐标系下
    // Convert the feature position from the world frame to
    // the cam0 and cam1 frame.
    Vector3d p_c0 = R_w_c0 * (p_w - t_c0_w);
    Vector3d p_c1 = R_w_c1 * (p_w - t_c1_w);
    // p_c1 = R_c0_c1 * R_w_c0 * (p_w - t_c0_w + R_w_c1.transpose() * t_cam0_cam1)
    //      = R_c0_c1 * (p_c0 + R_w_c0 * R_w_c1.transpose() * t_cam0_cam1)
    //      = R_c0_c1 * (p_c0 + R_c0_c1 * t_cam0_cam1)

    // Compute the Jacobians.
    // 5. 计算雅可比
    // 左相机归一化坐标点相对于左相机坐标系下的点的雅可比
    // (x, y) = (X / Z, Y / Z)
    Matrix<double, 4, 3> dz_dpc0 = Matrix<double, 4, 3>::Zero();
    dz_dpc0(0, 0) = 1 / p_c0(2);
    dz_dpc0(1, 1) = 1 / p_c0(2);
    dz_dpc0(0, 2) = -p_c0(0) / (p_c0(2) * p_c0(2));
    dz_dpc0(1, 2) = -p_c0(1) / (p_c0(2) * p_c0(2));

    // 与上同理
    Matrix<double, 4, 3> dz_dpc1 = Matrix<double, 4, 3>::Zero();
    dz_dpc1(2, 0) = 1 / p_c1(2);
    dz_dpc1(3, 1) = 1 / p_c1(2);
    dz_dpc1(2, 2) = -p_c1(0) / (p_c1(2) * p_c1(2));
    dz_dpc1(3, 2) = -p_c1(1) / (p_c1(2) * p_c1(2));

    // 左相机坐标系下的三维点相对于左相机位姿的雅可比 先r后t
    Matrix<double, 3, 6> dpc0_dxc = Matrix<double, 3, 6>::Zero();
    dpc0_dxc.leftCols(3) = skewSymmetric(p_c0);
    dpc0_dxc.rightCols(3) = -R_w_c0;

    // 右相机坐标系下的三维点相对于左相机位姿的雅可比 先r后t
    Matrix<double, 3, 6> dpc1_dxc = Matrix<double, 3, 6>::Zero();
    dpc1_dxc.leftCols(3) = R_c0_c1 * skewSymmetric(p_c0);
    dpc1_dxc.rightCols(3) = -R_w_c1;

    // Vector3d p_c0 = R_w_c0 * (p_w - t_c0_w);
    // Vector3d p_c1 = R_w_c1 * (p_w - t_c1_w);
    // p_c0 对 p_w
    Matrix3d dpc0_dpg = R_w_c0;
    // p_c1 对 p_w
    Matrix3d dpc1_dpg = R_w_c1;

    // 两个雅可比
    H_x = dz_dpc0 * dpc0_dxc + dz_dpc1 * dpc1_dxc;
    H_f = dz_dpc0 * dpc0_dpg + dz_dpc1 * dpc1_dpg;

    // Modifty the measurement Jacobian to ensure
    // observability constrain.
    // 6. OC
    Matrix<double, 4, 6> A = H_x;
    Matrix<double, 6, 1> u = Matrix<double, 6, 1>::Zero();
    u.block<3, 1>(0, 0) = 
        quaternionToRotation(cam_state.orientation_null) * IMUState::gravity;
    u.block<3, 1>(3, 0) =
        skewSymmetric(p_w - cam_state.position_null) * IMUState::gravity;
    H_x = A - A * u * (u.transpose() * u).inverse() * u.transpose();
    H_f = -H_x.block<4, 3>(0, 3);

    // Compute the residual.
    // 7. 计算归一化平面坐标误差
    r = z - Vector4d(p_c0(0) / p_c0(2), p_c0(1) / p_c0(2),
                        p_c1(0) / p_c1(2), p_c1(1) / p_c1(2));

    return;
}


/**
 * @brief 当cam状态数达到最大值时，挑出若干cam状态待删除
 */
void MsckfVio::pruneCamStateBuffer()
{
    // 数量还不到该删的程度，配置文件里面是20个
    if (state_server.cam_states.size() < max_cam_state_size)
        return;

    // Find two camera states to be removed.
    // 1. 找出该删的相机状态的id，两个
    vector<StateIDType> rm_cam_state_ids(0);
    findRedundantCamStates(rm_cam_state_ids);

    // Find the size of the Jacobian matrix.
    // 2. 找到删减帧涉及的观测雅可比大小
    int jacobian_row_size = 0;
    for (auto &item : map_server)
    {
        auto &feature = item.second;
        // Check how many camera states to be removed are associated
        // with this feature.
        // 2.1 在待删去的帧中统计能观测到这个特征的帧
        vector<StateIDType> involved_cam_state_ids(0);
        for (const auto &cam_id : rm_cam_state_ids)
        {
            if (feature.observations.find(cam_id) !=
                feature.observations.end())
                involved_cam_state_ids.push_back(cam_id);
        }

        if (involved_cam_state_ids.size() == 0)
            continue;
        // 2.2 这个点只在一个里面有观测那就直接删
        // 只用一个观测更新不了状态
        if (involved_cam_state_ids.size() == 1)
        {
            feature.observations.erase(involved_cam_state_ids[0]);
            continue;
        }
        // 程序到这里的时候说明找到了一个特征，先不说他一共被几帧观测到
        // 到这里说明被两帧或两帧以上待删减的帧观测到
        // 2.3 如果没有做过三角化，做一下三角化，如果失败直接删
        if (!feature.is_initialized)
        {
            // Check if the feature can be initialize.
            if (!feature.checkMotion(state_server.cam_states))
            {
                // If the feature cannot be initialized, just remove
                // the observations associated with the camera states
                // to be removed.
                for (const auto &cam_id : involved_cam_state_ids)
                    feature.observations.erase(cam_id);
                continue;
            }
            else
            {
                if (!feature.initializePosition(state_server.cam_states))
                {
                    for (const auto &cam_id : involved_cam_state_ids)
                        feature.observations.erase(cam_id);
                    continue;
                }
            }
        }

        // 2.4 最后的最后得出了行数
        // 意味着有involved_cam_state_ids.size() 数量的观测要被删去
        // 但是因为待删去的帧间有共同观测的关系，直接删会损失这部分信息
        // 所以临删前做最后一次更新
        jacobian_row_size += 4 * involved_cam_state_ids.size() - 3;
    }

    // cout << "jacobian row #: " << jacobian_row_size << endl;

    // Compute the Jacobian and residual.
    // 3. 计算待删掉的这部分观测的雅可比与误差
    // 预设大小
    MatrixXd H_x = MatrixXd::Zero(jacobian_row_size,
                                    21 + 6 * state_server.cam_states.size());
    VectorXd r = VectorXd::Zero(jacobian_row_size);
    int stack_cntr = 0;

    // 又做了一遍类似上面的遍历，只不过该三角化的已经三角化，该删的已经删了
    for (auto &item : map_server)
    {
        auto &feature = item.second;
        // Check how many camera states to be removed are associated
        // with this feature.
        // 这段就是判断一下这个点是否都在待删除帧中有观测
        vector<StateIDType> involved_cam_state_ids(0);
        for (const auto &cam_id : rm_cam_state_ids)
        {
            if (feature.observations.find(cam_id) !=
                feature.observations.end())
                involved_cam_state_ids.push_back(cam_id);
        }

        // 一个的情况已经被删掉了
        if (involved_cam_state_ids.size() == 0)
            continue;

        // 计算出待删去的这部分的雅可比
        // 这个点假如有多个观测，但本次只用待删除帧上的观测
        MatrixXd H_xj;
        VectorXd r_j;
        featureJacobian(feature.id, involved_cam_state_ids, H_xj, r_j);

        if (gatingTest(H_xj, r_j, involved_cam_state_ids.size()))
        {
            H_x.block(stack_cntr, 0, H_xj.rows(), H_xj.cols()) = H_xj;
            r.segment(stack_cntr, r_j.rows()) = r_j;
            stack_cntr += H_xj.rows();
        }

        // 删去观测
        for (const auto &cam_id : involved_cam_state_ids)
            feature.observations.erase(cam_id);
    }

    H_x.conservativeResize(stack_cntr, H_x.cols());
    r.conservativeResize(stack_cntr);

    // Perform measurement update.
    // 4. 用待删去的这些观测更新一下
    measurementUpdate(H_x, r);

    // 5. 直接删掉对应的行列，直接干掉
    // 为啥没有做类似于边缘化的操作？
    // 个人认为是上面做最后的更新了，所以信息已经更新到了各个地方
    for (const auto &cam_id : rm_cam_state_ids)
    {
        int cam_sequence = std::distance(
            state_server.cam_states.begin(), state_server.cam_states.find(cam_id));
        int cam_state_start = 21 + 6 * cam_sequence;
        int cam_state_end = cam_state_start + 6;

        // Remove the corresponding rows and columns in the state
        // covariance matrix.
        if (cam_state_end < state_server.state_cov.rows())
        {
            state_server.state_cov.block(cam_state_start, 0,
                                            state_server.state_cov.rows() - cam_state_end,
                                            state_server.state_cov.cols()) =
                state_server.state_cov.block(cam_state_end, 0,
                                                state_server.state_cov.rows() - cam_state_end,
                                                state_server.state_cov.cols());

            state_server.state_cov.block(0, cam_state_start,
                                            state_server.state_cov.rows(),
                                            state_server.state_cov.cols() - cam_state_end) =
                state_server.state_cov.block(0, cam_state_end,
                                                state_server.state_cov.rows(),
                                                state_server.state_cov.cols() - cam_state_end);

            state_server.state_cov.conservativeResize(
                state_server.state_cov.rows() - 6, state_server.state_cov.cols() - 6);
        }
        else
        {
            state_server.state_cov.conservativeResize(
                state_server.state_cov.rows() - 6, state_server.state_cov.cols() - 6);
        }

        // Remove this camera state in the state vector.
        state_server.cam_states.erase(cam_id);
    }

    return;
}

/**
 * @brief 找出该删的相机状态的id
 * @param  rm_cam_state_ids 要删除的相机状态id
 */
void MsckfVio::findRedundantCamStates(
    vector<StateIDType> &rm_cam_state_ids)
{
    // Move the iterator to the key position.
    // 1. 找到倒数第四个相机状态，作为关键状态
    auto key_cam_state_iter = state_server.cam_states.end();
    for (int i = 0; i < 4; ++i)
        --key_cam_state_iter;

    // 倒数第三个相机状态
    auto cam_state_iter = key_cam_state_iter;
    ++cam_state_iter;

    // 序列中，第一个相机状态
    auto first_cam_state_iter = state_server.cam_states.begin();

    // Pose of the key camera state.
    // 2. 关键状态的位姿
    const Vector3d key_position =
        key_cam_state_iter->second.position;
    const Matrix3d key_rotation = quaternionToRotation(
        key_cam_state_iter->second.orientation);

    // Mark the camera states to be removed based on the
    // motion between states.
    // 3. 遍历两次，必然删掉两个状态，有可能是相对新的，有可能是最旧的
    // 但是永远删不到最新的
    for (int i = 0; i < 2; ++i)
    {
        // 从倒数第三个开始
        const Vector3d position =
            cam_state_iter->second.position;
        const Matrix3d rotation = quaternionToRotation(
            cam_state_iter->second.orientation);

        // 计算相对于关键相机状态的平移与旋转
        double distance = (position - key_position).norm();
        double angle = AngleAxisd(rotation * key_rotation.transpose()).angle();

        // 判断大小以及跟踪率，就是cam_state_iter这个状态与关键相机状态的相似度，
        // 且当前的点跟踪率很高
        // 删去这个帧，否则删掉最老的
        if (angle < rotation_threshold &&
            distance < translation_threshold &&
            tracking_rate > tracking_rate_threshold)
        {
            rm_cam_state_ids.push_back(cam_state_iter->first);
            ++cam_state_iter;
        }
        else
        {
            rm_cam_state_ids.push_back(first_cam_state_iter->first);
            ++first_cam_state_iter;
        }
    }

    // Sort the elements in the output vector.
    // 4. 排序
    sort(rm_cam_state_ids.begin(), rm_cam_state_ids.end());

    return;
}

// 卡方检验，这部分有点乱
bool MsckfVio::gatingTest(
    const MatrixXd &H, const VectorXd &r, const int &dof)
{
    // 输入的dof的值是所有相机观测，且没有去掉滑窗的
    // 而且按照维度这个卡方的维度也不对
    // 
    MatrixXd P1 = H * state_server.state_cov * H.transpose();
    MatrixXd P2 = Feature::observation_noise *
                    MatrixXd::Identity(H.rows(), H.rows());
    double gamma = r.transpose() * (P1 + P2).ldlt().solve(r);

    // cout << dof << " " << gamma << " " <<
    //   chi_squared_test_table[dof] << " ";

    if (gamma < chi_squared_test_table[dof])
    {
        // cout << "passed" << endl;
        return true;
    }
    else
    {
        // cout << "failed" << endl;
        return false;
    }
}

void MsckfVio::onlineReset()
{

    // Never perform online reset if position std threshold
    // is non-positive.
    if (position_std_threshold <= 0)
        return;
    static long long int online_reset_counter = 0;

    // Check the uncertainty of positions to determine if
    // the system can be reset.
    double position_x_std = std::sqrt(state_server.state_cov(12, 12));
    double position_y_std = std::sqrt(state_server.state_cov(13, 13));
    double position_z_std = std::sqrt(state_server.state_cov(14, 14));

    if (position_x_std < position_std_threshold &&
        position_y_std < position_std_threshold &&
        position_z_std < position_std_threshold)
        return;

    ROS_WARN("Start %lld online reset procedure...",
                ++online_reset_counter);
    ROS_INFO("Stardard deviation in xyz: %f, %f, %f",
                position_x_std, position_y_std, position_z_std);

    // Remove all existing camera states.
    state_server.cam_states.clear();

    // Clear all exsiting features in the map.
    map_server.clear();

    // Reset the state covariance.
    double gyro_bias_cov, acc_bias_cov, velocity_cov;
    nh.param<double>("initial_covariance/velocity",
                        velocity_cov, 0.25);
    nh.param<double>("initial_covariance/gyro_bias",
                        gyro_bias_cov, 1e-4);
    nh.param<double>("initial_covariance/acc_bias",
                        acc_bias_cov, 1e-2);

    double extrinsic_rotation_cov, extrinsic_translation_cov;
    nh.param<double>("initial_covariance/extrinsic_rotation_cov",
                        extrinsic_rotation_cov, 3.0462e-4);
    nh.param<double>("initial_covariance/extrinsic_translation_cov",
                        extrinsic_translation_cov, 1e-4);

    state_server.state_cov = MatrixXd::Zero(21, 21);
    for (int i = 3; i < 6; ++i)
        state_server.state_cov(i, i) = gyro_bias_cov;
    for (int i = 6; i < 9; ++i)
        state_server.state_cov(i, i) = velocity_cov;
    for (int i = 9; i < 12; ++i)
        state_server.state_cov(i, i) = acc_bias_cov;
    for (int i = 15; i < 18; ++i)
        state_server.state_cov(i, i) = extrinsic_rotation_cov;
    for (int i = 18; i < 21; ++i)
        state_server.state_cov(i, i) = extrinsic_translation_cov;

    ROS_WARN("%lld online reset complete...", online_reset_counter);
    return;
}

void MsckfVio::publish(const ros::Time &time)
{

    // Convert the IMU frame to the body frame.
    // 1. 计算body坐标，因为imu与body相对位姿是单位矩阵，所以就是imu的坐标
    const IMUState &imu_state = state_server.imu_state;
    Eigen::Isometry3d T_i_w = Eigen::Isometry3d::Identity();
    T_i_w.linear() = quaternionToRotation(
                            imu_state.orientation)
                            .transpose();
    T_i_w.translation() = imu_state.position;

    Eigen::Isometry3d T_b_w = IMUState::T_imu_body * T_i_w *
                                IMUState::T_imu_body.inverse();
    Eigen::Vector3d body_velocity =
        IMUState::T_imu_body.linear() * imu_state.velocity;

    // Publish tf
    // 2. 发布tf，实时的位姿，没有轨迹，没有协方差
    if (publish_tf)
    {
        tf::Transform T_b_w_tf;
        tf::transformEigenToTF(T_b_w, T_b_w_tf);
        tf_pub.sendTransform(tf::StampedTransform(
            T_b_w_tf, time, fixed_frame_id, child_frame_id));
    }

    // Publish the odometry
    // 3. 发布位姿，能在rviz留下轨迹的
    nav_msgs::Odometry odom_msg;
    odom_msg.header.stamp = time;
    odom_msg.header.frame_id = fixed_frame_id;
    odom_msg.child_frame_id = child_frame_id;

    tf::poseEigenToMsg(T_b_w, odom_msg.pose.pose);
    tf::vectorEigenToMsg(body_velocity, odom_msg.twist.twist.linear);

    // Convert the covariance.
    // 协方差，取出旋转平移部分，以及它们之间的公共部分组成6自由度的协方差
    Matrix3d P_oo = state_server.state_cov.block<3, 3>(0, 0);
    Matrix3d P_op = state_server.state_cov.block<3, 3>(0, 12);
    Matrix3d P_po = state_server.state_cov.block<3, 3>(12, 0);
    Matrix3d P_pp = state_server.state_cov.block<3, 3>(12, 12);
    Matrix<double, 6, 6> P_imu_pose = Matrix<double, 6, 6>::Zero();
    P_imu_pose << P_pp, P_po, P_op, P_oo;

    // 转下坐标，但是这里都是单位矩阵
    Matrix<double, 6, 6> H_pose = Matrix<double, 6, 6>::Zero();
    H_pose.block<3, 3>(0, 0) = IMUState::T_imu_body.linear();
    H_pose.block<3, 3>(3, 3) = IMUState::T_imu_body.linear();
    Matrix<double, 6, 6> P_body_pose = H_pose *
                                        P_imu_pose * H_pose.transpose();

    // 填充协方差
    for (int i = 0; i < 6; ++i)
        for (int j = 0; j < 6; ++j)
            odom_msg.pose.covariance[6 * i + j] = P_body_pose(i, j);

    // Construct the covariance for the velocity.
    // 速度协方差
    Matrix3d P_imu_vel = state_server.state_cov.block<3, 3>(6, 6);
    Matrix3d H_vel = IMUState::T_imu_body.linear();
    Matrix3d P_body_vel = H_vel * P_imu_vel * H_vel.transpose();
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            odom_msg.twist.covariance[i * 6 + j] = P_body_vel(i, j);

    // 发布位姿
    odom_pub.publish(odom_msg);

    // Publish the 3D positions of the features that
    // has been initialized.
    // 4. 发布点云
    boost::shared_ptr<pcl::PointCloud<pcl::PointXYZ>> feature_msg_ptr(
        new pcl::PointCloud<pcl::PointXYZ>());
    feature_msg_ptr->header.frame_id = fixed_frame_id;
    feature_msg_ptr->height = 1;
    for (const auto &item : map_server)
    {
        const auto &feature = item.second;
        if (feature.is_initialized)
        {
            Vector3d feature_position =
                IMUState::T_imu_body.linear() * feature.position;
            feature_msg_ptr->points.push_back(pcl::PointXYZ(
                feature_position(0), feature_position(1), feature_position(2)));
        }
    }
    feature_msg_ptr->width = feature_msg_ptr->points.size();

    feature_pub.publish(feature_msg_ptr);


    // // 5、 2023-04-26 将读取的GPS速度和位置消息类型输出
    // gps_pub.publish(GPS_buffer);
    // gps_vel_pub.publish(GPS_vel_buffer);


    return;
}

// // GPS fusion part
// void MsckfVio::GPSCallback(const sensor_msgs::NavSatFix::Ptr & gps_msg)
// {
//     double time = gps_msg->header.stamp.toSec();
//     sensor_msgs::NavSatFix GNSSposi;
//     GNSSposi.longitude=gps_msg->longitude;
//     GNSSposi.latitude=gps_msg->latitude;
//     GNSSposi.altitude=gps_msg->altitude;
//     // auto msg_modified = msg;
//     // msg_modified.altitude = 21.0;
//     // GPS_buffer.push_back(make_pair(time,msg_modified));
//     GPS_buffer.push_back(make_pair(time,GNSSposi));
//     // state_server.imu_state.R_n_e = geo_utils::CoordTrans::getRne(Vector3d(msg_modified.latitude,
//     //                                                                       msg_modified.longitude,
//     //                                                                       msg_modified.altitude)).transpose();
//     /*
//     ofsgps<<setprecision(20)<<time<<" "<<
//             setprecision(15)<<msg_modified.latitude<<" "<<
//             setprecision(15)<<msg_modified.longitude<<" "<<
//             setprecision(15)<<msg_modified.altitude<<" "<<endl;
//             */
//     gps_pub.publish(GNSSposi);

// }

// // void MsckfVio::GPSVelCallback(const geometry_msgs::Twist::Ptr & velo_msg)
// void MsckfVio::GPSVelCallback(const forsense_msg::RtkVelocity::Ptr & velo_msg)
// {    
//     double time = velo_msg->header.stamp.toSec();
//     forsense_msg::RtkVelocity GNSSvelo;
//     GNSSvelo.twist.linear.x = velo_msg->twist.linear.x;
//     GNSSvelo.twist.linear.y = velo_msg->twist.linear.y;
//     GNSSvelo.twist.linear.z = velo_msg->twist.linear.z;

//     GPS_vel_buffer.push_back(make_pair(time,GNSSvelo));

// }


} // namespace msckf_vio
