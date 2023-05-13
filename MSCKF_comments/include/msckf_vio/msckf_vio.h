/*
 * COPYRIGHT AND PERMISSION NOTICE
 * Penn Software MSCKF_VIO
 * Copyright (C) 2017 The Trustees of the University of Pennsylvania
 * All rights reserved.
 */

#ifndef MSCKF_VIO_H
#define MSCKF_VIO_H

#include <map>
#include <set>
#include <vector>
#include <string>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <boost/shared_ptr.hpp>

#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <nav_msgs/Odometry.h>
// 头文件的索引（为GPS）
#include <sensor_msgs/NavSatFix.h>
// #include <geometry_msgs/TwistWithCovarianceStamped.h>  //这是旧的，不用这个头文件
//增加对新消息RtkVelocity的索引
// #include <forsense_msg/RtkVelocity.h>
// 2023-04-29 这里改为旧的数据接口
#include <forsense_msg/Forsense.h>
// 2023-05-04 用于轨迹显示
#include <nav_msgs/Path.h>
// 2023-05-12 用于618Dpro新消息的真值获取
#include <nav_msgs/Odometry.h>

#include <tf/transform_broadcaster.h>
#include <std_srvs/Trigger.h>

#include "imu_state.h"
#include "cam_state.h"
#include "feature.hpp"
#include <msckf_vio/CameraMeasurement.h>

#define deg2rad 0.017453292519943
#define rad2deg 57.295779513082323

namespace msckf_vio
{
/*
 * @brief MsckfVio Implements the algorithm in
 *    Anatasios I. Mourikis, and Stergios I. Roumeliotis,
 *    "A Multi-State Constraint Kalman Filter for Vision-aided
 *    Inertial Navigation",
 *    http://www.ee.ucr.edu/~mourikis/tech_reports/TR_MSCKF.pdf
 */

// 2023-04-26 新增GPS的信息结构
struct GPS_msg{
    long long timestamp;
    Eigen::Vector3d position;
    Eigen::Vector3d velocity;

};

class MsckfVio
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    // Constructor
    MsckfVio(ros::NodeHandle &pnh);
    // Disable copy and assign constructor
    MsckfVio(const MsckfVio &) = delete;
    MsckfVio operator=(const MsckfVio &) = delete;

    // Destructor
    ~MsckfVio() {}

    /*
     * @brief initialize Initialize the VIO.
     */
    bool initialize();

    /*
     * @brief reset Resets the VIO to initial status.
     */
    void reset();

    typedef boost::shared_ptr<MsckfVio> Ptr;
    typedef boost::shared_ptr<const MsckfVio> ConstPtr;

private:
    /*
     * @brief StateServer Store one IMU states and several
     *    camera states for constructing measurement
     *    model.
     */
    struct StateServer
    {
        IMUState imu_state;
        // 别看他长，其实就是一个map类
        // key是 StateIDType 由 long long int typedef而来，把它当作int看就行
        // value是CAMState
        CamStateServer cam_states;

        // State covariance matrix
        Eigen::MatrixXd state_cov;
        Eigen::Matrix<double, 12, 12> continuous_noise_cov;
    };

    /*
     * @brief loadParameters
     *    Load parameters from the parameter server.
     */
    bool loadParameters();

    /*
     * @brief createRosIO
     *    Create ros publisher and subscirbers.
     */
    bool createRosIO();

    //for GPS fusion,GPS数据的callback
    // void GPSCallback(const sensor_msgs::NavSatFix::Ptr &gps_msg);
    // void GPSVelCallback(const forsense_msg::RtkVelocity::Ptr &velo_msg);
    // // 2023-04-29 使用原有的消息类型接口
    // void GPSCallback(const forsense_msg::Forsense::Ptr &gps_msg);




    /*
     * @brief imuCallback
     *    Callback function for the imu message.
     * @param msg IMU msg.
     */
    void imuCallback(const sensor_msgs::ImuConstPtr &msg);

    /*
     * @brief featureCallback
     *    Callback function for feature measurements.
     * @param msg Stereo feature measurements.
     */
    void featureCallback(const CameraMeasurementConstPtr &msg);

    /*
     * @brief publish Publish the results of VIO.
     * @param time The time stamp of output msgs.
     */
    void publish(const ros::Time &time);

    /*
     * @brief initializegravityAndBias
     *    Initialize the IMU bias and initial orientation
     *    based on the first few IMU readings.
     */
    void initializeGravityAndBias();

    /*
     * @biref resetCallback
     *    Callback function for the reset service.
     *    Note that this is NOT anytime-reset. This function should
     *    only be called before the sensor suite starts moving.
     *    e.g. while the robot is still on the ground.
     */
    bool resetCallback(std_srvs::Trigger::Request &req,
                        std_srvs::Trigger::Response &res);

    // Filter related functions
    // Propogate the state
    void batchImuProcessing(
        const double &time_bound);
    void processModel(const double &time, const Eigen::Vector3d &m_gyro, const Eigen::Vector3d &m_acc);
    void predictNewState(const double &dt, const Eigen::Vector3d &gyro, const Eigen::Vector3d &acc);

    // Measurement update
    void stateAugmentation(const double &time);
    void addFeatureObservations(const CameraMeasurementConstPtr &msg);
    // This function is used to compute the measurement Jacobian
    // for a single feature observed at a single camera frame.
    void measurementJacobian(const StateIDType &cam_state_id,
        const FeatureIDType &feature_id,
        Eigen::Matrix<double, 4, 6> &H_x,
        Eigen::Matrix<double, 4, 3> &H_f,
        Eigen::Vector4d &r);
    // This function computes the Jacobian of all measurements viewed
    // in the given camera states of this feature.
    void featureJacobian(const FeatureIDType &feature_id, const std::vector<StateIDType> &cam_state_ids, Eigen::MatrixXd &H_x, Eigen::VectorXd &r);
    void measurementUpdate(const Eigen::MatrixXd &H, const Eigen::VectorXd &r);
    bool gatingTest(const Eigen::MatrixXd &H, const Eigen::VectorXd &r, const int &dof);
    void removeLostFeatures();
    void findRedundantCamStates(std::vector<StateIDType> &rm_cam_state_ids);
    void pruneCamStateBuffer();
    // Reset the system online if the uncertainty is too large.
    void onlineReset();

    // Chi squared test table.
    static std::map<int, double> chi_squared_test_table;

    // State vector
    StateServer state_server;
    // Maximum number of camera states
    int max_cam_state_size;

    // Features used
    // 理解为一种特征管理，本质是一个map，key为特征点id，value为特征类
    MapServer map_server;

    // IMU data buffer
    // This is buffer is used to handle the unsynchronization or
    // transfer delay between IMU and Image messages.
    std::vector<sensor_msgs::Imu> imu_msg_buffer;

    // Indicate if the gravity vector is set.
    bool is_gravity_set;

    // Indicate if the received image is the first one. The
    // system will start after receiving the first image.
    bool is_first_img;

    // 2023-04-30 新增判断是否初始化完成（也就是参考值是否赋值作为初值）
    bool is_state_initial;

    // 2023-05-12 新增判断IMU陀螺是否初始化完毕
    bool is_gyrobias_set;

    // 2023-05-13 新增判断是否第一次GPS消息，便于对state_serve的时间进行赋值
    bool is_first_GPS;

    // The position uncertainty threshold is used to determine
    // when to reset the system online. Otherwise, the ever-
    // increaseing uncertainty will make the estimation unstable.
    // Note this online reset will be some dead-reckoning.
    // Set this threshold to nonpositive to disable online reset.
    double position_std_threshold;

    // Tracking rate
    double tracking_rate;

    // Threshold for determine keyframes
    double translation_threshold;
    double rotation_threshold;
    double tracking_rate_threshold;

    // Ros node handle
    ros::NodeHandle nh;

    // Subscribers and publishers
    ros::Subscriber imu_sub;
    ros::Subscriber feature_sub;
    ros::Publisher odom_pub;
    ros::Publisher feature_pub;
    tf::TransformBroadcaster tf_pub;
    ros::ServiceServer reset_srv;

    // Frame id
    std::string fixed_frame_id;
    std::string child_frame_id;

    // Whether to publish tf or not.
    bool publish_tf;

    // Framte rate of the stereo images. This variable is
    // only used to determine the timing threshold of
    // each iteration of the filter.
    double frame_rate;

    // Debugging variables and functions
    void mocapOdomCallback(const nav_msgs::OdometryConstPtr &msg);

    ros::Subscriber mocap_odom_sub;
    ros::Publisher mocap_odom_pub;
    geometry_msgs::TransformStamped raw_mocap_odom_msg;
    Eigen::Isometry3d mocap_initial_frame;

    // 2023-04-26 加入GPS的发布节点 subscribe 卫星的位置和速度（只将消息进行发布测试）
    // 2023-04-29 实际上这里是直接通过该sub将618Dpro的数据全部拿过来
    ros::Subscriber gps_sub;
    // ros::Subscriber gps_vel_sub;

    // 基于新的消息，将真值读取进来
    ros::Subscriber Tru_sub;

    // //temporarily//基于消息定义的GPS位置和速度buffer（注意要改这里的消息类型）
    // std::list<std::pair<double,sensor_msgs::NavSatFix>> GPS_buffer;
    // // std::list<std::pair<double,geometry_msgs::TwistWithCovarianceStamped>> GPS_vel_buffer;
    // // 这里重新改为自定义的新的消息类型
    // std::list<std::pair<double,forsense_msg::RtkVelocity>> GPS_vel_buffer;

    // 2023-04-29 重新对应上原有的618Dpro的消息并进行定义
    std::list<std::pair<double,forsense_msg::Forsense>> GPS_buffer;
    std::list<std::pair<double,forsense_msg::Forsense>> GPS_velo_buffer;
    std::list<std::pair<double,forsense_msg::Forsense>> Tru_buffer;
    // std::list<std::pair<double,forsense_msg::Forsense>> IMU_couter_buffer; // 用于对IMU数据计数以满足初始化的条件处理
      // IMU message buffer.
    std::list<forsense_msg::Forsense> IMU_couter_buffer; // 用于对IMU数据计数以满足初始化的条件处理
    std::list<forsense_msg::Forsense>  Tru_couter_buffer; // 拿真值的数据进行初始化赋值

    // //for GPS fusion,GPS数据的callback
    // void GPSCallback(const sensor_msgs::NavSatFix::Ptr &gps_msg);
    // void GPSVelCallback(const forsense_msg::RtkVelocity::Ptr &velo_msg);

    // 临时加一个pub，将读取的GPS速度和位置进行发布
    ros::Publisher gps_pub;
    ros::Publisher gps_vel_pub;

    // 2023-04-29 新增参考点的真值记录，pub出去看是否正确
    ros::Publisher Tru_pub;
    ros::Publisher Tru_vel_pub;
    ros::Publisher Tru_atti_pub;

    // 2023-05-02 新增一个轨迹pub
    ros::Publisher path_pub;
    // 2023-05-03 
    nav_msgs::Path path_msg;  // 将path定义在外部，避免"每次给这个path赋值,都是一个新的path,也就出现了很多poses,而每个poses只有一个pose."
    // nav_msgs::Odometry Tru_msg;   // 新增一个真值轨迹，用于与解算轨迹对比
    nav_msgs::Path Tru_msg;   // 新增一个真值轨迹，用于与解算轨迹对比


    // // ==================================以下是新增为了松组合添加的模块============================================
    // // Convert the msgs.
    // Eigen::Vector3d m_gyro_618Dpro, m_acc_618Dpro ;                            // 原始618Dpro机子里自带的消息
    // Eigen::Vector3d m_gyro_618Dpro_RFU, m_acc_618Dpro_RFU;                     // 转换为需要的三轴
    // Eigen::Vector3d m_gyro_618Dpro_RFU_deg, m_acc_618Dpro_RFU_mss ;            // 进行单位的转换
    // double m_time_618Dpro, m_time_618Dpro_s;                            // 原始的618Dpro记载的缺少了ROS时间戳，所以这里是拿GPS时间来处理
    // double dtime;                                                       // 记录时间差
    // Eigen::Vector3d gyro, acc;                                                 // 三轴加速度，角速度记录（去除bias的）

    // Eigen::Vector3d m_posi_GPS_618Dpro ;                                           // GPS提供的位置消息

    // // ------------kalman---------------
    // Eigen::MatrixXd F;                                                  // 状态传递矩阵
    // Eigen::MatrixXd Q;                                                  // 噪声阵
    // Eigen::MatrixXd Pestimated;                                         // 估计协方差
    // Eigen::VectorXd Xestimated;                                         // 估计误差状态量

    // Eigen::MatrixXd H;                                                  // 观测传递矩阵
    // Eigen::MatrixXd R;                                                  // 观测噪声阵
    // Eigen::Vector3d Z;                                                  // 观测误差向量


    // 2023-04-30 新增使用参考值实现对状态的初始化处理
    void INS_Inialization();

    // 2023-04-30 写一个消息转换的代码
    // ===============================参考值类========================
    void ForsenseMsgAttiToEigen(const forsense_msg::Forsense &m, Eigen::Vector3d &e)
    {
        // 这里主要是将自定义的数据转为Eigen中的向量形式(姿态参考值信息)
        e(0) = m.att[0];
        e(1) = m.att[1];
        e(2) = m.att[2];
    }

    void ForsenseMsgVeloToEigen(const forsense_msg::Forsense &m, Eigen::Vector3d &e)
    {
        // 这里主要是将自定义的数据转为Eigen中的向量形式（速度参考值信息）
        e(0) = m.vel[0];
        e(1) = m.vel[1];
        e(2) = m.vel[2];
    }

    void ForsenseMsgPosiToEigen(const forsense_msg::Forsense &m, Eigen::Vector3d &e)
    {
        // 这里主要是将自定义的数据转为Eigen中的向量形式（位置参考值信息）
        // 由于位置部分参考值对应的是 lati,long,alti ，即经度，纬度，高度，所以这里进行顺序调整
        e(0) = m.pos[1]/1E7;
        e(1) = m.pos[0]/1E7;
        e(2) = m.pos[2]/1E3;
    }

    // ===============================IMU类========================

    void ForsenseMsgGyroToEigen(const forsense_msg::Forsense &m, Eigen::Vector3d &e)
    {
        // 这里主要是将自定义的数据转为Eigen中的向量形式（IMU角速度值信息）
        // e(0) = m.gyro[0];
        // e(1) = m.gyro[1];
        // e(2) = m.gyro[2];
        e(0) = -m.gyro[1] * rad2deg;
        e(1) = m.gyro[0] * rad2deg;
        e(2) = m.gyro[2] * rad2deg;

        // m_gyro_618Dpro_RFU << -m_gyro_618Dpro[1], m_gyro_618Dpro[0], m_gyro_618Dpro[2];
        // m_gyro_618Dpro_RFU_deg = m_gyro_618Dpro_RFU * rad2deg;  // （deg/s）
    }

    void ForsenseMsgGyroRadToEigen(const forsense_msg::Forsense &m, Eigen::Vector3d &e)
    {
        // 这里主要是将自定义的数据转为Eigen中的向量形式（IMU角速度值信息）
        // e(0) = m.gyro[0];
        // e(1) = m.gyro[1];
        // e(2) = m.gyro[2];
        e(0) = -m.gyro[1];
        e(1) = m.gyro[0];
        e(2) = m.gyro[2];

        // m_gyro_618Dpro_RFU << -m_gyro_618Dpro[1], m_gyro_618Dpro[0], m_gyro_618Dpro[2];
        // m_gyro_618Dpro_RFU_deg = m_gyro_618Dpro_RFU * rad2deg;  // （deg/s）
    }


    void ForsenseMsgAccToEigen(const forsense_msg::Forsense &m, Eigen::Vector3d &e)
    {
        double g=9.7803698;         // 重力加速度
        // 这里主要是将自定义的数据转为Eigen中的向量形式（IMU加速度值信息）
        // e(0) = m.accel[0];
        // e(1) = m.accel[1];
        // e(2) = m.accel[2];
        e(0) = -m.accel[1] * g;
        e(1) = m.accel[0] * g;
        e(2) = m.accel[2] * g;


        // m_acc_618Dpro_RFU << -m_acc_618Dpro[1], m_acc_618Dpro[0], m_acc_618Dpro[2];
        // m_acc_618Dpro_RFU_mss = m_acc_618Dpro_RFU * g;  // （m/s/s）
    }

    void ForsenseMsgTimeToDouble(const forsense_msg::Forsense &m, double &e)
    {
        // 这里主要是将自定义的数据转为Eigen中的向量形式（IMU对应的时间信息，这里选择的是gps时间）
        // unsigned int time = gps_msg->time_gps_ms;
        // 实现了时间s的转换
        e = static_cast<double>(m.time_gps_ms)/1000.0;

    }

    // ===============================GPS类========================
    void ForsenseMsgGpsPosiToEigen(const forsense_msg::Forsense &m, Eigen::Vector3d &e)
    {
        // 这里主要是将自定义的数据转为Eigen中的向量形式（GPS提供的经纬高位置）
        e(0) = m.lon;
        e(1) = m.lat;
        e(2) = m.alt;
    }

    // 2023-05-01 准备松组合部分的处理模块

    // void kalmanFilterInitializaion(Eigen::MatrixXd Pestimated);               // kaman滤波器初始化处理
    void kalmanFilterInitializaion();               // kaman滤波器初始化处理

    // void INS_Update(const double& time,
    //                 const Eigen::Vector3d& m_gyro,
    //                 const Eigen::Vector3d& m_acc,
    //                 double dtime,
    //                 Eigen::Vector3d gyro,
    //                 Eigen::Vector3d acc);

        void INS_Update();

    // void INS_Attitude_Update();
    // void INS_Velocity_Update();
    // void INS_Position_Update();

    // void system_model_cal(double dtime,
    //                       Eigen::Vector3d gyro,
    //                       Eigen::Vector3d acc,
    //                       Eigen::MatrixXd F, 
    //                       Eigen::MatrixXd Q);            // 系统模型构建
        void system_model_cal();            // 系统模型构建

    // void KF_time_update(Eigen::VectorXd Xestimated,
    //                     Eigen::MatrixXd Pestimated,
    //                     Eigen::MatrixXd F, 
    //                     Eigen::MatrixXd Q);              // 运动状态迭代
        void KF_time_update();              // 运动状态迭代

    // void measur_model_cal(Eigen::Vector3d m_posi_GPS,
    //                       Eigen::MatrixXd H,
    //                       Eigen::MatrixXd R,
    //                       Eigen::Vector3d Z);           // GPS观测模型的构建
        void measur_model_cal(Eigen::Vector3d m_posi_GPS);           // GPS观测模型的构建

    // void KF_meas_update(Eigen::VectorXd Xestimated,
    //                     Eigen::MatrixXd Pestimated,
    //                     Eigen::MatrixXd H,
    //                     Eigen::MatrixXd R,
    //                     Eigen::Vector3d Z);             // 滤波器修正迭代
        void KF_meas_update();             // 滤波器修正迭代

    // void INS_Correction(Eigen::Vector3d PosNError,
    //                     Eigen::Vector3d VelNError,
    //                     Eigen::Vector3d AttNError);     // IMU状态修正

    // void INS_Correction(Eigen::VectorXd Xestimated);     // IMU状态修正
        void INS_Correction();     // IMU状态修正

    // void publish_loose();                              // 解算的轨迹等消息发布
    void publish_loose(forsense_msg::Forsense msg_copy2);                             // 解算的轨迹等消息发布


    // ===================================以下是为了新的消息类型新增函数段=================================
    // 2023-05-12 使用新的标准消息类型接口
    void GPSCallback(const sensor_msgs::NavSatFix::Ptr &gps_msg);

    void ReferenceCallback(const nav_msgs::Odometry::Ptr  &reference_msg);


    // 用于IMUcallback中IMU的角速度误差的修正处理
    void initializeGyroBias();

    // ===============================IMU类========================
    // 基于新的消息类型进行的数据传输定义
    void ForsenseMsgGyroRadNewToEigen(const sensor_msgs::Imu &m, Eigen::Vector3d &e)
    {
        // 这里主要是将自定义的数据转为Eigen中的向量形式（IMU角速度值信息）
        e(0) = -m.angular_velocity.y;       // pitch 向上为+（右向）rad/s
        e(1) = m.angular_velocity.x;        // roll 右旋为+(前向)  rad/s
        e(2) = m.angular_velocity.z;        // yaw 左相为+（天向）  rad/s

        // m_gyro_618Dpro_RFU << -m_gyro_618Dpro[1], m_gyro_618Dpro[0], m_gyro_618Dpro[2];
        // m_gyro_618Dpro_RFU_deg = m_gyro_618Dpro_RFU * rad2deg;  // （deg/s）
    }

    void ForsenseMsgGyroNewToEigen(const sensor_msgs::Imu &m, Eigen::Vector3d &e)
    {
        // 这里主要是将自定义的数据转为Eigen中的向量形式（IMU角速度值信息）
        // e(0) = m.gyro[0];
        // e(1) = m.gyro[1];
        // e(2) = m.gyro[2];
        e(0) = -m.angular_velocity.y * rad2deg;   // w_pitch  y轴 
        e(1) = m.angular_velocity.x * rad2deg;    // w_roll   x轴
        e(2) = m.angular_velocity.z * rad2deg;    // w_yaw    z轴

        // m_gyro_618Dpro_RFU << -m_gyro_618Dpro[1], m_gyro_618Dpro[0], m_gyro_618Dpro[2];
        // m_gyro_618Dpro_RFU_deg = m_gyro_618Dpro_RFU * rad2deg;  // （deg/s）
    }


    // ===============================参考值类========================
    void ForsenseMsgAttiNewToEigen(const nav_msgs::Odometry::Ptr &m, Eigen::Vector3d &e)
    {
        // 这里主要是将自定义的数据转为Eigen中的向量形式(姿态参考值信息)
        e(0) = m->twist.twist.angular.x;  // roll(deg)
        e(1) = m->twist.twist.angular.y;  // pitch(deg)
        e(2) = m->twist.twist.angular.z;  // yaw(deg)
    }

    void ForsenseMsgVeloNewToEigen(const nav_msgs::Odometry::Ptr &m, Eigen::Vector3d &e)
    {
        // 这里主要是将自定义的数据转为Eigen中的向量形式（速度参考值信息）
        // e(0) = m.twist.twist.linear.x;   // n(m/s/s)
        // e(1) = m.twist.twist.linear.y;   // e(m/s/s)
        // e(2) = m.twist.twist.linear.z;   // d(m/s/s)

        e(0) = m->twist.twist.linear.y;   // e(m/s/s) 
        e(1) = m->twist.twist.linear.x;   // n(m/s/s)
        e(2) = -m->twist.twist.linear.z;   // u(m/s/s)
    }

    void ForsenseMsgPosiNewToEigen(const nav_msgs::Odometry::Ptr &m, Eigen::Vector3d &e)
    {
        // 这里主要是将自定义的数据转为Eigen中的向量形式（位置参考值信息）
        // 由于位置部分参考值对应的是 lati,long,alti ，即经度，纬度，高度，所以这里进行顺序调整
        e(0) = m->pose.pose.position.y/1E7;    //long
        e(1) = m->pose.pose.position.x/1E7;    //lati
        e(2) = m->pose.pose.position.z/1E3;    //alti
    }



protected:
    // ==================================以下是新增为了松组合添加的模块============================================
    // Convert the msgs.
    Eigen::Vector3d m_gyro_618Dpro, m_acc_618Dpro ;                            // 原始618Dpro机子里自带的消息
    Eigen::Vector3d m_gyro_618Dpro_RFU, m_acc_618Dpro_RFU;                     // 转换为需要的三轴
    Eigen::Vector3d m_gyro_618Dpro_RFU_deg, m_acc_618Dpro_RFU_mss ;            // 进行单位的转换
    double m_time_618Dpro, m_time_618Dpro_s;                            // 原始的618Dpro记载的缺少了ROS时间戳，所以这里是拿GPS时间来处理
    double my_dtime;                                                       // 记录时间差
    Eigen::Vector3d my_gyro, my_acc;                                                 // 三轴加速度，角速度记录（去除bias的）

    Eigen::Vector3d m_posi_GPS_618Dpro ;                                           // GPS提供的位置消息

    // ------------kalman---------------
    Eigen::MatrixXd my_F;                                                  // 状态传递矩阵
    Eigen::MatrixXd my_Q;                                                  // 噪声阵
    Eigen::MatrixXd my_Pestimated;                                         // 估计协方差
    Eigen::VectorXd my_Xestimated;                                         // 估计误差状态量

    Eigen::MatrixXd my_H;                                                  // 观测传递矩阵
    Eigen::MatrixXd my_R;                                                  // 观测噪声阵
    Eigen::Vector3d my_Z;                                                  // 观测误差向量


};

typedef MsckfVio::Ptr MsckfVioPtr;
typedef MsckfVio::ConstPtr MsckfVioConstPtr;

} // namespace msckf_vio

#endif
