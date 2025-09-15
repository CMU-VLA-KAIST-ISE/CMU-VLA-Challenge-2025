#include <math.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <vector>

#include <ros/ros.h>

#include <nav_msgs/Odometry.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>

#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>

#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

using namespace std;
using namespace cv;

// ========================= Params & Globals =========================
static const double PI = 3.14159265358979323846;

// Ring buffer for odom (simple and fast)
static const int kStackNum = 400;
float lidarXStack[kStackNum];
float lidarYStack[kStackNum];
float lidarZStack[kStackNum];
float lidarRollStack[kStackNum];
float lidarPitchStack[kStackNum];
float lidarYawStack[kStackNum];
double odomTimeStack[kStackNum];
int odomIDPointer = -1;

int imageIDPointer = 0; // kept for potential time-sync usage

bool imageInit = false;
double imageTime = 0.0;

bool newLaserCloud = false;
double laserCloudTime = 0.0;

// Clouds
pcl::PointCloud<pcl::PointXYZ>::Ptr laserCloudRaw(new pcl::PointCloud<pcl::PointXYZ>());
pcl::PointCloud<pcl::PointXYZRGB>::Ptr laserCloudColorAbs(new pcl::PointCloud<pcl::PointXYZRGB>());
pcl::PointCloud<pcl::PointXYZRGB>::Ptr laserCloudColorRel(new pcl::PointCloud<pcl::PointXYZRGB>());

// Publishers
ros::Publisher pubAbsPtr;
ros::Publisher pubRelPtr;

// Latest image
cv_bridge::CvImageConstPtr colorImageCv;

// Node params
string odom_topic;
string image_topic;
string input_scan_topic;
string abs_topic;
string rel_topic;
string parent_frame; // usually "map" or "odom"
string base_frame;   // usually "base_link"
string camera_frame; // usually "camera" or optical frame name

// camera offset (base_link->camera)
double cam_off_x = 0.0, cam_off_y = 0.0, cam_off_z = 0.0;
double cam_off_roll = 0.0, cam_off_pitch = 0.0, cam_off_yaw = 0.0; // radians

// legacy param kept for compatibility
double cameraOffsetZ = 0.0; // used in rel calc below (subtract along robot Z)

// ========================= Helpers =========================
static inline void rpyToQuat(double r, double p, double y, tf::Quaternion &q)
{
  tf::Matrix3x3 R;
  R.setRPY(r, p, y);
  R.getRotation(q);
}

// ========================= Callbacks =========================
void odomHandler(const nav_msgs::Odometry::ConstPtr &odom)
{
  double roll, pitch, yaw;
  const geometry_msgs::Quaternion &geoQuat = odom->pose.pose.orientation;
  tf::Matrix3x3(tf::Quaternion(geoQuat.x, geoQuat.y, geoQuat.z, geoQuat.w)).getRPY(roll, pitch, yaw);

  odomIDPointer = (odomIDPointer + 1) % kStackNum;
  odomTimeStack[odomIDPointer]   = odom->header.stamp.toSec();
  lidarXStack[odomIDPointer]     = odom->pose.pose.position.x;
  lidarYStack[odomIDPointer]     = odom->pose.pose.position.y;
  lidarZStack[odomIDPointer]     = odom->pose.pose.position.z;
  lidarRollStack[odomIDPointer]  = roll;
  lidarPitchStack[odomIDPointer] = pitch;
  lidarYawStack[odomIDPointer]   = yaw;

  // === Broadcast TF: parent_frame -> base_frame ===
  static tf::TransformBroadcaster br;
  tf::Transform T;
  T.setOrigin(tf::Vector3(lidarXStack[odomIDPointer], lidarYStack[odomIDPointer], lidarZStack[odomIDPointer]));
  tf::Quaternion q;
  rpyToQuat(roll, pitch, yaw, q);
  T.setRotation(q);

  // prefer header frame if provided, else configured parent_frame
  const std::string parent = odom->header.frame_id.empty() ? parent_frame : odom->header.frame_id;
  br.sendTransform(tf::StampedTransform(T, odom->header.stamp, parent, base_frame));

  // === Broadcast TF: base_frame -> camera_frame (static-like) ===
  tf::Transform Tbc;
  Tbc.setOrigin(tf::Vector3(cam_off_x, cam_off_y, cam_off_z));
  tf::Quaternion qbc; rpyToQuat(cam_off_roll, cam_off_pitch, cam_off_yaw, qbc);
  Tbc.setRotation(qbc);
  br.sendTransform(tf::StampedTransform(Tbc, odom->header.stamp, base_frame, camera_frame));
}

void colorImageHandler(const sensor_msgs::ImageConstPtr &image)
{
  imageTime = image->header.stamp.toSec();
  colorImageCv = cv_bridge::toCvShare(image, "bgr8");
  imageInit = true;
  ROS_INFO_ONCE("[color_scan_generation] first image received");
}

void laserCloudHandler(const sensor_msgs::PointCloud2ConstPtr &laserCloudIn)
{
  laserCloudTime = laserCloudIn->header.stamp.toSec();
  laserCloudRaw->clear();
  pcl::fromROSMsg(*laserCloudIn, *laserCloudRaw);
  newLaserCloud = true;
  //ROS_INFO_THROTTLE(2.0, "DBG /registered_scan received: size=%zu", laserCloudRaw->points.size());
}

// ========================= Main =========================
int main(int argc, char **argv)
{
  ros::init(argc, argv, "colorScanGeneration");
  ros::NodeHandle nh;
  ros::NodeHandle pnh("~");

  // ---- Load params ----
  pnh.param<string>("odom_topic",         odom_topic,         string("/state_estimation"));
  pnh.param<string>("image_topic",        image_topic,        string("/camera/image"));
  pnh.param<string>("input_scan_topic",   input_scan_topic,   string("/registered_scan"));
  pnh.param<string>("abs_topic",          abs_topic,          string("/color_scan"));
  pnh.param<string>("rel_topic",          rel_topic,          string("/color_scan_relative"));
  pnh.param<string>("parent_frame",       parent_frame,       string("map"));
  pnh.param<string>("base_frame",         base_frame,         string("base_link"));
  pnh.param<string>("camera_frame",       camera_frame,       string("camera"));

  pnh.param("cameraOffsetZ", cameraOffsetZ, 0.0); // legacy support

  pnh.param("cam_off_x", cam_off_x, 0.0);
  pnh.param("cam_off_y", cam_off_y, 0.0);
  pnh.param("cam_off_z", cam_off_z, 0.0);
  pnh.param("cam_off_roll",  cam_off_roll,  0.0);
  pnh.param("cam_off_pitch", cam_off_pitch, 0.0);
  pnh.param("cam_off_yaw",   cam_off_yaw,   0.0);

  ROS_INFO("[color_scan_generation] parent_frame=%s base_frame=%s camera_frame=%s", parent_frame.c_str(), base_frame.c_str(), camera_frame.c_str());
  ROS_INFO("[color_scan_generation] topics: odom=%s image=%s in_scan=%s abs_out=%s rel_out=%s",
           odom_topic.c_str(), image_topic.c_str(), input_scan_topic.c_str(), abs_topic.c_str(), rel_topic.c_str());

  // ---- Subs & Pubs ----
  ros::Subscriber subOdom  = nh.subscribe<nav_msgs::Odometry>(odom_topic, 50, odomHandler);
  ros::Subscriber subImage = nh.subscribe<sensor_msgs::Image>(image_topic, 2, colorImageHandler);
  ros::Subscriber subScan  = nh.subscribe<sensor_msgs::PointCloud2>(input_scan_topic, 2, laserCloudHandler);

  pubAbsPtr = nh.advertise<sensor_msgs::PointCloud2>(abs_topic, 2);
  pubRelPtr = nh.advertise<sensor_msgs::PointCloud2>(rel_topic, 2);

  ros::Rate rate(200);

  while (ros::ok())
  {
    ros::spinOnce();

    //ROS_INFO_THROTTLE(1.0, "DBG imageInit=%d newLaserCloud=%d odomID=%d", imageInit, newLaserCloud, odomIDPointer);

    if (!imageInit || !newLaserCloud) {
      rate.sleep();
      continue;
    }

    newLaserCloud = false; // consume

    const int laserCloudSize = static_cast<int>(laserCloudRaw->points.size());
    if (laserCloudSize <= 0) {
      rate.sleep();
      continue;
    }

    if (odomIDPointer < 0) {
      ROS_WARN_THROTTLE(1.0, "No odom yet; skip cloud coloring");
      rate.sleep();
      continue;
    }

    // Take the most recent odom (simple, robust). If you want strict sync, add logic.
    const int idx = odomIDPointer;
    const float lidarX    = lidarXStack[idx];
    const float lidarY    = lidarYStack[idx];
    const float lidarZ    = lidarZStack[idx];
    const float lidarRoll = lidarRollStack[idx];
    const float lidarPitch= lidarPitchStack[idx];
    const float lidarYaw  = lidarYawStack[idx];

    const int imageWidth  = colorImageCv->image.cols;
    const int imageHeight = colorImageCv->image.rows;

    const float sinR = sinf(lidarRoll),  cosR = cosf(lidarRoll);
    const float sinP = sinf(lidarPitch), cosP = cosf(lidarPitch);
    const float sinY = sinf(lidarYaw),   cosY = cosf(lidarYaw);

    laserCloudColorAbs->clear();
    laserCloudColorRel->clear();
    laserCloudColorAbs->reserve(laserCloudSize);
    laserCloudColorRel->reserve(laserCloudSize);

    for (int i = 0; i < laserCloudSize; ++i) {
      const float X = laserCloudRaw->points[i].x;
      const float Y = laserCloudRaw->points[i].y;
      const float Z = laserCloudRaw->points[i].z;

      // ===== Absolute (map): just forward absolute coordinates, color from image =====
      pcl::PointXYZRGB p_abs;
      p_abs.x = X; p_abs.y = Y; p_abs.z = Z;

      // ===== Relative (base_link): transform (map -> base_link) inverse applied =====
      // map point -> base_link coordinates
      const float x1 = X - lidarX;
      const float y1 = Y - lidarY;
      const float z1 = Z - lidarZ;

      // yaw
      const float x2 =  x1 * cosY + y1 * sinY;
      const float y2 = -x1 * sinY + y1 * cosY;
      const float z2 =  z1;

      // pitch
      const float x3 =  x2 * cosP - z2 * sinP;
      const float y3 =  y2;
      const float z3 =  x2 * sinP + z2 * cosP;

      // roll
      const float x4 =  x3;
      const float y4 =  y3 * cosR + z3 * sinR;
      const float z4 = -y3 * sinR + z3 * cosR - static_cast<float>(cameraOffsetZ);

      pcl::PointXYZRGB p_rel;
      p_rel.x = x4; p_rel.y = y4; p_rel.z = z4;

      // ==== Color from image (simple spherical projection heuristic) ====
      const float horiDis = sqrtf(x4 * x4 + y4 * y4);
      const int   u = -static_cast<int>( static_cast<float>(imageWidth) / (2.0f * PI) * atan2f(y4, x4) ) + imageWidth / 2 + 1;
      const int   v = -static_cast<int>( static_cast<float>(imageWidth) / (2.0f * PI) * atanf(z4 / (horiDis + 1e-6f)) ) + imageHeight / 2 + 1;

      if (u >= 0 && u < imageWidth && v >= 0 && v < imageHeight) {
        const int pixelID = imageWidth * v + u;
        const uchar b = colorImageCv->image.data[3 * pixelID + 0];
        const uchar g = colorImageCv->image.data[3 * pixelID + 1];
        const uchar r = colorImageCv->image.data[3 * pixelID + 2];
        p_abs.b = b; p_abs.g = g; p_abs.r = r;
        p_rel.b = b; p_rel.g = g; p_rel.r = r;
      } else {
        p_abs.r = p_abs.g = p_abs.b = 255;
        p_rel.r = p_rel.g = p_rel.b = 255;
      }

      laserCloudColorAbs->push_back(p_abs);
      laserCloudColorRel->push_back(p_rel);
    }

    // ===== Publish absolute cloud (parent_frame) =====
    sensor_msgs::PointCloud2 cloudAbsMsg;
    pcl::toROSMsg(*laserCloudColorAbs, cloudAbsMsg);
    cloudAbsMsg.header.stamp = ros::Time().fromSec(laserCloudTime);
    cloudAbsMsg.header.frame_id = parent_frame; // "map" by default
    pubAbsPtr.publish(cloudAbsMsg);

    // ===== Publish relative cloud (base_frame) =====
    sensor_msgs::PointCloud2 cloudRelMsg;
    pcl::toROSMsg(*laserCloudColorRel, cloudRelMsg);
    cloudRelMsg.header.stamp = ros::Time().fromSec(laserCloudTime);
    cloudRelMsg.header.frame_id = base_frame;   // "base_link"
    pubRelPtr.publish(cloudRelMsg);

    //ROS_INFO_THROTTLE(1.0, "Published abs=%zu rel=%zu (frames: %s, %s)",
    //                  laserCloudColorAbs->points.size(),
    //                  laserCloudColorRel->points.size(),
    //                  parent_frame.c_str(), base_frame.c_str());

    rate.sleep();
  }

  return 0;
}
