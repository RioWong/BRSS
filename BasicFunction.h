#pragma once
#ifndef BASICFUNCTION_H
#define BASICFUNCTION_H

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/pcl_base.h>
#include <pcl/io/pcd_io.h>
#include <pcl/features/boundary.h>
#include <pcl/features/normal_3d.h>
#include <pcl/search/kdtree.h>
#include <pcl/visualization/pcl_visualizer.h>

using namespace std;
using namespace pcl;

typedef PointXYZ PointT;
typedef PointCloud<PointT> PointCloudT;
typedef PointCloud<Normal> CloudNormal;

class BasicFunction
{
public:
	BasicFunction();
	~BasicFunction();
	//get density function
	float computeCloudResolution(const PointCloudT::ConstPtr cloud, int k);
	float computeLPD(const PointCloudT::ConstPtr cloud, int index, int k);//бшнд38
	float computeMDK(const PointCloudT::ConstPtr cloud, int index, int k);
	//get normal cloud
	void computeNormals(const PointCloudT::ConstPtr &cloud, float r, CloudNormal::Ptr &normalcloud);
	void showTwoCloud(const PointCloud<PointXYZL>::ConstPtr &cloud1, const PointCloud<PointXYZL>::ConstPtr &cloud2);
	void showOneCloud(const PointCloud<PointXYZL>::ConstPtr &cloud);
	void showNormalCloud(const PointCloud<PointT>::ConstPtr &cloud, const PointCloud<Normal>::ConstPtr &normalcloud);
	PointCloud<PointXYZ>::Ptr txt2pcd();
	void normalize(PointCloud<PointXYZL>::Ptr &cloud);
};

#endif BASICFUNCTION_H