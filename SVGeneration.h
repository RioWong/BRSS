#pragma once
#ifndef SVGENERATION_H
#define SVGENERATION_H
#include <pcl/segmentation/supervoxel_clustering.h>
#include <pcl/segmentation/lccp_segmentation.h>

using namespace pcl;
using namespace std;

typedef PointXYZ PointT;
typedef PointCloud<PointT> PointCloudT;
typedef multimap<uint32_t, uint32_t>::iterator int_multimap;
typedef map<uint32_t, Supervoxel<PointT>::Ptr > supervoxelmap;

const double eps = 1.0e-6;

class SVGeneration
{
public:
	SVGeneration(PointCloudT::Ptr &cloud, PointCloud<Normal>::Ptr &cloud_normal):
		cloud_(cloud), normalcloud_(cloud_normal)
	{		
	}
	~SVGeneration()
	{
	}
	//存储边界点信息
	class BoundaryData
	{
	public:
		BoundaryData() : xyz_(0.0f, 0.0f, 0.0f), normal_(0.0f, 0.0f, 0.0f)
		{
		}
		Eigen::Vector3f xyz_;
		Eigen::Vector3f normal_;
		float curvature_;
		//float distance_;
		int idx_;
	};
	void getVCCS(float Rvoxel, float Rseed, PointCloud<PointXYZL>::Ptr &sv_labeled_cloud1, PointCloud<PointXYZL>::Ptr &sv_labeled_cloud2);
	void getSVBoundary(map<uint32_t, vector<BoundaryData>> &boundaries);
	void expandBSV();
	void updateCentroid();
	float distCal(BoundaryData p1, Supervoxel<PointT>::Ptr p2);
	void removeOutliers(PointCloudT::Ptr cloud1, PointCloudT::Ptr cloud2, PointCloud<Normal>::Ptr normal);
	supervoxelmap get_sv_clusters()
	{
		return sv_clusters_;
	}
	multimap<uint32_t, uint32_t> get_nei_labels()
	{
		return nei_labels_;
	}
	PointCloud<PointXYZL>::Ptr getLabelCloud();//show the sv result
	float calNCE(supervoxelmap sv);
	float getVCCS_NCE(){
		return vccs_nce;
	}
	float getBRSS_NCE(){
		return brss_nce;
	}
private:
	PointCloudT::Ptr cloud_;
	PointCloud<Normal>::Ptr normalcloud_;
	float Rvoxel_, Rseed_;
	supervoxelmap sv_clusters_;//VCCS的结果
	float vccs_nce, brss_nce;
	multimap<uint32_t, uint32_t> nei_labels_;//VCCS的超体素邻接关系
	map<uint32_t, vector<BoundaryData>> sv_boundaries_;//超体素标签->边界点实体
	
};

#endif SVGENERATION_H