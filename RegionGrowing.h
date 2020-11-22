#pragma once
#ifndef REGIONGROWING_H
#define REGIONGROWING_H
#include "SVGeneration.h"


class RegionGrowing
{
public:
	RegionGrowing(supervoxelmap sv_clusters, multimap<uint32_t, uint32_t> nei_labels)
		: sv_clusters_(sv_clusters), nei_labels_(nei_labels)
	{}
	~RegionGrowing()
	{}
	/****区域增长***/
	void getRegions();
	int svGrowing(int seed, int segment_number);
	bool isValidated(int seed, int neighbor, bool &is_a_seed);
	PointCloud<PointXYZL>::Ptr getPatchCloud();
	/***基于凸性合并***/
	void mergingConvex();
	bool crossVal(int label_i, int label_j);//交叉验证
	//判断凹凸性
	bool isConvex1(int s_label, int t_label);
	bool isConvex2(int s_label, int t_label);
	//refinement
	void mergeSmallRegions();
	//静态方法，用于map按value排序
	static bool cmp(const pair<int, vector<int> > &r1, const pair<int, vector<int> > &r2){
		return r1.second.size() > r2.second.size();
	}

private:
	supervoxelmap sv_clusters_;//边界细化超体素集，标签->实体
	multimap<uint32_t, uint32_t> nei_labels_;//超体素邻接关系，标签：一对多

	map<int, int> sv_labels_;//超体素标签->区域标签
	map<int, float> label_wis_;//用于存储每个超体素的wi

	map<int, vector<Supervoxel<PointT>::Ptr>> regions_;//区域标签->内部超体素实体，用于存放结果，便于显示
	map<int, vector<int> > seglabel_to_svlist_;//区域标签->内部超体素标签
	map<int, int> region_object_;

	map<int, set<int>> region_neis_;//区域标签->邻接区域标签集
};

#endif REGIONGROWING_H