#include "BasicFunction.h"
#include "SVGeneration.h"
#include "RegionGrowing.h"
#include <pcl/io/pcd_io.h>
#include <pcl/console/time.h>
#include <pcl/filters/filter.h>
#include <pcl/visualization/pcl_visualizer.h>

using namespace std;
using namespace pcl;

int main()
{
	//*******************Define and Init****************//
	PointCloud<PointXYZ>::Ptr cloud_input(new PointCloud<PointXYZ>());
	BasicFunction bas;
	CloudNormal::Ptr normalcloud(new CloudNormal());
	map<uint32_t, vector<SVGeneration::BoundaryData>> cloud_boundary;
	float Rvoxel = 0.05f;//0.005f
	float Rseed = 0.6f;//0.06f
	bas.txt2pcd();
	io::loadPCDFile("pcdFile/testCloud.pcd", *cloud_input);
	cout << cloud_input->size() << endl;
	//float res = bas.computeCloudResolution(cloud_input, 2);
	//cout << res << endl;
	//*******************Preprocess****************//
	vector<int> mapping;
	removeNaNFromPointCloud(*cloud_input, *cloud_input, mapping);
	console::TicToc tt;
	bas.computeNormals(cloud_input, Rvoxel, normalcloud);
	//****************Supervoxel generation******************//
	SVGeneration brsv(cloud_input, normalcloud);
	PointCloud<PointXYZL>::Ptr sv_labeled_cloud1(new PointCloud<PointXYZL>);//存储超体素生成结果
	PointCloud<PointXYZL>::Ptr sv_labeled_cloud2(new PointCloud<PointXYZL>);
	tt.tic();
	brsv.getVCCS(Rvoxel, Rseed, sv_labeled_cloud1, sv_labeled_cloud2);
	io::savePCDFileASCII<PointXYZL>("testCloud_out.pcd", *sv_labeled_cloud2);
	supervoxelmap sv_clusters = brsv.get_sv_clusters();
	multimap<uint32_t, uint32_t> nei_labels = brsv.get_nei_labels();
	bas.normalize(sv_labeled_cloud1);
	bas.normalize(sv_labeled_cloud2);
	bas.showTwoCloud(sv_labeled_cloud1, sv_labeled_cloud2);
	//*******************Region growing****************//
	RegionGrowing RG(sv_clusters, nei_labels);
	RG.getRegions();
	bas.showOneCloud(RG.getPatchCloud());
	//*******************Convexity mergering****************//
	RG.mergingConvex();
	RG.mergeSmallRegions();
	cout << "time:" << tt.toc() << "ms" << endl;
	bas.showOneCloud(RG.getPatchCloud());
	//*******************Accuracy assessments****************//
	cout << "SV size:" << sv_clusters.size() << endl;
	cout << "VCCS compactness:" << brsv.getVCCS_NCE() << endl;
	cout << "BRSS compactness:" << brsv.getBRSS_NCE() << endl;
	return 0;
}
