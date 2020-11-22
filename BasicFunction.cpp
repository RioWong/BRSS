#include "BasicFunction.h"

BasicFunction::BasicFunction()
{
}

BasicFunction::~BasicFunction()
{
}
//以K邻域最大距离计算每个点的密度,点数k/最大半径圆面积
float BasicFunction::computeLPD(const PointCloudT::ConstPtr cloud, int index, int k)
{
	float res = 0.0;
	vector<int> indices(k);
	vector<float> sqr_distances(k);
	search::KdTree<PointT> tree;
	tree.setInputCloud(cloud);
	if (tree.nearestKSearch(index, k, indices, sqr_distances) == k)
	{
		float dk = sqr_distances[k - 1];
		res = float(k) / static_cast<float> (M_PI)*dk*dk;
	}

	return res;
}
//k个邻近点距离和/k
float BasicFunction::computeMDK(const PointCloudT::ConstPtr cloud, int index, int k)
{
	float res = 0.0;
	vector<int> indices(k);
	vector<float> sqr_distances(k);
	search::KdTree<PointT> tree;
	tree.setInputCloud(cloud);
	if (tree.nearestKSearch(index, k, indices, sqr_distances) == k)
	{
		for (int i = 0; i < k; i++)
		{
			res += sqr_distances[i];
		}
	}

	return res / k;
}
//计算点云的平均距离，单点所占面积
float BasicFunction::computeCloudResolution(const PointCloudT::ConstPtr cloud, int k)
{
	float res = 0.0;
	int n_points = 0;
	int nres;
	vector<int> indices(k);
	vector<float> sqr_distances(k);
	search::KdTree<PointT> tree;
	tree.setInputCloud(cloud);
	for (size_t i = 0; i < cloud->size(); ++i)
	{
		if (!isfinite((*cloud)[i].x))
		{
			continue;
		}
		nres = tree.nearestKSearch(i, k, indices, sqr_distances);
		if (nres == k)
		{
			for (int i = 1; i < k; i++)
			{
				res += sqrt(sqr_distances[i]);
				++n_points;
			}
		}
	}
	if (n_points != 0)
	{
		res /= n_points;
	}
	return res;
}
//计算点云法向量
void BasicFunction::computeNormals(const PointCloudT::ConstPtr &cloud, float r, CloudNormal::Ptr &normalcloud)
{
	pcl::NormalEstimation<PointT, pcl::Normal> normalEst;
	CloudNormal::Ptr normals(new CloudNormal());
	normalEst.setInputCloud(cloud);
	//normalEst.setKSearch(100);
	normalEst.setRadiusSearch(r);
	normalEst.compute(*normals);
	normalcloud = normals;
}
//可视化点云
void BasicFunction::showOneCloud(const PointCloud<PointXYZL>::ConstPtr &cloud)
{
	visualization::PCLVisualizer::Ptr viewer(new visualization::PCLVisualizer("cloud viewer"));
	viewer->setBackgroundColor(1.0, 1.0, 1.0);
	//visualization::PointCloudColorHandlerCustom<PointXYZ> single_color(cloud, 150, 150, 150); // gray
	viewer->addPointCloud(cloud, "cloud");
	viewer->setPointCloudRenderingProperties(visualization::PCL_VISUALIZER_POINT_SIZE, 1.0, "cloud");
	while (!viewer->wasStopped())
	{
		viewer->spinOnce();
	}
}
void BasicFunction::showTwoCloud(const PointCloud<PointXYZL>::ConstPtr &cloud1, const PointCloud<PointXYZL>::ConstPtr &cloud2)
{
	visualization::PCLVisualizer::Ptr viewer(new visualization::PCLVisualizer("cloud viewer"));
	int v1(0), v2(0);
	viewer->createViewPort(0.0, 0.0, 0.5, 1.0, v1);
	viewer->setBackgroundColor(1.0, 1.0, 1.0, v1);
	viewer->addText("viewer1", 10, 10, "v1 text", v1);
	viewer->addPointCloud(cloud1, "cloud1", v1);
	viewer->createViewPort(0.5, 0.0, 1.0, 1.0, v2);
	viewer->setBackgroundColor(1.0, 1.0, 1.0, v2);
	viewer->addText("viewer2", 10, 10, "v2 text", v2);
	viewer->addPointCloud(cloud2, "cloud2", v2);
	viewer->setPointCloudRenderingProperties(visualization::PCL_VISUALIZER_POINT_SIZE, 2.0, "cloud1");
	viewer->setPointCloudRenderingProperties(visualization::PCL_VISUALIZER_POINT_SIZE, 2.0, "cloud2");
	while (!viewer->wasStopped())
	{
		viewer->spinOnce();
	}
}
//标准化坐标值较大的数据，便于显示
void BasicFunction::normalize(PointCloud<PointXYZL>::Ptr &cloud)
{
	int count = cloud->size();
	float x_avg = 0.0, y_avg = 0.0, z_avg = 0.0;
	for (int i = 0; i < count; i++){
		x_avg += cloud->at(i).x;
		y_avg += cloud->at(i).y;
		z_avg += cloud->at(i).z;
	}
	for (int i = 0; i < count; i++){
		cloud->at(i).x -= x_avg / count;
		cloud->at(i).y -= y_avg / count;
		cloud->at(i).z -= z_avg / count;
	}
}
void BasicFunction::showNormalCloud(const PointCloud<PointT>::ConstPtr &cloud,const PointCloud<Normal>::ConstPtr &normalcloud)
{
	visualization::PCLVisualizer::Ptr viewer(new visualization::PCLVisualizer("cloud viewer"));
	//viewer->setBackgroundColor(1.0, 1.0, 1.0);
	viewer->addPointCloud(cloud, "cloud");
	viewer->addPointCloudNormals<PointT, Normal>(cloud, normalcloud, 5, 0.01, "cloud_normal");
	viewer->setPointCloudRenderingProperties(visualization::PCL_VISUALIZER_POINT_SIZE, 1.0, "cloud");
	while (!viewer->wasStopped())
	{
		viewer->spinOnce();
	}
}
//txt文件转为pcd文件
PointCloud<PointXYZ>::Ptr BasicFunction::txt2pcd(){
	FILE *fp_txt;
	PointXYZ point;
	PointCloud<PointXYZ>::Ptr cloud (new PointCloud<PointXYZ>());
	fp_txt = fopen("pcdTest/10.txt", "r");
	float x, y, z;
	if (fp_txt){
		while (3 == fscanf(fp_txt,"%f %f %f \n",&x,&y,&z))
		{
			//cout << x << y << z << endl;
			point.x = x;
			point.y = y;
			point.z = z;
			cloud->push_back(point);
		}
	}
	else{
		cout << "文件加载失败" << endl;
	}
	pcl::PCDWriter writer;
	writer.writeASCII<PointXYZ>("pcdTest/10.pcd", *cloud);
	return cloud;
}