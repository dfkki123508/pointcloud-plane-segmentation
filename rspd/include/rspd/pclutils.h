#ifndef PCLUTILS_H
#define PCLUTILS_H

#include <Eigen/Core>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

Eigen::Vector3f getMinPoint(const pcl::PointCloud<pcl::PointNormal>::ConstPtr &cloud)
{
    // Initialize minimum vector with the coordinates of the first point
    Eigen::Vector3f min_vec = cloud->points[0].getVector3fMap();

    for (const auto &point : cloud->points)
    {
        Eigen::Vector3f point_vec = point.getVector3fMap();
        min_vec = min_vec.cwiseMin(point_vec);
    }
    return min_vec;
}

Eigen::Vector3f getMaxPoint(const pcl::PointCloud<pcl::PointNormal>::ConstPtr &cloud)
{
    // Initialize minimum vector with the coordinates of the first point
    Eigen::Vector3f min_vec = cloud->points[0].getVector3fMap();

    for (const auto &point : cloud->points)
    {
        Eigen::Vector3f point_vec = point.getVector3fMap();
        min_vec = min_vec.cwiseMax(point_vec);
    }
    return min_vec;
}

#endif // PCLUTILS_H