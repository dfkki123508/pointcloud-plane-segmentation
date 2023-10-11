#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#endif

#include <chrono>
#include <iostream>
#include <limits>
#include <memory>
#include <set>
#include <vector>

#include <Eigen/Dense>

#include <rspd/planedetector.h>
#include <rspd/geometryutils.h>

#include <pcl/io/ply_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/features/normal_3d_omp.h>

template <typename PointT = pcl::PointNormal>
typename pcl::PointCloud<PointT>::Ptr LoadPointCloud(const std::string &pcdfilepath)
{
    auto points = std::make_shared<typename pcl::PointCloud<PointT>>();
    if (pcl::io::loadPLYFile<PointT>(pcdfilepath, *points) == -1)
    {
        throw std::runtime_error("Reading PCD file from failed!");
    }
    return points;
}

void CalcNormals(pcl::PointCloud<pcl::PointNormal>::Ptr inputCloud, double radius)
{
    pcl::NormalEstimationOMP<pcl::PointNormal, pcl::PointNormal> ne;
    ne.setInputCloud(inputCloud);
    ne.setRadiusSearch(radius);
    ne.compute(*inputCloud);
}

// ----------------------------------------------------------------------------

int main(int argc, char *argv[])
{

    // utility::SetVerbosityLevel(utility::VerbosityLevel::Debug);
    if (argc < 2)
    {
        std::cout << "Given args (" << argc << "):" << std::endl;
        for (int i = 0; i < argc; ++i)
        {
            std::cout << argv[i] << " ";
        }
        std::cout << std::endl;

        return 1;
    }

    // Defaults produce the best results currently
    double minNormalDiff = 25.0;
    double maxDist = 30.0;
    double outlierRatio = 0.1;
    if (argc > 2)
    {
        minNormalDiff = std::stod(argv[2]);
        if (argc > 3)
        {
            maxDist = std::stod(argv[3]);
            if (argc > 4)
                outlierRatio = std::stod(argv[4]);
        }
    }

    std::cout << "minNormalDiff: " << minNormalDiff << std::endl;
    std::cout << "maxDist: " << maxDist << std::endl;
    std::cout << "outlierRatio: " << outlierRatio << std::endl;

    static constexpr int nrNeighbors = 75;

    auto cloud_ptr = LoadPointCloud(argv[1]);

    std::cout << "Loaded point cloud with " << cloud_ptr->size() << " points." << std::endl;

    auto t1 = std::chrono::high_resolution_clock::now();
    CalcNormals(cloud_ptr, 0.25);
    const double t_normals = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::high_resolution_clock::now() - t1).count();
    std::cout << "o3d EstimateNormals: " << t_normals << " seconds" << std::endl;

    std::vector<std::vector<int>> neighbors;
    neighbors.resize(cloud_ptr->size());

    // KD tree with pcl
    t1 = std::chrono::high_resolution_clock::now();

    std::cout << "Constructing kdtree..." << std::endl;
    pcl::KdTreeFLANN<pcl::PointNormal> kdtree;
    kdtree.setInputCloud(cloud_ptr);

    std::cout << "Finding neighbors..." << std::endl;
#pragma omp parallel for schedule(static)
    for (int i = 0; i < (int)cloud_ptr->size(); i++)
    {
        const auto &searchPoint = cloud_ptr->points[i];
        std::vector<int> indices;
        std::vector<float> distance2;
        if (kdtree.nearestKSearch(searchPoint, nrNeighbors, indices, distance2))
        {
            neighbors[i] = indices;
        }
    }

    const double t_kdtree = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::high_resolution_clock::now() - t1).count();
    std::cout << "kdtree search: " << t_kdtree << " seconds" << std::endl;

    t1 = std::chrono::high_resolution_clock::now();
    PlaneDetector rspd(cloud_ptr, neighbors);
    rspd.minNormalDiff(std::cos(GeometryUtils::deg2rad(minNormalDiff)));
    rspd.maxDist(std::cos(GeometryUtils::deg2rad(maxDist)));
    rspd.outlierRatio(outlierRatio);

    std::set<Plane *> planes = rspd.detect();
    const double t_rspd = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::high_resolution_clock::now() - t1).count();
    std::cout << "rspd detect: " << t_rspd << " seconds" << std::endl;
    std::cout << std::endl;

    std::cout << "Detected the following " << planes.size() << " planes:" << std::endl;
    for (const auto &p : planes)
    {
        std::cout << p->normal().transpose() << " ";
        std::cout << p->distanceFromOrigin() << "\t";
        std::cout << p->center().transpose() << "\t";
        std::cout << p->basisU().transpose() << "\t";
        std::cout << p->basisV().transpose() << std::endl;
    }
    std::cout << "==============================" << std::endl;

    std::cout << std::endl;
    std::cout << "EstimateNormals: " << t_normals << " seconds" << std::endl;
    std::cout << "DetectPlanarPatches: " << planes.size() << " in " << (t_rspd + t_kdtree) << " seconds" << std::endl;

    //     //
    //     // Visualization
    //     //

    //     // create a vector of geometries to visualize, starting with input point cloud
    //     std::vector<std::shared_ptr<const geometry::Geometry>> geometries;
    //     geometries.reserve(planes.size());
    //     // geometries.push_back(cloud_ptr);

    //     // Colors (default MATLAB colors)
    //     std::vector<Eigen::Vector3d> colors;
    //     colors.push_back(Eigen::Vector3d(0.8500, 0.3250, 0.0980));
    //     colors.push_back(Eigen::Vector3d(0.9290, 0.6940, 0.1250));
    //     colors.push_back(Eigen::Vector3d(0.4940, 0.1840, 0.5560));
    //     colors.push_back(Eigen::Vector3d(0.4660, 0.6740, 0.1880));
    //     colors.push_back(Eigen::Vector3d(0.3010, 0.7450, 0.9330));
    //     colors.push_back(Eigen::Vector3d(0.6350, 0.0780, 0.1840));

    //     // add any planes
    //     size_t i = 0;
    //     for (const auto& p : planes) {
    //         auto pviz = makePlane(p->center(), p->normal(), p->basisU(), p->basisV());
    //         pviz->PaintUniformColor(colors[i%6]);
    //         geometries.push_back(pviz);

    //         ++i;
    //     }

    //     visualization::DrawGeometries(geometries, "Points and Planes", 1600, 900);

    // visualization::VisualizerWithVertexSelection visualizer;
    // visualizer.CreateVisualizerWindow("Plane Selection: Select Point Close to Desired Plane", 1600, 900);
    // visualizer.AddGeometry(cloud_ptr);
    // visualizer.Run();
    // visualizer.DestroyVisualizerWindow();
    // const auto pts = visualizer.GetPickedPoints();

    // for (const auto& pt : pts) {
    //     double d = std::numeric_limits<double>::max();
    //     Plane* closest_plane;
    //     for (const auto& p : planes) {
    //         if (std::abs(p->getSignedDistanceFromSurface(pt.coord.cast<float>())) < d) {
    //             d = std::abs(p->getSignedDistanceFromSurface(pt.coord.cast<float>()));
    //             closest_plane = p;
    //         }
    //     }

    //     if (closest_plane == nullptr) {
    //         std::cout << "Could not find closest plane to selected point!" << std::endl;
    //     } else {
    //         std::cout << closest_plane->normal().transpose() << " ";
    //         std::cout << closest_plane->distanceFromOrigin() << "\t";
    //         std::cout << closest_plane->center().transpose() << "\t";
    //         std::cout << closest_plane->basisU().transpose() << "\t";
    //         std::cout << closest_plane->basisV().transpose() << std::endl;
    //     }
    // }

    // utility::LogInfo("End of the test.\n");

    return 0;
}
