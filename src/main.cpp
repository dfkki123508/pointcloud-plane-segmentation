#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#endif

#include <vtkActor.h>
#include <vtkCellArray.h>
#include <vtkNamedColors.h>
#include <vtkNew.h>
#include <vtkPoints.h>
#include <vtkPolyData.h>
#include <vtkPolyDataMapper.h>
#include <vtkProperty.h>
#include <vtkQuad.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkRenderer.h>

#include <Eigen/Dense>

#include <pcl/io/ply_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/visualization/pcl_visualizer.h>

#include <chrono>
#include <iostream>
#include <limits>
#include <memory>
#include <set>
#include <thread>

#include <rspd/planedetector.h>
#include <rspd/geometryutils.h>

using namespace std::chrono_literals;

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

void CalcNormals(pcl::PointCloud<pcl::PointNormal>::Ptr inputCloud, double radius, int nrNeighbors)
{
    pcl::NormalEstimationOMP<pcl::PointNormal, pcl::PointNormal> ne;
    ne.setInputCloud(inputCloud);
    // ne.setRadiusSearch(radius);
    ne.setKSearch(nrNeighbors);
    ne.compute(*inputCloud);
}

void AddPlane(const Eigen::Vector3d &center, const Eigen::Vector3d &basisU, const Eigen::Vector3d &basisV, const std::array<double, 3> &color, vtkSmartPointer<vtkRenderer> renderer)
{
    // Create four points (must be in counter clockwise order)
    Eigen::Vector3d p0 = center - basisU - basisV;
    Eigen::Vector3d p1 = center - basisU + basisV;
    Eigen::Vector3d p2 = center + basisU + basisV;
    Eigen::Vector3d p3 = center + basisU - basisV;

    // Add the points to a vtkPoints object
    vtkNew<vtkPoints> points;
    points->InsertNextPoint(p0.array().data());
    points->InsertNextPoint(p1.array().data());
    points->InsertNextPoint(p2.array().data());
    points->InsertNextPoint(p3.array().data());

    // Create a quad on the four points
    vtkNew<vtkQuad> quad;
    quad->GetPointIds()->SetId(0, 0);
    quad->GetPointIds()->SetId(1, 1);
    quad->GetPointIds()->SetId(2, 2);
    quad->GetPointIds()->SetId(3, 3);

    // Create a cell array to store the quad in
    vtkNew<vtkCellArray> quads;
    quads->InsertNextCell(quad);

    // Create a polydata to store everything in
    vtkNew<vtkPolyData> polydata;

    // Add the points and quads to the dataset
    polydata->SetPoints(points);
    polydata->SetPolys(quads);

    // Setup actor and mapper
    vtkNew<vtkPolyDataMapper> mapper;
    mapper->SetInputData(polydata);

    vtkNew<vtkActor> actor;
    actor->SetMapper(mapper);
    actor->GetProperty()->SetColor(color[0], color[1], color[2]);

    renderer->AddActor(actor);
}

static constexpr std::array<std::array<double, 3>, 6> GetDefaultColorPalatte()
{
    // Colors (default MATLAB colors)
    std::array<std::array<double, 3>, 6> colors = {{
        {0.8500, 0.3250, 0.0980},
        {0.9290, 0.6940, 0.1250},
        {0.4940, 0.1840, 0.5560},
        {0.4660, 0.6740, 0.1880},
        {0.3010, 0.7450, 0.9330},
        {0.6350, 0.0780, 0.1840}
    }};
    return colors;
}

// ----------------------------------------------------------------------------

int main(int argc, char *argv[])
{

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

    //
    // Load point cloud
    //

    auto cloud_ptr = LoadPointCloud(argv[1]);
    std::cout << "Loaded point cloud with " << cloud_ptr->size() << " points." << std::endl;

    //
    // Estimate normals
    //

    auto t1 = std::chrono::high_resolution_clock::now();
    CalcNormals(cloud_ptr, 0.25, nrNeighbors);
    const double t_normals = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::high_resolution_clock::now() - t1).count();
    std::cout << "o3d EstimateNormals: " << t_normals << " seconds" << std::endl;

    //
    // KD tree search with pcl
    //

    t1 = std::chrono::high_resolution_clock::now();

    std::cout << "Constructing kdtree..." << std::endl;
    pcl::KdTreeFLANN<pcl::PointNormal> kdtree;
    kdtree.setInputCloud(cloud_ptr);

    std::vector<std::vector<int>> neighbors;
    neighbors.resize(cloud_ptr->size());

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

    //
    // Run RSPD aka PlaneDetector
    //

    t1 = std::chrono::high_resolution_clock::now();
    PlaneDetector rspd(cloud_ptr, neighbors);
    rspd.minNormalDiff(std::cos(GeometryUtils::deg2rad(minNormalDiff)));
    rspd.maxDist(std::cos(GeometryUtils::deg2rad(maxDist)));
    rspd.outlierRatio(outlierRatio);

    std::set<Plane *> planes = rspd.detect();
    const double t_rspd = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::high_resolution_clock::now() - t1).count();
    std::cout << "rspd detect: " << t_rspd << " seconds" << std::endl;
    std::cout << std::endl;

    //
    // Print out the detected planes
    //

    std::cout << "==============================" << std::endl;
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

    //
    // Visualization
    //

    // Create a renderer, render window, and interactor
    vtkNew<vtkRenderer> renderer;
    vtkNew<vtkRenderWindow> renderWindow;
    renderWindow->AddRenderer(renderer);

    pcl::visualization::PCLVisualizer visualizer(renderer, renderWindow, "Points and planes");
    visualizer.getRenderWindow()->GlobalWarningDisplayOff();
    visualizer.setBackgroundColor(1.0, 1.0, 1.0);
    visualizer.initCameraParameters();
    visualizer.setSize(1600, 900);
    visualizer.setCameraPosition(0, 0, 10, 0, 1, 0); // Move Camera 10 up and have cam-up towards +Y
    visualizer.addCoordinateSystem(1.0, "global");
    visualizer.addPointCloud<pcl::PointNormal>(cloud_ptr, pcl::visualization::PointCloudColorHandlerCustom<pcl::PointNormal>{cloud_ptr, 255.0, 0.0, 0.0}, "cloud");

    // add any planes
    size_t i = 0;
    for (const auto &p : planes)
    {
        AddPlane(p->center(), p->basisU(), p->basisV(), GetDefaultColorPalatte()[i % 6], renderer);
        ++i;
    }

    while (!visualizer.wasStopped())
    {
        visualizer.spinOnce(100);
        std::this_thread::sleep_for(100ms);
    }
    visualizer.close();

    return 0;
}
