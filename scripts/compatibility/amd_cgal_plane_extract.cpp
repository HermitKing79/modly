// Private proof adapter to official CGAL plane extraction. No box template.
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/IO/read_points.h>
#include <CGAL/property_map.h>
#include <CGAL/Shape_detection/Efficient_RANSAC.h>
#include <CGAL/Random.h>
#include <boost/tuple/tuple.hpp>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <iostream>

using K = CGAL::Exact_predicates_inexact_constructions_kernel;
using PNI = boost::tuple<K::Point_3, K::Vector_3, int, unsigned int>;
using Points = std::vector<PNI>;
using PM = CGAL::Nth_of_tuple_property_map<0, PNI>;
using NM = CGAL::Nth_of_tuple_property_map<1, PNI>;
using Traits = CGAL::Shape_detection::Efficient_RANSAC_traits<K, Points, PM, NM>;
using Ransac = CGAL::Shape_detection::Efficient_RANSAC<Traits>;
using Plane = CGAL::Shape_detection::Plane<Traits>;

int main(int argc, char** argv) {
  if (argc != 5) return 2; // input PLY, measured AI diagonal, output PLY, planes CSV
  const double diagonal = std::stod(argv[2]);
  if (!std::isfinite(diagonal) || diagonal <= 0) return 3;
  Points points;
  std::ifstream input(argv[1]);
  if (!CGAL::IO::read_PLY_with_properties(input, std::back_inserter(points),
      CGAL::IO::make_ply_point_reader(PM()), CGAL::IO::make_ply_normal_reader(NM()),
      std::make_pair(CGAL::Nth_of_tuple_property_map<3, PNI>(),
                     CGAL::IO::PLY_property<unsigned int>("sample_id")))
      || points.empty() || points.size() > 65536) return 4;
  for (auto& p: points) p.get<2>() = -1;
  CGAL::get_default_random() = CGAL::Random(20261030);
  Ransac ransac;
  ransac.set_input(points);
  ransac.add_shape_factory<Plane>();
  Ransac::Parameters params;
  params.probability = 0.001;
  params.min_points = 24;
  params.epsilon = 0.002 * diagonal;
  params.cluster_epsilon = 0.02 * diagonal;
  params.normal_threshold = std::cos(10.0 * std::acos(-1.0) / 180.0);
  if (!ransac.detect(params)) return 7; // One invocation; incomplete is failure.
  std::ofstream planes(argv[4]); planes << std::setprecision(17);
  planes << "id,nx,ny,nz,offset,training_points\n";
  std::size_t id = 0;
  for (const auto& shape : ransac.shapes()) {
    const Plane* plane = dynamic_cast<const Plane*>(shape.get());
    if (!plane) return 5;
    const auto n = plane->plane_normal();
    const auto& assigned = plane->indices_of_assigned_points();
    for (auto i : assigned) {
      points[i].get<2>() = static_cast<int>(id);
    }
    // CGAL's exact detected plane convention is n.p + d = 0. Do not recenter.
    planes << id << ',' << n.x() << ',' << n.y() << ',' << n.z() << ',' << -plane->d() << ',' << assigned.size() << '\n';
    ++id;
  }
  std::ofstream out(argv[3]); out << std::setprecision(17);
  out << "ply\nformat ascii 1.0\nelement vertex " << points.size()
      << "\nproperty double x\nproperty double y\nproperty double z"
      << "\nproperty double nx\nproperty double ny\nproperty double nz"
      << "\nproperty int segment_index\nproperty uint sample_id\nend_header\n";
  for (const auto& row : points) out << row.get<0>() << ' ' << row.get<1>() << ' ' << row.get<2>() << ' ' << row.get<3>() << '\n';
  std::cout << "MODLY_RANSAC planes=" << id << " points=" << points.size()
            << " unassigned=" << ransac.number_of_unassigned_points() << '\n';
  return (id >= 4 && id <= 48 && planes.good() && out.good()) ? 0 : 6;
}
