// Adaptation of CGAL's official envelope simplification example; one pass.
#include <CGAL/Simple_cartesian.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/IO/OFF.h>
#include <CGAL/Surface_mesh_simplification/edge_collapse.h>
#include <CGAL/Surface_mesh_simplification/Policies/Edge_collapse/LindstromTurk_cost.h>
#include <CGAL/Surface_mesh_simplification/Policies/Edge_collapse/LindstromTurk_placement.h>
#include <CGAL/Surface_mesh_simplification/Policies/Edge_collapse/Bounded_normal_change_filter.h>
#include <CGAL/Surface_mesh_simplification/Policies/Edge_collapse/Polyhedral_envelope_filter.h>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
using K=CGAL::Simple_cartesian<double>;using Mesh=CGAL::Surface_mesh<K::Point_3>;namespace SMS=CGAL::Surface_mesh_simplification;
struct Face_stop{const Mesh* mesh;template<class F,class P>bool operator()(const F&,const P&,std::size_t,std::size_t)const{return mesh->number_of_faces()<=1000;}};
int main(int argc,char**argv){try{
 if(argc!=4)return 2;if(std::filesystem::exists(argv[2])||std::filesystem::exists(argv[3]))throw std::runtime_error("Exclusive simplification output");
 Mesh m;std::ifstream input(argv[1]);if(!CGAL::IO::read_OFF(input,m)||!CGAL::is_triangle_mesh(m)||!CGAL::is_closed(m))throw std::runtime_error("Qualified closed wrap required");
 size_t before=m.number_of_faces();if(before<=1000||before>100000)throw std::runtime_error("Simplification budget admission");
 SMS::Polyhedral_envelope_filter<K,SMS::Bounded_normal_change_filter<>> filter(.05); // .0005 * original D100
 auto removed=SMS::edge_collapse(m,Face_stop{&m},CGAL::parameters::get_cost(SMS::LindstromTurk_cost<Mesh>()).get_placement(SMS::LindstromTurk_placement<Mesh>()).filter(filter));
 m.collect_garbage();std::ofstream out(argv[2]);out<<std::setprecision(17);if(!CGAL::IO::write_OFF(out,m))throw std::runtime_error("Simplification export failed");
 std::ofstream r(argv[3]);r<<"{\"raw_triangles\":"<<before<<",\"final_triangles\":"<<m.number_of_faces()<<",\"removed_edges\":"<<removed<<",\"envelope_native\":0.05,\"one_pass\":true,\"asset_quality_PASS\":false}\n";
 std::cout<<"SIMPLIFICATION_EXPORTED faces="<<m.number_of_faces()<<std::endl;return 0;
 }catch(const std::exception&e){std::cerr<<e.what()<<std::endl;return 10;}}
