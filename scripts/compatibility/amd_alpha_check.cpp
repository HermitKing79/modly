// Independent output checks; no geometry edits or repair.
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/IO/OFF.h>
#include <CGAL/Polygon_mesh_processing/self_intersections.h>
#include <CGAL/Polygon_mesh_processing/connected_components.h>
#include <CGAL/Polygon_mesh_processing/manifoldness.h>
#include <CGAL/Polygon_mesh_processing/orientation.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <set>
#include <array>
using K=CGAL::Exact_predicates_inexact_constructions_kernel;using Mesh=CGAL::Surface_mesh<K::Point_3>;
namespace PMP=CGAL::Polygon_mesh_processing;
int main(int argc,char**argv){try{
 if(argc!=3)return 2;if(std::filesystem::exists(argv[2]))throw std::runtime_error("Exclusive check output");
 Mesh m;std::ifstream input(argv[1]);if(!CGAL::IO::read_OFF(input,m)||!CGAL::is_triangle_mesh(m))throw std::runtime_error("Triangle mesh read failed");
 if(!m.number_of_faces()||m.number_of_faces()>100000)throw std::runtime_error("Bounded output complexity");
 size_t degenerate=0,duplicates=0,unused=0;std::set<std::array<size_t,3>> keys;std::set<K::Point_3> positions;
 for(auto v:m.vertices()){auto p=m.point(v);if(!std::isfinite(p.x())||!std::isfinite(p.y())||!std::isfinite(p.z()))throw std::runtime_error("Nonfinite output");positions.insert(p);unused+=m.is_isolated(v);}
 for(auto f:m.faces()){std::array<size_t,3> ids;std::array<K::Point_3,3> p;size_t k=0;for(auto v:CGAL::vertices_around_face(m.halfedge(f),m)){ids[k]=v.idx();p[k++]=m.point(v);}degenerate+=CGAL::collinear(p[0],p[1],p[2]);std::sort(ids.begin(),ids.end());duplicates+=!keys.insert(ids).second;}
 std::vector<Mesh::Halfedge_index> bad;PMP::non_manifold_vertices(m,std::back_inserter(bad));
 auto cc=m.add_property_map<Mesh::Face_index,size_t>("f:component",0).first;size_t components=PMP::connected_components(m,cc);
 std::vector<std::pair<Mesh::Face_index,Mesh::Face_index>> intersections;PMP::self_intersections(m,std::back_inserter(intersections));
 bool closed=CGAL::is_closed(m),valid=CGAL::is_valid_polygon_mesh(m),outward=closed&&PMP::is_outward_oriented(m);
 bool pass=closed&&valid&&outward&&bad.empty()&&components==1&&intersections.empty()&&!degenerate&&!duplicates&&!unused&&positions.size()==m.number_of_vertices();
 std::ofstream out(argv[2]);out<<"{\"classification\":\""<<(pass?"PASS_CLOSED_EMBEDDED_MESH":"FAIL_GEOMETRY")<<"\",\"vertices\":"<<m.number_of_vertices()<<",\"triangles\":"<<m.number_of_faces()<<",\"closed\":"<<(closed?"true":"false")<<",\"valid_polygon_mesh\":"<<(valid?"true":"false")<<",\"outward\":"<<(outward?"true":"false")<<",\"components\":"<<components<<",\"nonmanifold_vertices\":"<<bad.size()<<",\"complete_self_intersections\":"<<intersections.size()<<",\"degenerate_triangles\":"<<degenerate<<",\"duplicate_triangles\":"<<duplicates<<",\"unused_vertices\":"<<unused<<",\"duplicate_positions\":"<<(m.number_of_vertices()-positions.size())<<"}\n";
 std::cout<<"MESH_CHECK "<<(pass?"PASS":"FAIL")<<" faces="<<m.number_of_faces()<<std::endl;return 0;
 }catch(const std::exception&e){std::cerr<<e.what()<<std::endl;return 10;}}
