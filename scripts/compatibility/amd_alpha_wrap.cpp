// Source-only adapter to the unchanged CGAL triangle-soup interface.
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/alpha_wrap_3.h>
#include <CGAL/Random.h>
#include <CGAL/IO/OFF.h>
#include <array>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
using K=CGAL::Exact_predicates_inexact_constructions_kernel;
int main(int argc,char** argv){try{
 if(argc!=4)return 2;
 if(std::filesystem::exists(argv[2])||std::filesystem::exists(argv[3]))throw std::runtime_error("Exclusive output consumed");
 std::ifstream in(argv[1],std::ios::binary);uint64_t nv=0,nf=0;in.read((char*)&nv,8);in.read((char*)&nf,8);
 if(!in||nv!=1023995||nf!=2047210)throw std::runtime_error("Frozen raw dimensions differ");
 std::vector<K::Point_3> points;points.reserve(nv);CGAL::Bbox_3 bbox;
 for(size_t i=0;i<nv;++i){std::array<double,3> p;in.read((char*)p.data(),24);for(double x:p)if(!std::isfinite(x))throw std::runtime_error("Nonfinite input");points.emplace_back(p[0],p[1],p[2]);bbox+=points.back().bbox();}
 std::vector<std::array<size_t,3>> faces;faces.reserve(nf);
 for(size_t i=0;i<nf;++i){std::array<uint32_t,3> f;in.read((char*)f.data(),12);for(auto x:f)if(x>=nv)throw std::runtime_error("Source index");faces.push_back({f[0],f[1],f[2]});}
 if(!in)throw std::runtime_error("Truncated original soup");
 double dx=bbox.xmax()-bbox.xmin(),dy=bbox.ymax()-bbox.ymin(),dz=bbox.zmax()-bbox.zmin();double D=std::sqrt(dx*dx+dy*dy+dz*dz);
 if(std::abs(D-100)>1e-9)throw std::runtime_error("Source normalization differs");
 const double alpha=D/20.,offset=D/2000.;CGAL::Surface_mesh<K::Point_3> wrap;
 CGAL::get_default_random()=CGAL::Random(20261030);
 auto start=std::chrono::steady_clock::now();
 std::cout<<"AI_TRIANGLE_SOUP_WRAP_BEGIN vertices="<<nv<<" triangles="<<nf<<" alpha="<<alpha<<" offset="<<offset<<std::endl;
 CGAL::alpha_wrap_3(points,faces,alpha,offset,wrap);
 double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
 std::ofstream out(argv[2]);out<<std::setprecision(17);if(!CGAL::IO::write_OFF(out,wrap))throw std::runtime_error("Raw wrap export failed");out.close();
 std::ofstream r(argv[3]);r<<std::setprecision(17)<<"{\"label\":\"RAW ALPHA WRAP; UNVALIDATED AI-DERIVED RECONSTRUCTION\",\"input_vertices\":"<<nv<<",\"input_triangles\":"<<nf<<",\"vertices\":"<<wrap.number_of_vertices()<<",\"triangles\":"<<wrap.number_of_faces()<<",\"alpha_native\":"<<alpha<<",\"offset_native\":"<<offset<<",\"wrap_seconds\":"<<seconds<<"}\n";
 std::cout<<"AI_WRAP_EXPORTED triangles="<<wrap.number_of_faces()<<" seconds="<<seconds<<std::endl;return 0;
 }catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 10;}}
