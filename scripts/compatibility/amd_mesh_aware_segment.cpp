// AI triangle-soup adapter to CGAL's official generic Region_growing API.
// No vertex welding, source edits, box assumptions, or RANSAC.
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Shape_detection/Region_growing/Region_growing.h>
#include <CGAL/linear_least_squares_fitting_3.h>
#include <CGAL/Eigen_diagonalize_traits.h>
#include <boost/property_map/property_map.hpp>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <algorithm>
#include <array>
#include <iostream>
#include <cstdint>
#include <cmath>

using K=CGAL::Exact_predicates_inexact_constructions_kernel;
using P=K::Point_3;using T=K::Triangle_3;using V=K::Vector_3;
struct Soup {
 std::vector<P> p;std::vector<std::array<uint32_t,3>> f;
 std::vector<std::array<int32_t,3>> adj;std::vector<uint8_t> active;
 std::vector<V> normals;std::vector<double> area;
 T triangle(size_t i)const{return T(p[f[i][0]],p[f[i][1]],p[f[i][2]]);}
};
struct Neighbors {
 const Soup& s;
 void operator()(size_t i,std::vector<size_t>& out)const {
  out.clear();for(auto j:s.adj[i])if(j>=0 && s.active[j])out.push_back(size_t(j));
 }
};
struct Region {
 using Item=size_t;using Primitive=K::Plane_3;using Items=std::vector<Item>;
 using Region_index_map=boost::iterator_property_map<std::vector<size_t>::iterator,boost::identity_property_map>;
 const Soup& s;std::vector<size_t> labels;Primitive plane;V unit=V(1,0,0);
 Region(const Soup& source):s(source),labels(s.f.size(),size_t(-1)){}
 Region_index_map region_index_map(){return boost::make_iterator_property_map(labels.begin(),boost::identity_property_map());}
 bool is_part_of_region(Item i,const Items&)const {
  if(!s.active[i] || std::abs(unit*s.normals[i])<std::cos(10*std::acos(-1.0)/180))return false;
  for(auto index:s.f[i])if(CGAL::squared_distance(s.p[index],plane)>0.2*0.2)return false;
  return true;
 }
 bool update(const Items& ids){
  if(ids.empty())return false;
  if(ids.size()==1)plane=s.triangle(ids[0]).supporting_plane();
  else {
   std::vector<T> triangles;triangles.reserve(ids.size());for(auto i:ids)triangles.push_back(s.triangle(i));
   P centroid;
   CGAL::linear_least_squares_fitting_3(triangles.begin(),triangles.end(),plane,centroid,
       CGAL::Dimension_tag<2>(),K(),CGAL::Eigen_diagonalize_traits<double,3>());
  }
  auto n=plane.orthogonal_vector();double len=std::sqrt(n.squared_length());
  if(!std::isfinite(len)||len<=0)return false;unit=n/len;
  return true;
 }
 bool is_valid_region(const Items& ids)const {
  if(ids.size()<24)return false;
  // Generic CGAL propagation can retain a region after a refit violates its
  // old membership; explicitly recheck every member against the FINAL plane.
  for(auto i:ids)if(!is_part_of_region(i,ids))return false;return true;
 }
 Primitive primitive()const{return plane;}
};
template<class T0>void read(std::ifstream& in,std::vector<T0>& v,size_t n){
 v.resize(n);in.read(reinterpret_cast<char*>(v.data()),std::streamsize(n*sizeof(T0)));
 if(!in)throw std::runtime_error("Truncated binary source");
}
int segment(const std::string& base){
 Soup s;std::ifstream in(base+"/input/mesh.bin",std::ios::binary);uint64_t nv=0,nf=0;
 in.read(reinterpret_cast<char*>(&nv),8);in.read(reinterpret_cast<char*>(&nf),8);
 if(!in||nv>1500000||nf>2500000||nv<3||nf<1)return 3;
 std::vector<std::array<double,3>> coords;read(in,coords,nv);s.p.reserve(nv);
 for(auto p:coords){for(auto x:p)if(!std::isfinite(x))return 4;s.p.emplace_back(p[0],p[1],p[2]);}
 coords.clear();coords.shrink_to_fit();read(in,s.f,nf);read(in,s.adj,nf);read(in,s.active,nf);
 if(in.peek()!=EOF)return 5;
 s.normals.reserve(nf);s.area.reserve(nf);std::vector<size_t> items;
 for(size_t i=0;i<nf;++i){
  for(auto v:s.f[i])if(v>=nv)return 6;
  for(auto j:s.adj[i])if(j < -1 || j>=int64_t(nf))return 7;
  auto t=s.triangle(i);V n=CGAL::cross_product(t[1]-t[0],t[2]-t[0]);double mag=std::sqrt(n.squared_length());
  if(!std::isfinite(mag)||mag<=0)return 8;s.area.push_back(mag/2);s.normals.push_back(n/mag);
  if(s.active[i])items.push_back(i);
 }
 auto seeds=items;
 std::stable_sort(seeds.begin(),seeds.end(),[&](size_t a,size_t b){return s.area[a]>s.area[b];});
 Neighbors neighbor{s};Region type(s);
 CGAL::Shape_detection::Region_growing<Neighbors,Region> grow(items,seeds,neighbor,type);
 using Found=CGAL::Shape_detection::Region_growing<Neighbors,Region>::Primitive_and_region;
 std::vector<Found> found;grow.detect(std::back_inserter(found));
 std::cout<<"MODLY_REGIONS "<<found.size()<<std::endl;
 std::ofstream out(base+"/regions.csv");out<<std::setprecision(17)<<"id,nx,ny,nz,offset,faces,area\n";
 std::vector<int32_t> labels(nf,-1);size_t id=0;
 for(const auto& r:found){
  auto n=r.first.orthogonal_vector();double length=std::sqrt(n.squared_length());n=n/length;
  double d=-r.first.d()/length;int axis=0;for(int a=1;a<3;++a)if(std::abs(n[a])>std::abs(n[axis]))axis=a;
  if(n[axis]<0){n=-n;d=-d;}
  double area=0;for(auto i:r.second){labels[i]=int32_t(id);area+=s.area[i];}
  out<<id<<','<<n.x()<<','<<n.y()<<','<<n.z()<<','<<d<<','<<r.second.size()<<','<<area<<'\n';++id;
 }
 std::ofstream labels_out(base+"/region-labels.bin",std::ios::binary);
 labels_out.write(reinterpret_cast<const char*>(labels.data()),labels.size()*sizeof(int32_t));
 if(!out||!labels_out)return 9;
 std::cout<<"MODLY_SEGMENTED faces="<<std::count_if(labels.begin(),labels.end(),[](int x){return x>=0;})<<" total="<<nf<<std::endl;
 return 0;
}
int main(int argc,char** argv){
 try{if(argc!=3||std::string(argv[1])!="segment")return 2;return segment(argv[2]);}
 catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 10;}
}
