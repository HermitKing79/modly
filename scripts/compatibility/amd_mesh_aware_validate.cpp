// Independent triangle-surface, intersection and matched CPU raster checks.
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/IO/OFF.h>
#include <CGAL/Polygon_mesh_processing/triangulate_faces.h>
#include <CGAL/Polygon_mesh_processing/self_intersections.h>
#include <CGAL/AABB_tree.h>
#include <CGAL/AABB_traits_3.h>
#include <CGAL/AABB_triangle_primitive_3.h>
#include <fstream>
#include <iomanip>
#include <array>
#include <vector>
#include <cmath>
#include <filesystem>
#include <iostream>
using K=CGAL::Exact_predicates_inexact_constructions_kernel;using P=K::Point_3;using V=K::Vector_3;using T=K::Triangle_3;
using Triangles=std::vector<T>;using Primitive=CGAL::AABB_triangle_primitive_3<K,Triangles::iterator>;using Tree=CGAL::AABB_tree<CGAL::AABB_traits_3<K,Primitive>>;
using Mesh=CGAL::Surface_mesh<P>;
template<class Z>void read(std::ifstream& f,std::vector<Z>& v,size_t n){v.resize(n);f.read(reinterpret_cast<char*>(v.data()),sizeof(Z)*n);if(!f)throw std::runtime_error("Truncated validation binary");}
struct Source {std::vector<P> vertices;Triangles triangles;};
Source load_source(const std::string& root){
 std::ifstream in(root+"/input/mesh.bin",std::ios::binary);uint64_t nv=0,nf=0;in.read((char*)&nv,8);in.read((char*)&nf,8);
 if(!in||nv>1500000||nf>2500000)throw std::runtime_error("Source dimensions");
 std::vector<std::array<double,3>> p;std::vector<std::array<uint32_t,3>> f;read(in,p,nv);read(in,f,nf);
 Source s;s.vertices.reserve(nv);for(auto a:p)s.vertices.emplace_back(a[0],a[1],a[2]);s.triangles.reserve(nf);
 for(auto a:f){for(auto i:a)if(i>=nv)throw std::runtime_error("Source index");s.triangles.emplace_back(s.vertices[a[0]],s.vertices[a[1]],s.vertices[a[2]]);}return s;
}
Triangles triangle_mesh(const Mesh& mesh){
 Triangles out;for(auto face:mesh.faces()){
  std::vector<P> p;for(auto v:CGAL::vertices_around_face(mesh.halfedge(face),mesh))p.push_back(mesh.point(v));
  if(p.size()!=3)throw std::runtime_error("Triangulation incomplete");out.emplace_back(p[0],p[1],p[2]);
 }return out;
}
double distance(Tree& tree,const P& p){return std::sqrt(CGAL::to_double(tree.squared_distance(p)));}
bool certify(Tree& tree,const T& triangle,size_t& queries,int depth=0){
 if(++queries>2000000)throw std::runtime_error("Finite footprint query cap");
 P c=CGAL::centroid(triangle[0],triangle[1],triangle[2]);double d=distance(tree,c);double radius=0;
 for(int i=0;i<3;++i)radius=std::max(radius,std::sqrt(CGAL::to_double(CGAL::squared_distance(c,triangle[i]))));
 // Distance to a fixed reference surface is 1-Lipschitz; this certifies the
 // ENTIRE triangle cell, not just samples. Native source diagonal is100.
 if(d+radius<0.2-1e-8)return true;
 if(d>0.2||depth>=20)return false;
 // Longest-edge bisection avoids oversampling narrow genuine bevel triangles.
 int a=0,b=1,c2=2;double longest=CGAL::squared_distance(triangle[0],triangle[1]);
 if(CGAL::squared_distance(triangle[1],triangle[2])>longest){a=1;b=2;c2=0;longest=CGAL::squared_distance(triangle[1],triangle[2]);}
 if(CGAL::squared_distance(triangle[2],triangle[0])>longest){a=2;b=0;c2=1;}
 P midpoint=CGAL::midpoint(triangle[a],triangle[b]);
 return certify(tree,T(triangle[a],midpoint,triangle[c2]),queries,depth+1)&&certify(tree,T(midpoint,triangle[b],triangle[c2]),queries,depth+1);
}
struct Projected {double x,y,z;};
double edge(const Projected&a,const Projected&b,double x,double y){return (b.x-a.x)*(y-a.y)-(b.y-a.y)*(x-a.x);}
void render(const Triangles& triangles,int az,const std::string& path){
 const double pi=std::acos(-1.0),a=az*pi/180,e=20*pi/180;V view(std::cos(a)*std::cos(e),std::sin(a)*std::cos(e),std::sin(e));
 V right(-std::sin(a),std::cos(a),0),up=CGAL::cross_product(view,right);P origin(0,0,0);
 std::vector<double> depth(1024*1024,-1e100);std::vector<unsigned char> pixels(1024*1024*3,235),mask(1024*1024,0);
 for(auto& triangle:triangles){
  Projected p[3];for(int i=0;i<3;++i){auto q=triangle[i]-origin;p[i]={512+(q*right)*1024/115,512-(q*up)*1024/115,q*view};}
  double ar=edge(p[0],p[1],p[2].x,p[2].y);if(std::abs(ar)<1e-18)continue;
  int lx=std::max(0,int(std::floor(std::min({p[0].x,p[1].x,p[2].x})))),hx=std::min(1023,int(std::ceil(std::max({p[0].x,p[1].x,p[2].x}))));
  int ly=std::max(0,int(std::floor(std::min({p[0].y,p[1].y,p[2].y})))),hy=std::min(1023,int(std::ceil(std::max({p[0].y,p[1].y,p[2].y}))));
  V n=CGAL::cross_product(triangle[1]-triangle[0],triangle[2]-triangle[0]);double length=std::sqrt(n.squared_length());if(length<=0)throw std::runtime_error("Zero area output");n=n/length;
  V light=right*.35+up*.6+view*.72;light=light/std::sqrt(light.squared_length());int shade=int(70+155*std::abs(n*light));
  for(int y=ly;y<=hy;++y)for(int x=lx;x<=hx;++x){double px=x+.5,py=y+.5,w0=edge(p[1],p[2],px,py)/ar,w1=edge(p[2],p[0],px,py)/ar,w2=1-w0-w1;
   if(w0>=-1e-12&&w1>=-1e-12&&w2>=-1e-12){size_t k=y*1024+x;double z=w0*p[0].z+w1*p[1].z+w2*p[2].z;
    mask[k]=255;if(z>depth[k]){depth[k]=z;for(int c=0;c<3;++c)pixels[k*3+c]=static_cast<unsigned char>(shade);}}
  }
 }
 std::ofstream image(path+".ppm",std::ios::binary);image<<"P6\n1024 1024\n255\n";image.write((char*)pixels.data(),pixels.size());
 std::ofstream silhouette(path+".pgm",std::ios::binary);silhouette<<"P5\n1024 1024\n255\n";silhouette.write((char*)mask.data(),mask.size());
}
int main(int argc,char** argv){
 try{
  if(argc!=2)return 2;std::string root=argv[1],out=root+"/validation";
  if(std::filesystem::exists(out))throw std::runtime_error("Exclusive validation path consumed");std::filesystem::create_directory(out);
  Mesh mesh;std::ifstream input(root+"/owned-session/candidate/raw-reconstruction.off");
  if(!CGAL::IO::read_OFF(input,mesh)||!CGAL::Polygon_mesh_processing::triangulate_faces(mesh))throw std::runtime_error("Candidate read/triangulation failed");
  if(mesh.number_of_faces()>1000||mesh.number_of_faces()==0)throw std::runtime_error("Frozen triangle budget");
  std::vector<std::pair<Mesh::Face_index,Mesh::Face_index>> intersections;
  CGAL::Polygon_mesh_processing::self_intersections(mesh,std::back_inserter(intersections));
  std::ofstream ti(out+"/triangulated-native.off");ti<<std::setprecision(17);CGAL::IO::write_OFF(ti,mesh);ti.close();
  auto candidate=triangle_mesh(mesh);auto source=load_source(root);
  Tree st(source.triangles.begin(),source.triangles.end()),ct(candidate.begin(),candidate.end());st.accelerate_distance_queries();ct.accelerate_distance_queries();
  std::ifstream samples(root+"/validation-points.bin",std::ios::binary);std::vector<std::array<double,3>> points;read(samples,points,65536);
  std::vector<double> sdist;sdist.reserve(65536);for(auto p:points)sdist.push_back(distance(ct,P(p[0],p[1],p[2]))/100);
  std::ofstream sf(out+"/source-to-candidate.bin",std::ios::binary);sf.write((char*)sdist.data(),sdist.size()*sizeof(double));
  double maxvertex=0;for(auto p:source.vertices)maxvertex=std::max(maxvertex,distance(ct,p)/100);
  // Deterministic candidate area-strata; independent barycentric sequence.
  std::vector<double> cdf;double area=0;for(auto t:candidate){area+=std::sqrt(t.squared_area());cdf.push_back(area);}
  std::vector<double> cdist;cdist.reserve(65536);uint64_t state=20261030;
  auto uniform=[&](){state=state*6364136223846793005ULL+1442695040888963407ULL;return ((state>>11)+.5)/9007199254740992.0;};
  for(size_t i=0;i<65536;++i){size_t j=std::lower_bound(cdf.begin(),cdf.end(),area*(i+.5)/65536)-cdf.begin();double r=std::sqrt(uniform()),u=uniform();auto t=candidate[j];
   P p(t[0].x()*(1-r)+t[1].x()*r*(1-u)+t[2].x()*r*u,t[0].y()*(1-r)+t[1].y()*r*(1-u)+t[2].y()*r*u,t[0].z()*(1-r)+t[1].z()*r*(1-u)+t[2].z()*r*u);cdist.push_back(distance(st,p)/100);
  }
  std::ofstream cf(out+"/candidate-to-source.bin",std::ios::binary);cf.write((char*)cdist.data(),cdist.size()*sizeof(double));
  size_t queries=0;bool footprint=true;for(auto t:candidate){if(!certify(st,t,queries)){footprint=false;break;}}
  for(int az=0;az<360;az+=45){render(source.triangles,az,out+"/source-"+std::to_string(az));render(candidate,az,out+"/candidate-"+std::to_string(az));}
  std::ofstream result(out+"/native.json");result<<std::setprecision(17)<<"{\"triangles\":"<<candidate.size()<<",\"vertices\":"<<mesh.number_of_vertices()<<",\"complete_self_intersections\":"<<intersections.size()<<",\"full_source_vertex_distance_max_over_D\":"<<maxvertex<<",\"complete_finite_footprint_certified\":"<<(footprint?"true":"false")<<",\"footprint_queries\":"<<queries<<"}\n";
  std::cout<<"MODLY_VALIDATION_COMPLETE triangles="<<candidate.size()<<" intersections="<<intersections.size()<<" footprint="<<footprint<<std::endl;return 0;
 }catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 10;}
}
