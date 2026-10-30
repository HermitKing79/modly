// Official CGAL user-provided-plane example adapted for a pre-MIP observation gate.
#include <fstream>
#include <sstream>
#include <iomanip>
#include <filesystem>
#include <iostream>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/IO/read_points.h>
#include <CGAL/property_map.h>
#include <CGAL/Surface_mesh.h>
template<class Set> void modly_observe_planes(Set&);
#define MODLY_OBSERVE_PLANES(set) modly_observe_planes(set)
#include <CGAL/Polygonal_surface_reconstruction.h>
#include <CGAL/SCIP_mixed_integer_program_traits.h>

using K=CGAL::Exact_predicates_inexact_constructions_kernel;
using PNI=boost::tuple<K::Point_3,K::Vector_3,int>;
using PM=CGAL::Nth_of_tuple_property_map<0,PNI>;
using NM=CGAL::Nth_of_tuple_property_map<1,PNI>;
using IM=CGAL::Nth_of_tuple_property_map<2,PNI>;
using Mesh=CGAL::Surface_mesh<K::Point_3>;
static std::string root,mode;
std::string content(const std::string& path){std::ifstream f(path,std::ios::binary);if(!f)throw std::runtime_error("Missing gate file");return {std::istreambuf_iterator<char>(f),{}};}
void exclusive(const std::string& path,const std::string& text){
 if(std::filesystem::exists(path))throw std::runtime_error("Exclusive native evidence consumed");
 std::ofstream out(path,std::ios::binary);out<<text;if(!out)throw std::runtime_error("Native evidence write failed");
}
template<class Set>void modly_observe_planes(Set& set){
 std::ostringstream planes,members;planes<<std::setprecision(17)<<"id,nx,ny,nz,offset,faces\n";members<<"group,input_row\n";
 size_t id=0;
 for(auto segment:set.planar_segments()){
  auto plane=*segment->supporting_plane();auto n=plane.orthogonal_vector();double length=std::sqrt(n.squared_length());
  if(length<=0||!std::isfinite(length))throw std::runtime_error("Invalid final CGAL plane");n=n/length;
  planes<<id<<','<<n.x()<<','<<n.y()<<','<<n.z()<<','<<-plane.d()/length<<','<<segment->size()<<'\n';
  for(auto row:*segment)members<<id<<','<<row<<'\n';++id;
 }
 if(id<4||id>48)throw std::runtime_error("Actual final CGAL plane cap");
 if(mode=="reconstruct"){
  if(content(root+"/arrangement/prepared-planes.csv")!=planes.str()||content(root+"/arrangement/prepared-members.csv")!=members.str())throw std::runtime_error("Actual CGAL refit changed after admission");
 }
 std::string prefix=mode=="reconstruct"?"solver":"prepared";
 exclusive(root+"/arrangement/"+prefix+"-planes.csv",planes.str());exclusive(root+"/arrangement/"+prefix+"-members.csv",members.str());
}
int main(int argc,char** argv){
 try{
  if(argc!=3)return 2;mode=argv[1];root=argv[2];
  if(mode!="prepare-candidates"&&mode!="reconstruct"&&mode!="owned-session")return 3;
  std::ifstream in(root+"/augmented/train.ply");std::vector<PNI> points;
  if(!CGAL::IO::read_PLY_with_properties(in,std::back_inserter(points),CGAL::IO::make_ply_point_reader(PM()),CGAL::IO::make_ply_normal_reader(NM()),std::make_pair(IM(),CGAL::IO::PLY_property<int>("segment_index")))||points.empty()||points.size()>120000)return 4;
  CGAL::Polygonal_surface_reconstruction<K> algo(points,PM(),NM(),IM());
  Mesh candidates;algo.output_candidate_faces(candidates);
  if(candidates.number_of_faces()>20000||candidates.number_of_faces()<4)throw std::runtime_error("Unsafe candidate complexity");
  std::ostringstream encoded;encoded<<std::setprecision(17);CGAL::IO::write_OFF(encoded,candidates);
  if(mode=="prepare-candidates"){
   exclusive(root+"/arrangement/candidates.off",encoded.str());
   std::cout<<"MODLY_PREPARED_CANDIDATES "<<candidates.number_of_faces()<<std::endl;return 0;
  }
  if(mode=="owned-session"){
   exclusive(root+"/arrangement/candidates.off",encoded.str());
   exclusive(root+"/READY.txt","READY\n");
   std::string token;std::getline(std::cin,token);
   if(token!="GO"||std::cin.peek()!=EOF)throw std::runtime_error("Exact owned GO and EOF required");
  }else if(content(root+"/arrangement/candidates.off")!=encoded.str())throw std::runtime_error("Candidate arrangement changed after admission");
  exclusive(root+"/MIP-started.txt","ONE_BOUNDED_AI_MIP\n");
  Mesh output;
  if(!algo.reconstruct<CGAL::SCIP_mixed_integer_program_traits<double>>(output))throw std::runtime_error(algo.error_message());
  std::ostringstream raw;raw<<std::setprecision(17);CGAL::IO::write_OFF(raw,output);
  exclusive(root+"/candidate/raw-reconstruction.off",raw.str());
  std::cout<<"AI_DERIVED_RECONSTRUCTION vertices="<<output.number_of_vertices()<<" polygons="<<output.number_of_faces()<<std::endl;
  return 0;
 }catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 10;}
}
