// Reuse the bounded reference-distance certificate; restrict each output face
// to genuine source triangles compatible with its actual refined plane.
#define main modly_global_validation_entry
#include "amd_mesh_aware_validate.cpp"
#undef main
int main(int argc,char** argv){
 try{
  if(argc!=2)return 2;std::string root=argv[1];
  std::ifstream in(root+"/finite-domain-input.bin",std::ios::binary);uint64_t np=0,nr=0;in.read((char*)&np,8);in.read((char*)&nr,8);
  if(!in||np<4||np>48||nr>10000000)throw std::runtime_error("Finite domain input bounds");
  std::vector<std::array<double,4>> planes;std::vector<std::array<uint32_t,2>> rows;read(in,planes,np);read(in,rows,nr);
  Source source=load_source(root);Mesh mesh;std::ifstream file(root+"/validation/triangulated-native.off");
  if(!CGAL::IO::read_OFF(file,mesh)||mesh.number_of_faces()>1000)throw std::runtime_error("Candidate domain mesh");auto triangles=triangle_mesh(mesh);
  std::vector<std::vector<T>> candidate(np);std::vector<std::vector<uint32_t>> ids(np);
  for(auto row:rows){if(row[0]>=np||row[1]>=source.triangles.size())throw std::runtime_error("Finite domain index");ids[row[0]].push_back(row[1]);}
  for(auto t:triangles){
   V n=CGAL::cross_product(t[1]-t[0],t[2]-t[0]);n=n/std::sqrt(n.squared_length());int selected=-1;
   for(size_t i=0;i<np;++i){auto p=planes[i];V pn(p[0],p[1],p[2]);if(std::abs(n*pn)<std::cos(1e-5*std::acos(-1.)/180))continue;
    double residual=0;for(int k=0;k<3;++k)residual=std::max(residual,std::abs(p[0]*t[k].x()+p[1]*t[k].y()+p[2]*t[k].z()-p[3]));
    if(residual<=1e-6){if(selected!=-1)throw std::runtime_error("Ambiguous output plane");selected=int(i);}
   }
   if(selected<0)throw std::runtime_error("Output plane has no observed source group");candidate[selected].push_back(t);
  }
  std::string result=root+"/domain-certification.json";if(std::filesystem::exists(result))throw std::runtime_error("Domain evidence consumed");
  size_t queries=0,certified=0;bool all=true;std::ostringstream stats;stats<<'[';
  for(size_t i=0;i<np;++i){
   if(i)stats<<',';Triangles domain;domain.reserve(ids[i].size());for(auto row:ids[i])domain.push_back(source.triangles[row]);
   if(domain.empty())throw std::runtime_error("Absent finite source domain");Tree tree(domain.begin(),domain.end());tree.accelerate_distance_queries();
   bool ok=true;for(auto t:candidate[i]){if(!certify(tree,t,queries)){ok=false;break;}++certified;}
   all=all&&ok;stats<<"{\"group\":"<<i<<",\"source_triangles\":"<<domain.size()<<",\"candidate_triangles\":"<<candidate[i].size()<<",\"certified\":"<<(ok?"true":"false")<<'}';
  }
  stats<<']';std::ofstream out(result);out<<"{\"all_output_faces_match_actual_refined_planes\":true,\"complete_compatible_domain_footprints\":"<<(all?"true":"false")<<",\"certified_triangles\":"<<certified<<",\"queries\":"<<queries<<",\"groups\":"<<stats.str()<<"}\n";
  std::cout<<"MODLY_COMPATIBLE_DOMAINS certified="<<certified<<" queries="<<queries<<" pass="<<all<<std::endl;return 0;
 }catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 10;}
}
