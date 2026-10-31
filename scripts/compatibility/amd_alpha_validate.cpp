// Reuse the reviewed CPU renderer, AABB distance and triangle-source readers.
#define main phase14_validation_entry_not_called
#include "amd_mesh_aware_validate.cpp"
#undef main

enum Certificate {CERTIFIED,UNSUPPORTED,UNRESOLVED};
struct Evidence {size_t queries=0;double witness_distance=0;P witness=P(0,0,0);};
Certificate support(Tree& tree,const T& t,Evidence& e,int depth=0){
 if(++e.queries>32000000)return UNRESOLVED;
 P c=CGAL::centroid(t[0],t[1],t[2]);double d=distance(tree,c),radius=0;
 for(int i=0;i<3;++i)radius=std::max(radius,std::sqrt(CGAL::to_double(CGAL::squared_distance(c,t[i]))));
 if(d+radius<.2-1e-8)return CERTIFIED;
 if(d>.2+1e-8){e.witness_distance=d;e.witness=c;return UNSUPPORTED;}
 if(depth>=20)return UNRESOLVED;
 int a=0,b=1,c2=2;double longest=CGAL::squared_distance(t[0],t[1]);
 if(CGAL::squared_distance(t[1],t[2])>longest){a=1;b=2;c2=0;longest=CGAL::squared_distance(t[1],t[2]);}
 if(CGAL::squared_distance(t[2],t[0])>longest){a=2;b=0;c2=1;}
 P m=CGAL::midpoint(t[a],t[b]);auto first=support(tree,T(t[a],m,t[c2]),e,depth+1);
 if(first!=CERTIFIED)return first;return support(tree,T(m,t[b],t[c2]),e,depth+1);
}
const char* label(Certificate c){return c==CERTIFIED?"CERTIFIED":c==UNSUPPORTED?"CONFIRMED_UNSUPPORTED":"UNRESOLVED";}
void certificate(const Triangles& triangles,Tree& tree,const std::string& path){
 Evidence e;size_t row=0;Certificate state=CERTIFIED;
 for(;row<triangles.size();++row){state=support(tree,triangles[row],e);if(state!=CERTIFIED)break;}
 std::ofstream r(path);r<<std::setprecision(17)<<"{\"classification\":\""<<label(state)<<"\",\"completed_triangles\":"<<row<<",\"total_triangles\":"<<triangles.size()<<",\"queries\":"<<e.queries<<",\"witness_distance_native\":"<<e.witness_distance<<",\"witness\":["<<e.witness.x()<<","<<e.witness.y()<<","<<e.witness.z()<<"],\"limit_native\":0.2,\"guard_native\":1e-8,\"exact_symbolic_claim\":false}\n";
}
int main(int argc,char** argv){try{
 if(argc!=4)return 2;std::string root=argv[1],out=argv[3];
 if(std::filesystem::exists(out))throw std::runtime_error("Exclusive validation consumed");std::filesystem::create_directory(out);
 Mesh mesh;std::ifstream input(argv[2]);if(!CGAL::IO::read_OFF(input,mesh)||!CGAL::is_triangle_mesh(mesh)||!mesh.number_of_faces()||mesh.number_of_faces()>1000)throw std::runtime_error("Frozen candidate budget/schema");
 auto candidate=triangle_mesh(mesh);auto source=load_source(root);Tree st(source.triangles.begin(),source.triangles.end()),ct(candidate.begin(),candidate.end());st.accelerate_distance_queries();ct.accelerate_distance_queries();
 std::ifstream samples(root+"/validation-points.bin",std::ios::binary);std::vector<std::array<double,3>> points;read(samples,points,65536);
 std::vector<double> sd;for(auto p:points)sd.push_back(distance(ct,P(p[0],p[1],p[2]))/100);
 std::ofstream sf(out+"/source-to-candidate.bin",std::ios::binary);sf.write((char*)sd.data(),sd.size()*8);sf.close();
 std::vector<double> cdf;double area=0;for(auto t:candidate){area+=std::sqrt(t.squared_area());cdf.push_back(area);}
 std::vector<double> cd;uint64_t state=20261030;auto uniform=[&](){state=state*6364136223846793005ULL+1442695040888963407ULL;return ((state>>11)+.5)/9007199254740992.0;};
 for(size_t i=0;i<65536;++i){size_t j=std::lower_bound(cdf.begin(),cdf.end(),area*(i+.5)/65536)-cdf.begin();double r=std::sqrt(uniform()),u=uniform();auto t=candidate[j];P p(t[0].x()*(1-r)+t[1].x()*r*(1-u)+t[2].x()*r*u,t[0].y()*(1-r)+t[1].y()*r*(1-u)+t[2].y()*r*u,t[0].z()*(1-r)+t[1].z()*r*(1-u)+t[2].z()*r*u);cd.push_back(distance(st,p)/100);}
 std::ofstream cf(out+"/candidate-to-source.bin",std::ios::binary);cf.write((char*)cd.data(),cd.size()*8);cf.close();
 certificate(candidate,st,out+"/candidate-full-support.json");certificate(source.triangles,ct,out+"/source-full-support.json");
 // All original face normals and vertex incidences in saved finite features.
 // A conservative expanded triangle box finds every candidate that could be
 // within .2 of any of its vertices; final distance/normal tests are unrelaxed.
 std::ifstream features(std::filesystem::path(out).parent_path()/"feature-input.bin",std::ios::binary);uint64_t groups=0;features.read((char*)&groups,8);if(groups!=51)throw std::runtime_error("Frozen feature groups");
 std::ofstream fr(out+"/features.jsonl");fr<<std::setprecision(17);
 const double cosine=std::cos(10*std::acos(-1.)/180)+1e-12;
 for(uint64_t g=0;g<groups;++g){int32_t region;uint64_t count;features.read((char*)&region,4);features.read((char*)&count,8);if(!features||!count||count>source.triangles.size())throw std::runtime_error("Feature schema");double maximum=0;size_t missing=0,failures=0;
  for(uint64_t j=0;j<count;++j){uint32_t id;features.read((char*)&id,4);if(!features||id>=source.triangles.size())throw std::runtime_error("Feature face index");const T&t=source.triangles[id];V n=CGAL::cross_product(t[1]-t[0],t[2]-t[0]);double n2=n.squared_length();if(n2<=0)throw std::runtime_error("Degenerate source feature");auto bb=t.bbox();const double e=.2+1e-8;K::Iso_cuboid_3 box(bb.xmin()-e,bb.ymin()-e,bb.zmin()-e,bb.xmax()+e,bb.ymax()+e,bb.zmax()+e);
   std::vector<Triangles::iterator> hits;ct.all_intersected_primitives(box,std::back_inserter(hits));double best[3]={INFINITY,INFINITY,INFINITY};
   for(auto it:hits){const T&c=*it;V cn=CGAL::cross_product(c[1]-c[0],c[2]-c[0]);if(std::abs(cn*n)<cosine*std::sqrt(cn.squared_length()*n2))continue;for(int k=0;k<3;++k)best[k]=std::min(best[k],CGAL::to_double(CGAL::squared_distance(t[k],c)));}
   for(double d2:best){if(!std::isfinite(d2)){++missing;++failures;}else{maximum=std::max(maximum,std::sqrt(d2)/100);if(d2>(.2-1e-8)*(.2-1e-8))++failures;}}
  }
  fr<<"{\"region\":"<<region<<",\"source_faces\":"<<count<<",\"vertex_incidences\":"<<count*3<<",\"missing_local_normal_compatible_incidences\":"<<missing<<",\"failed_incidences\":"<<failures<<",\"maximum_measured_local_distance_over_D\":"<<maximum<<",\"PASS\":"<<(failures==0?"true":"false")<<"}\n";
 }
 fr.close();if(features.peek()!=std::char_traits<char>::eof())throw std::runtime_error("Feature trailing bytes");
 for(int az=0;az<360;az+=45){render(source.triangles,az,out+"/source-"+std::to_string(az));render(candidate,az,out+"/candidate-"+std::to_string(az));}
 std::ofstream r(out+"/native.json");r<<"{\"complete\":true,\"triangles\":"<<candidate.size()<<",\"source_triangles\":"<<source.triangles.size()<<",\"samples_each\":65536,\"views\":8,\"feature_groups\":"<<groups<<",\"plane_checks\":\"NOT_APPLICABLE\"}\n";
 std::cout<<"ALPHA_VALIDATION_MEASUREMENTS_COMPLETE"<<std::endl;return 0;
 }catch(const std::exception&e){std::cerr<<e.what()<<std::endl;return 10;}}
