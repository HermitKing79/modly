// Read-only two-output feature census and original finite-source context.
#define main phase14_validation_entry_not_called
#include "amd_mesh_aware_validate.cpp"
#undef main
#include <CGAL/intersections.h>
#include <map>
#include <set>
#include <variant>
using Indices=std::array<uint32_t,3>;
struct CheckedOutput:std::ofstream {explicit CheckedOutput(const std::string&p,std::ios_base::openmode m=std::ios::out):std::ofstream(){exceptions(std::ios::badbit|std::ios::failbit);open(p,m);}};
const double limit=.2-1e-8,cosine=std::cos(10*std::acos(-1.)/180)+1e-12;
V normal(const T&t){V n=CGAL::cross_product(t[1]-t[0],t[2]-t[0]);if(!std::isfinite(n.squared_length())||n.squared_length()<=0)throw std::runtime_error("Invalid source/output normal");return n;}
double angle(const V&a,const V&b){return std::acos(std::min(1.,std::abs(a*b)/std::sqrt(a.squared_length()*b.squared_length())))*180/std::acos(-1.);}
bool compatible(const V&a,const V&b){return std::abs(a*b)>=cosine*std::sqrt(a.squared_length()*b.squared_length());}
void point(std::ostream&o,const P&p){o<<'['<<p.x()<<','<<p.y()<<','<<p.z()<<']';}
void vector_json(std::ostream&o,const V&n){double l=std::sqrt(n.squared_length());o<<'['<<n.x()/l<<','<<n.y()/l<<','<<n.z()/l<<']';}
struct Output{
 Triangles triangles;std::vector<V> normals;std::unique_ptr<Tree> tree;
 Output(const std::string& path){Mesh m;std::ifstream f(path);if(!CGAL::IO::read_OFF(f,m)||!CGAL::is_triangle_mesh(m)||m.number_of_faces()>100000||!m.number_of_faces())throw std::runtime_error("Saved output schema/cap");triangles=triangle_mesh(m);for(auto&t:triangles)normals.push_back(normal(t));tree=std::make_unique<Tree>(triangles.begin(),triangles.end());tree->accelerate_distance_queries();}
 uint8_t failures(const T&t){auto bb=t.bbox();double e=.2+1e-8;K::Iso_cuboid_3 box(bb.xmin()-e,bb.ymin()-e,bb.zmin()-e,bb.xmax()+e,bb.ymax()+e,bb.zmax()+e);std::vector<Triangles::iterator> hits;tree->all_intersected_primitives(box,std::back_inserter(hits));V n=normal(t);double best[3]={INFINITY,INFINITY,INFINITY};for(auto it:hits){size_t j=it-triangles.begin();if(!compatible(n,normals[j]))continue;for(int k=0;k<3;++k)best[k]=std::min(best[k],CGAL::to_double(CGAL::squared_distance(t[k],*it)));}uint8_t mask=0;for(int k=0;k<3;++k)if(best[k]>limit*limit)mask|=1<<k;return mask;}
 void metric(std::ostream&o,const P&p,const V&n){auto nearest=tree->closest_point_and_primitive(p);size_t id=nearest.second-triangles.begin();double best=INFINITY,nearangle=INFINITY;int match=-1,nearid=-1;
  for(size_t j=0;j<triangles.size();++j){double d=CGAL::to_double(CGAL::squared_distance(p,triangles[j]));if(d<=limit*limit){double a=angle(n,normals[j]);if(a<nearangle){nearangle=a;nearid=int(j);}}if(compatible(n,normals[j])&&d<best){best=d;match=int(j);}}
  o<<"{\"nearest_triangle\":"<<id<<",\"nearest_point_native\":";point(o,nearest.first);o<<",\"distance_over_D\":"<<std::sqrt(CGAL::squared_distance(p,nearest.first))/100<<",\"nearest_angle_degrees\":"<<angle(n,normals[id])<<",\"nearest_compatible_triangle\":"<<match<<",\"nearest_compatible_distance_over_D\":";if(match<0)o<<"null";else o<<std::sqrt(best)/100;o<<",\"minimum_nearby_angle_degrees\":";if(nearid<0)o<<"null";else o<<nearangle;o<<",\"minimum_nearby_angle_triangle\":"<<nearid<<",\"PASS\":"<<(best<=limit*limit?"true":"false")<<'}';
 }
};
struct Edge{uint64_t key;uint32_t face;};
uint64_t edgekey(uint32_t a,uint32_t b){return (uint64_t(std::min(a,b))<<32)|std::max(a,b);}
struct Contact{int dimension=-1;std::string state="NONE";double area=0;};
Contact contact(const T&a,const T&b,const Indices&ia,const Indices&ib){
 auto hit=CGAL::intersection(a,b);if(!hit)return {};std::vector<P> common;for(int i=0;i<3;++i)if(std::find(ib.begin(),ib.end(),ia[i])!=ib.end())common.push_back(a[i]);
 std::vector<P> points;Contact r;
 if(auto p=std::get_if<P>(&*hit)){r.dimension=0;points.push_back(*p);}
 else if(auto s=std::get_if<K::Segment_3>(&*hit)){r.dimension=1;points={s->source(),s->target()};}
 else if(auto t=std::get_if<T>(&*hit)){r.dimension=2;points={(*t)[0],(*t)[1],(*t)[2]};r.area=std::sqrt(t->squared_area());}
 else if(auto p=std::get_if<std::vector<P>>(&*hit)){r.dimension=2;points=*p;for(size_t i=1;i+1<p->size();++i)r.area+=std::sqrt(T((*p)[0],(*p)[i],(*p)[i+1]).squared_area());}
 else throw std::runtime_error("Unknown triangle contact type");
 if(common.empty()){r.state="NON_INDEX_CONNECTED_CONTACT";return r;}
 bool allowed=true;double maxdist2=0;
 for(auto&p:points){if(common.size()==1){allowed=allowed&&(p==common[0]);maxdist2=std::max(maxdist2,CGAL::to_double(CGAL::squared_distance(p,common[0])));}else if(common.size()==2){K::Segment_3 edge(common[0],common[1]);allowed=allowed&&edge.has_on(p);maxdist2=std::max(maxdist2,CGAL::to_double(CGAL::squared_distance(p,edge)));}else throw std::runtime_error("Duplicate source index triangle");}
 r.state=allowed?"SHARED_INDEX_SIMPLEX":maxdist2<=1e-16?"NUMERICALLY_NEAR_SHARED_SIMPLEX":"BEYOND_SHARED_INDEX_SIMPLEX";return r;
}
void fixtures(const std::string& path){if(std::filesystem::exists(path))throw std::runtime_error("Exclusive fixtures");T a(P(0,0,0),P(1,0,0),P(0,1,0)),b(P(1,0,0),P(1,1,0),P(0,1,0)),c(P(.1,.1,0),P(.8,.1,0),P(.1,.8,0));
 if(contact(a,b,{0,1,2},{1,3,2}).state!="SHARED_INDEX_SIMPLEX")throw std::runtime_error("Shared edge fixture");
 auto x=contact(a,c,{0,1,2},{4,5,6});if(x.dimension!=2||x.area<=0||x.state!="NON_INDEX_CONNECTED_CONTACT")throw std::runtime_error("Overlap fixture");
 if(!compatible(V(0,0,1),V(0,0,-1))||compatible(V(0,0,1),V(1,0,0)))throw std::runtime_error("Unsigned normal fixture");
 CheckedOutput f(path);f<<"{\"classification\":\"PASS\",\"fixtures\":[\"shared-edge-contact\",\"non-indexed-positive-area-overlap\",\"unsigned-normal\",\"perpendicular-normal-rejection\"]}\n";
 f.close();}
int main(int argc,char**argv){try{
 if(argc==3&&std::string(argv[1])=="fixtures"){fixtures(argv[2]);return 0;}if(argc!=4)return 2;
 std::string src=argv[1],prior=argv[2],out=argv[3];if(std::filesystem::exists(out))throw std::runtime_error("Exclusive census consumed");std::filesystem::create_directory(out);
 auto source=load_source(src);if(source.vertices.size()!=1023995||source.triangles.size()!=2047210)throw std::runtime_error("Source dimensions");
 std::ifstream input(src+"/input/mesh.bin",std::ios::binary);input.seekg(16+source.vertices.size()*24);std::vector<Indices> index;read(input,index,source.triangles.size());
 std::vector<int32_t> region(source.triangles.size(),-1);std::ifstream fi(prior+"/feature-input.bin",std::ios::binary);uint64_t groups=0;fi.read((char*)&groups,8);if(groups!=51)throw std::runtime_error("Frozen group count");size_t tracked=0;
 for(size_t g=0;g<groups;++g){int32_t id;uint64_t n;fi.read((char*)&id,4);fi.read((char*)&n,8);if(!fi||id<0||id>=51||n>source.triangles.size())throw std::runtime_error("Feature schema");for(size_t j=0;j<n;++j){uint32_t face;fi.read((char*)&face,4);if(!fi||face>=region.size()||region[face]!=-1)throw std::runtime_error("Feature correspondence");region[face]=id;++tracked;}}
 if(tracked!=1925348||fi.peek()!=std::char_traits<char>::eof())throw std::runtime_error("Frozen tracked scope differs");
 Output raw(prior+"/raw-wrap.off"),final(prior+"/simplified.off");std::vector<uint8_t> rm(region.size(),0),fm(region.size(),0);std::vector<uint32_t> affected;uint64_t counts[4]={0,0,0,0},patch[51][4]={};
 CheckedOutput masks(out+"/tracked-face-outcomes.bin",std::ios::binary);masks.write((char*)&tracked,8);
 for(uint32_t i=0;i<region.size();++i)if(region[i]>=0){const T&t=source.triangles[i];rm[i]=raw.failures(t);fm[i]=final.failures(t);masks.write((char*)&i,4);masks.write((char*)&region[i],4);masks.write((char*)&rm[i],1);masks.write((char*)&fm[i],1);for(int k=0;k<3;++k){int c=((rm[i]>>k)&1)+2*((fm[i]>>k)&1);++counts[c];++patch[region[i]][c];}if(rm[i]||fm[i]){affected.push_back(i);if(affected.size()>10000)throw std::runtime_error("Affected face cap");}}
 masks.close();std::cout<<"CENSUS_MASKS_COMPLETE affected_faces="<<affected.size()<<" final_failed="<<counts[2]+counts[3]<<std::endl;
 if(counts[2]+counts[3]!=1568)throw std::runtime_error("Phase18 final incidence mismatch");
 std::vector<Edge> edges;edges.reserve(index.size()*3);for(uint32_t i=0;i<index.size();++i)for(int k=0;k<3;++k)edges.push_back({edgekey(index[i][k],index[i][(k+1)%3]),i});std::sort(edges.begin(),edges.end(),[](auto&a,auto&b){return a.key<b.key;});size_t boundaries=0,junctions=0;
 for(size_t i=0;i<edges.size();){size_t j=i+1;while(j<edges.size()&&edges[j].key==edges[i].key)++j;boundaries+=j-i==1;junctions+=j-i>2;i=j;}if(boundaries!=1421||junctions!=519)throw std::runtime_error("Original incidence differs");
 Tree st(source.triangles.begin(),source.triangles.end());st.build();size_t context_count=0;CheckedOutput records(out+"/affected-faces.jsonl");records<<std::setprecision(17);
 for(auto id:affected){const T&t=source.triangles[id];V n=normal(t);double area=std::sqrt(t.squared_area()),longest=0;for(int k=0;k<3;++k)longest=std::max(longest,std::sqrt(CGAL::squared_distance(t[k],t[(k+1)%3])));P center=CGAL::centroid(t[0],t[1],t[2]);
  records<<"{\"face\":"<<id<<",\"region\":"<<region[id]<<",\"indices\":["<<index[id][0]<<','<<index[id][1]<<','<<index[id][2]<<"],\"vertices_native\":[";for(int k=0;k<3;++k){if(k)records<<',';point(records,t[k]);}records<<"],\"normal\":";vector_json(records,n);records<<",\"area_over_D2\":"<<area/10000<<",\"minimum_altitude_over_D\":"<<2*area/longest/100<<",\"aspect_longest_over_altitude\":"<<longest*longest/(2*area)<<",\"raw_mask\":"<<int(rm[id])<<",\"final_mask\":"<<int(fm[id])<<",\"incidences\":[";
  bool first=true;for(int k=0;k<3;++k)if((rm[id]|fm[id])&(1<<k)){if(!first)records<<',';first=false;records<<"{\"corner\":"<<k<<",\"raw\":";raw.metric(records,t[k],n);records<<",\"final\":";final.metric(records,t[k],n);records<<'}';}records<<"],\"edges\":[";
  for(int k=0;k<3;++k){if(k)records<<',';uint64_t key=edgekey(index[id][k],index[id][(k+1)%3]);auto begin=std::lower_bound(edges.begin(),edges.end(),key,[](const Edge&e,uint64_t v){return e.key<v;});records<<"{\"vertices\":["<<std::min(index[id][k],index[id][(k+1)%3])<<','<<std::max(index[id][k],index[id][(k+1)%3])<<"],\"faces\":[";bool f=true;for(auto it=begin;it!=edges.end()&&it->key==key;++it){if(!f)records<<',';f=false;records<<it->face;}records<<"]}";}records<<"],\"nearby_source\":[";
  auto bb=t.bbox();double e=.2+1e-8;K::Iso_cuboid_3 box(bb.xmin()-e,bb.ymin()-e,bb.zmin()-e,bb.xmax()+e,bb.ymax()+e,bb.zmax()+e);std::vector<Triangles::iterator> hits;st.all_intersected_primitives(box,std::back_inserter(hits));std::sort(hits.begin(),hits.end());first=true;
  for(auto it:hits){uint32_t other=uint32_t(it-source.triangles.begin());if(other==id)continue;double md=CGAL::squared_distance(center,*it);int corner_mask=0;for(int k=0;k<3;++k){double d=CGAL::squared_distance(t[k],*it);md=std::min(md,d);if(d<=.04+1e-12)corner_mask|=1<<k;}if(md>.04+1e-12)continue;if(++context_count>2000000)throw std::runtime_error("Context cap");int shared=0;for(auto v:index[id])shared+=std::find(index[other].begin(),index[other].end(),v)!=index[other].end();auto c=contact(t,*it,index[id],index[other]);
   if(!first)records<<',';first=false;records<<"{\"face\":"<<other<<",\"region\":"<<region[other]<<",\"shared_indices\":"<<shared<<",\"near_corners_mask\":"<<corner_mask<<",\"minimum_probe_distance_over_D\":"<<std::sqrt(md)/100<<",\"normal\":";vector_json(records,normal(*it));records<<",\"angle_degrees\":"<<angle(n,normal(*it))<<",\"area_over_D2\":"<<std::sqrt(it->squared_area())/10000<<",\"contact_dimension\":"<<c.dimension<<",\"contact_state\":\""<<c.state<<"\",\"contact_area_over_D2\":"<<c.area/10000<<'}';
  }records<<"]}\n";
 }
 records.close();CheckedOutput summary(out+"/summary.json");summary<<"{\"classification\":\"COMPLETE_SAVED_OUTPUT_CENSUS\",\"source_faces\":"<<region.size()<<",\"tracked_faces\":"<<tracked<<",\"untracked_faces\":"<<region.size()-tracked<<",\"tracked_incidences\":"<<tracked*3<<",\"affected_faces\":"<<affected.size()<<",\"context_records\":"<<context_count<<",\"boundary_edges\":"<<boundaries<<",\"nonmanifold_edges\":"<<junctions<<",\"transition_order\":[\"pass_both\",\"raw_fail_final_pass\",\"raw_pass_final_fail\",\"fail_both\"],\"transitions\":["<<counts[0]<<','<<counts[1]<<','<<counts[2]<<','<<counts[3]<<"],\"patch_transitions\":[";for(int i=0;i<51;++i){if(i)summary<<',';summary<<'['<<patch[i][0]<<','<<patch[i][1]<<','<<patch[i][2]<<','<<patch[i][3]<<']';}summary<<"],\"geometry_modified\":false}\n";summary.close();std::cout<<"COMPLETE_SAVED_OUTPUT_CENSUS"<<std::endl;return 0;
 }catch(const std::exception&e){std::cerr<<e.what()<<std::endl;return 10;}}
