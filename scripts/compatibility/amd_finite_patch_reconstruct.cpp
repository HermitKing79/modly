// Private CGAL CPU adapter: complete face eligibility and explicit MIP vetoes.
// Dependency headers remain private; original objective/manifold code is reused.
#define main modly_previous_validation_entry
#include "amd_mesh_aware_validate.cpp"
#undef main
#include <CGAL/Polygon_2.h>
#include <CGAL/IO/read_points.h>
#include <CGAL/property_map.h>
#include <sstream>
#include <map>
#include <set>
#include <numeric>
template<class S>void modly_observe_planes(S&);
template<class M,class I,class G>bool modly_classify(M&,I,G);
template<class M,class I,class G,class S,class Vars>void modly_veto(M&,I,G,S&,const Vars&);
template<class M,class I,class G>bool modly_selection(M&,I,G,const std::vector<double>&,size_t);
void modly_before_solve();
#define MODLY_OBSERVE_PLANES(s) modly_observe_planes(s)
#define MODLY_CLASSIFY(m,i,g) modly_classify(m,i,g)
#define MODLY_VETO(m,i,g,s,v) modly_veto(m,i,g,s,v)
#define MODLY_BEFORE_SOLVE() modly_before_solve()
#define MODLY_SELECTION(m,i,g,x,n) modly_selection(m,i,g,x,n)
#include <CGAL/Polygonal_surface_reconstruction.h>
#include <CGAL/SCIP_mixed_integer_program_traits.h>
using Solver=CGAL::SCIP_mixed_integer_program_traits<double>;
using PNI=boost::tuple<P,V,int>;
using PM=CGAL::Nth_of_tuple_property_map<0,PNI>;
using NM=CGAL::Nth_of_tuple_property_map<1,PNI>;
using IM=CGAL::Nth_of_tuple_property_map<2,PNI>;
enum State {CERTIFIED_ELIGIBLE,CONFIRMED_UNSUPPORTED,UNRESOLVED};
const char* name(State s){return s==CERTIFIED_ELIGIBLE?"CERTIFIED_ELIGIBLE":s==CONFIRMED_UNSUPPORTED?"CONFIRMED_UNSUPPORTED":"UNRESOLVED";}
struct Result {State state=UNRESOLVED;std::string reason="NOT_CLASSIFIED";P witness=P(0,0,0);double distance=0;size_t queries=0;int depth=0;};
struct Face {size_t variable=0,descriptor=0,group=0;double planarity=0;std::vector<P> vertices;std::string key;Result proof;};
static std::string base,prior,out;
static bool fixture=false,require_nonempty=false,empty_optimum=false,insufficient=false;
static std::vector<PNI> input_points;
static std::map<const K::Plane_3*,size_t> plane_groups;
static std::vector<std::array<double,4>> plane_coeffs,domain_coeffs;
static std::vector<Triangles> fixture_domains;
static std::vector<std::vector<uint32_t>> domain_rows;
static std::vector<Face> records;
static size_t total_queries=0,selected_faces=0,expected_variables=0;
static int face_max_depth=0;
static constexpr size_t QUERY_LIMIT=2000000;
static constexpr int DEPTH_LIMIT=20;
static constexpr double EPS=.2,MARGIN=1e-8,PLANARITY_LIMIT=1e-10;

void exclusive(const std::string& path,const std::string& text){
 if(std::filesystem::exists(path))throw std::runtime_error("Exclusive evidence consumed: "+path);
 std::ofstream stream(path,std::ios::binary);stream<<text;stream.flush();if(!stream)throw std::runtime_error("Evidence write failed");
}
std::string contents(const std::string& path){std::ifstream f(path,std::ios::binary);if(!f)throw std::runtime_error("Missing authenticated input");return {std::istreambuf_iterator<char>(f),{}};}
std::array<double,4> coefficients(const K::Plane_3& plane){
 V n=plane.orthogonal_vector();double length=std::sqrt(n.squared_length());
 if(!(length>0)||!std::isfinite(length))throw std::runtime_error("Invalid plane");
 n=n/length;return {n.x(),n.y(),n.z(),-plane.d()/length};
}
std::array<double,4> canonical(std::array<double,4> p){int axis=0;for(int i=1;i<3;++i)if(std::abs(p[i])>std::abs(p[axis]))axis=i;if(p[axis]<0)for(auto& x:p)x=-x;return p;}
std::string geometry_key(size_t group,const std::vector<P>& points){
 std::vector<std::string> strings;for(auto p:points){std::ostringstream s;s<<std::hexfloat<<p.x()<<','<<p.y()<<','<<p.z();strings.push_back(s.str());}
 std::string best;for(size_t i=0;i<strings.size();++i)for(int sign:{-1,1}){
  std::string key=std::to_string(group)+"|";for(size_t j=0;j<strings.size();++j)key+=strings[(i+strings.size()+sign*int(j))%strings.size()]+";";
  if(best.empty()||key<best)best=key;
 }return best;
}
Result query(Tree& tree,const P& point){
 Result result;if(total_queries>=QUERY_LIMIT){result.reason="QUERY_LIMIT";return result;}
 ++total_queries;double d=distance(tree,point);if(!std::isfinite(d))throw std::runtime_error("Nonfinite reference distance");
 result.distance=d;result.witness=point;
 if(d>EPS+MARGIN){result.state=CONFIRMED_UNSUPPORTED;result.reason="OUTSIDE_COMPATIBLE_DOMAIN";}
 else {result.state=CERTIFIED_ELIGIBLE;result.reason="POINT_ONLY";}
 return result;
}
Result cell(Tree& tree,const T& triangle,int depth){
 face_max_depth=std::max(face_max_depth,depth);
 P c=CGAL::centroid(triangle[0],triangle[1],triangle[2]);auto r=query(tree,c);r.depth=depth;
 if(r.state==UNRESOLVED||r.state==CONFIRMED_UNSUPPORTED)return r;
 double radius=0;for(int k=0;k<3;++k)radius=std::max(radius,std::sqrt(CGAL::to_double(CGAL::squared_distance(c,triangle[k]))));
 if(r.distance+radius<EPS-MARGIN){r.reason="WHOLE_CELL_LIPSCHITZ";return r;}
 r.state=UNRESOLVED;r.reason="DEPTH_LIMIT_OR_NUMERICAL_BAND";if(depth>=DEPTH_LIMIT)return r;
 int a=0,b=1,c2=2;double longest=CGAL::squared_distance(triangle[0],triangle[1]);
 if(CGAL::squared_distance(triangle[1],triangle[2])>longest){a=1;b=2;c2=0;longest=CGAL::squared_distance(triangle[1],triangle[2]);}
 if(CGAL::squared_distance(triangle[2],triangle[0])>longest){a=2;b=0;c2=1;}
 P middle=CGAL::midpoint(triangle[a],triangle[b]);auto left=cell(tree,T(triangle[a],middle,triangle[c2]),depth+1);
 if(left.state==CONFIRMED_UNSUPPORTED)return left;
 auto right=cell(tree,T(middle,triangle[b],triangle[c2]),depth+1);
 if(right.state==CONFIRMED_UNSUPPORTED)return right;
 if(left.state==UNRESOLVED)return left;right.depth=std::max(left.depth,right.depth);return right;
}
Result polygon(Tree& tree,const std::vector<P>& points,const K::Plane_3& plane){
 size_t before=total_queries;face_max_depth=0;Result result;result.reason="UNSAFE_POLYGON";
 if(points.size()<3)return result;
 CGAL::Polygon_2<K> projected;std::set<std::tuple<double,double,double>> unique;
 for(auto p:points){
  if(!std::isfinite(p.x())||!std::isfinite(p.y())||!std::isfinite(p.z()))throw std::runtime_error("Nonfinite candidate geometry");
  if(!unique.insert({p.x(),p.y(),p.z()}).second)return result;
  if(CGAL::squared_distance(p,plane)>PLANARITY_LIMIT*PLANARITY_LIMIT){result.reason="NONPLANAR_CANDIDATE";return result;}
  projected.push_back(plane.to_2d(p));
 }
 if(!projected.is_simple()||!projected.is_convex()||projected.area()==0)return result;
 // Exact-predicate simplicity/convexity proves the fan covers the whole cell.
 // Collinear fan pieces are boundary subsets of the nondegenerate pieces.
 for(auto p:points){auto q=query(tree,p);if(q.state!=CERTIFIED_ELIGIBLE){q.queries=total_queries-before;return q;}}
 result.state=CERTIFIED_ELIGIBLE;result.reason="WHOLE_CONVEX_POLYGON_CERTIFIED";
 for(size_t i=1;i+1<points.size();++i){
  if(CGAL::collinear(points[0],points[i],points[i+1]))continue;
  auto part=cell(tree,T(points[0],points[i],points[i+1]),0);
  if(part.state==CONFIRMED_UNSUPPORTED){result=part;break;}
  if(part.state==UNRESOLVED)result=part;
 }
 // Alternate triangulations of this convex nearly-planar cell differ by at
 // most 2*PLANARITY_LIMIT, well below the reserved MARGIN. No output surface
 // can spend more than this explicit part of the distance guard.
 result.queries=total_queries-before;result.depth=face_max_depth;return result;
}

template<class S>void modly_observe_planes(S& set){
 plane_groups.clear();plane_coeffs.clear();
 std::vector<std::array<double,4>> expected;
 if(!fixture){
  std::istringstream in(contents(prior+"/owned-session/arrangement/prepared-planes.csv"));std::string line;std::getline(in,line);
  while(std::getline(in,line)){std::replace(line.begin(),line.end(),',',' ');std::istringstream row(line);size_t i,sz;std::array<double,4> p;
   if(!(row>>i>>p[0]>>p[1]>>p[2]>>p[3]>>sz)||i!=expected.size())throw std::runtime_error("Expected plane records");expected.push_back(p);
  }
 }
 size_t count=set.planar_segments().size();if(count<4||count>48)throw std::runtime_error("Refined plane cap");plane_coeffs.resize(count);std::set<size_t> seen;
 std::vector<std::pair<size_t,size_t>> membership;
 for(auto segment:set.planar_segments()){
  auto ptr=segment->supporting_plane();auto p=coefficients(*ptr);size_t group=size_t(-1);
  if(fixture){
   for(auto index:*segment){size_t label=input_points.at(index).get<2>();if(group==size_t(-1))group=label;else if(group!=label)throw std::runtime_error("Unexpected fixture plane merge");}
  }else{
   for(size_t i=0;i<expected.size();++i)if(canonical(p)==canonical(expected[i])){if(group!=size_t(-1))throw std::runtime_error("Ambiguous refined plane identity");group=i;}
  }
  if(group>=count||!seen.insert(group).second)throw std::runtime_error("Missing/duplicate refined plane identity");
  plane_groups[ptr]=group;plane_coeffs[group]=p;for(auto index:*segment)membership.emplace_back(group,index);
 }
 std::sort(membership.begin(),membership.end());
 std::ostringstream planes,members;planes<<std::setprecision(17)<<"id,nx,ny,nz,offset,faces\n";members<<"group,input_row\n";
 for(size_t i=0;i<count;++i){auto p=plane_coeffs[i];planes<<i<<','<<p[0]<<','<<p[1]<<','<<p[2]<<','<<p[3]<<','<<std::count_if(membership.begin(),membership.end(),[&](auto r){return r.first==i;})<<'\n';}
 for(auto row:membership)members<<row.first<<','<<row.second<<'\n';
 if(!fixture){
  std::istringstream in(contents(prior+"/owned-session/arrangement/prepared-members.csv"));std::string line;std::getline(in,line);std::vector<std::pair<size_t,size_t>> old;
  while(std::getline(in,line)){std::replace(line.begin(),line.end(),',',' ');std::istringstream row(line);size_t a,b;if(!(row>>a>>b))throw std::runtime_error("Expected membership record");old.emplace_back(a,b);}std::sort(old.begin(),old.end());
  if(old!=membership)throw std::runtime_error("Refined source membership changed");
 }
 exclusive(out+"/arrangement/prepared-planes.csv",planes.str());exclusive(out+"/arrangement/prepared-members.csv",members.str());
}

template<class M,class I,class G>Face snapshot(M& mesh,I index,G group,typename M::Face_index face){
 Face r;r.variable=index[face];r.descriptor=face.idx();auto found=plane_groups.find(group[face]);
 if(found==plane_groups.end())throw std::runtime_error("Candidate has no live observed plane");r.group=found->second;
 for(auto v:CGAL::vertices_around_face(mesh.halfedge(face),mesh)){r.vertices.push_back(mesh.point(v));r.planarity=std::max(r.planarity,std::sqrt(CGAL::to_double(CGAL::squared_distance(mesh.point(v),*group[face]))));}r.key=geometry_key(r.group,r.vertices);return r;
}
template<class M,class I,class G>void verify_snapshot(M& mesh,I index,G group){
 if(mesh.number_of_faces()!=records.size())throw std::runtime_error("Face count mutated");size_t iteration=0;
 for(auto f:mesh.faces()){
  auto r=snapshot(mesh,index,group,f);
  if(r.variable!=iteration||r.variable>=records.size())throw std::runtime_error("Face iteration/index ambiguity");
  auto& old=records[r.variable];if(old.descriptor!=r.descriptor||old.group!=r.group||old.key!=r.key)throw std::runtime_error("Live face geometry or plane changed");++iteration;
 }
}
template<class M,class I,class G>bool modly_classify(M& mesh,I index,G group){
 if(mesh.number_of_faces()>20000)throw std::runtime_error("Candidate cap");records.clear();total_queries=0;std::set<std::string> keys;
 for(auto f:mesh.faces()){
  auto r=snapshot(mesh,index,group,f);if(r.variable!=records.size()||!keys.insert(r.key).second)throw std::runtime_error("Candidate index/key ambiguity");records.push_back(r);
 }
 Source source;
 if(!fixture){
  source=load_source(prior);double lo[3]={1e100,1e100,1e100},hi[3]={-1e100,-1e100,-1e100};
  for(auto p:source.vertices)for(int k=0;k<3;++k){lo[k]=std::min(lo[k],p[k]);hi[k]=std::max(hi[k],p[k]);}
  double d=0;for(int k=0;k<3;++k)d+=(hi[k]-lo[k])*(hi[k]-lo[k]);if(std::abs(std::sqrt(d)-100)>1e-9)throw std::runtime_error("Native source diagonal differs");
  std::ifstream in(prior+"/finite-domain-input.bin",std::ios::binary);uint64_t np=0,nr=0;in.read((char*)&np,8);in.read((char*)&nr,8);
  if(!in||np!=plane_coeffs.size()||nr>10000000)throw std::runtime_error("Domain dimensions");read(in,domain_coeffs,np);std::vector<std::array<uint32_t,2>> rows;read(in,rows,nr);domain_rows.assign(np,{});
  for(auto row:rows){if(row[0]>=np||row[1]>=source.triangles.size())throw std::runtime_error("Domain index");domain_rows[row[0]].push_back(row[1]);}
  for(size_t i=0;i<np;++i){auto a=canonical(plane_coeffs[i]),b=canonical(domain_coeffs[i]);for(int k=0;k<4;++k)if(std::abs(a[k]-b[k])>1e-10)throw std::runtime_error("Domain metadata plane differs");}
 }
 for(size_t i=0;i<plane_coeffs.size();++i){
  Triangles domain;if(fixture)domain=fixture_domains.at(i);else{domain.reserve(domain_rows[i].size());for(auto row:domain_rows[i])domain.push_back(source.triangles[row]);}
  if(domain.empty())throw std::runtime_error("Empty authenticated domain");Tree tree(domain.begin(),domain.end());tree.accelerate_distance_queries();
  const K::Plane_3* pointer=nullptr;for(auto pair:plane_groups)if(pair.second==i)pointer=pair.first;
  if(!pointer)throw std::runtime_error("Missing live plane pointer");
  // Geometry-key order makes budget exhaustion independent of descriptor order.
  std::vector<size_t> work;for(size_t j=0;j<records.size();++j)if(records[j].group==i)work.push_back(j);
  std::sort(work.begin(),work.end(),[](size_t a,size_t b){return records[a].key<records[b].key;});
  for(auto j:work)records[j].proof=polygon(tree,records[j].vertices,*pointer);
 }
 verify_snapshot(mesh,index,group);
 std::ostringstream log;log<<std::setprecision(17);size_t eligible=0,unsupported=0,unresolved=0;
 for(auto r:records){eligible+=r.proof.state==CERTIFIED_ELIGIBLE;unsupported+=r.proof.state==CONFIRMED_UNSUPPORTED;unresolved+=r.proof.state==UNRESOLVED;
  log<<"{\"variable\":"<<r.variable<<",\"face_descriptor\":"<<r.descriptor<<",\"group\":"<<r.group<<",\"key\":\""<<r.key<<"\",\"state\":\""<<name(r.proof.state)<<"\",\"reason\":\""<<r.proof.reason<<"\",\"queries\":"<<r.proof.queries<<",\"depth\":"<<r.proof.depth<<",\"max_planarity_residual_native\":"<<r.planarity<<",\"witness_distance_native\":"<<r.proof.distance<<",\"witness\":["<<r.proof.witness.x()<<','<<r.proof.witness.y()<<','<<r.proof.witness.z()<<"],\"plane\":[";
  auto p=plane_coeffs[r.group];for(int k=0;k<4;++k){if(k)log<<',';log<<p[k];}log<<"],\"vertices\":[";
  for(size_t k=0;k<r.vertices.size();++k){if(k)log<<',';auto p=r.vertices[k];log<<'['<<p.x()<<','<<p.y()<<','<<p.z()<<']';}log<<"]}\n";
 }
 exclusive(out+"/eligibility.jsonl",log.str());
 std::ostringstream summary;summary<<"{\"candidate_faces\":"<<records.size()<<",\"eligibility_records\":"<<records.size()<<",\"eligible_faces\":"<<eligible<<",\"confirmed_unsupported\":"<<unsupported<<",\"unresolved\":"<<unresolved<<",\"queries\":"<<total_queries<<",\"exact_live_index_mapping\":true,\"full_source_domains_consume_both_heldout_areas\":"<<(fixture?"false":"true")<<"}\n";
 exclusive(out+"/eligibility-summary.json",summary.str());insufficient=eligible<4;return fixture||!insufficient;
}
template<class M,class I,class G,class S,class Vars>void modly_veto(M& mesh,I index,G group,S& solver,const Vars& variables){
 verify_snapshot(mesh,index,group);expected_variables=variables.size();if(expected_variables<records.size())throw std::runtime_error("Variable count");
 size_t count=0;for(auto r:records)if(r.proof.state!=CERTIFIED_ELIGIBLE){auto c=solver.create_constraint(0.,0.);c->add_coefficient(variables[r.variable],1.);++count;}
 if(fixture&&require_nonempty){auto c=solver.create_constraint(1.,1e20);for(size_t i=0;i<records.size();++i)c->add_coefficient(variables[i],1.);}
 exclusive(out+"/veto-constraints.json","{\"face_variables_fixed_zero\":"+std::to_string(count)+",\"total_variables\":"+std::to_string(expected_variables)+",\"original_objective_unchanged\":true,\"fixture_only_positive_selection\":"+(fixture&&require_nonempty?"true":"false")+"}\n");
}
void modly_before_solve(){
 if(!fixture){
  exclusive(base+"/ELIGIBILITY-READY.json",contents(out+"/eligibility-summary.json"));
  std::string token;std::getline(std::cin,token);if(token!="GO"||std::cin.peek()!=EOF)throw std::runtime_error("Exact owned GO and EOF required");
  exclusive(base+"/AI-MIP-started.txt","ONE_CONSTRAINED_AI_MIP\n");
 }
}
template<class M,class I,class G>bool modly_selection(M& mesh,I index,G group,const std::vector<double>& x,size_t total){
 verify_snapshot(mesh,index,group);if(x.size()!=total||x.size()!=expected_variables)throw std::runtime_error("Incomplete solution");
 for(auto value:x)if(!std::isfinite(value)||(value!=0&&value!=1))throw std::runtime_error("Nonbinary solution");
 selected_faces=0;std::ostringstream s;s<<"{\"values\":[";for(size_t i=0;i<x.size();++i){if(i)s<<',';s<<x[i];}s<<"],\"selected_faces\":[";
 bool first=true;for(auto r:records){if(r.proof.state!=CERTIFIED_ELIGIBLE&&x[r.variable]!=0)throw std::runtime_error("Excluded face selected");if(x[r.variable]==1){if(!first)s<<',';s<<r.variable;first=false;++selected_faces;}}
 s<<"],\"selected_face_count\":"<<selected_faces<<",\"all_exclusions_exactly_zero\":true,\"mapping_reverified_before_deletion\":true}\n";exclusive(out+"/selection.json",s.str());
 empty_optimum=selected_faces==0;return !empty_optimum;
}

void reset(const std::string& folder){out=folder;std::filesystem::create_directories(out+"/arrangement");records.clear();plane_groups.clear();plane_coeffs.clear();empty_optimum=false;insufficient=false;selected_faces=0;expected_variables=0;total_queries=0;}
void populate_fixture(bool shrink){
 std::vector<P> v={P(0,0,0),P(20,0,0),P(0,20,0),P(0,0,20)};
 std::vector<std::array<int,3>> faces={{{0,2,1}},{{0,1,3}},{{0,3,2}},{{1,2,3}}};fixture_domains.clear();input_points.clear();
 for(size_t g=0;g<faces.size();++g){auto f=faces[g];T t(v[f[0]],v[f[1]],v[f[2]]);
  if(shrink&&g==0)t=T(t[0],CGAL::midpoint(t[0],t[1]),CGAL::midpoint(t[0],t[2]));fixture_domains.push_back({t});
  V n=CGAL::cross_product(t[1]-t[0],t[2]-t[0]);n=n/std::sqrt(n.squared_length());
  for(int k=0;k<3;++k)input_points.emplace_back(t[k],n,int(g));
  for(int i=1;i<=16;++i)for(int j=1;j<=16;++j){double a=i/18.,b=j/18.*(1-a);P p(t[0].x()*(1-a-b)+t[1].x()*a+t[2].x()*b,t[0].y()*(1-a-b)+t[1].y()*a+t[2].y()*b,t[0].z()*(1-a-b)+t[1].z()*a+t[2].z()*b);input_points.emplace_back(p,n,int(g));}
 }
}
int fixtures(){
 fixture=true;std::filesystem::create_directory(base+"/fixtures");
 bool all=true,successful_depth=false;std::ostringstream cases;cases<<'[';
 for(int test=0;test<3;++test){if(test)cases<<',';reset(base+"/fixtures/case-"+std::to_string(test));require_nonempty=test==2;populate_fixture(test!=0);
  CGAL::Polygonal_surface_reconstruction<K> algo(input_points,PM(),NM(),IM());Mesh model;bool ok;
  std::ostringstream captured;auto original=std::cout.rdbuf(captured.rdbuf());
  try{ok=algo.reconstruct<Solver>(model);}catch(...){std::cout.rdbuf(original);throw;}std::cout.rdbuf(original);
  exclusive(out+"/solver-stdout.txt",captured.str());
  bool infeasible=captured.str().find("MODLY_SCIP_STATUS optimal=0 code="+std::to_string(int(SCIP_STATUS_INFEASIBLE))+" ")!=std::string::npos;
  bool passed=test==0?(ok&&selected_faces==4&&model.number_of_faces()==4):test==1?(!ok&&empty_optimum&&selected_faces==0):(!ok&&!empty_optimum&&infeasible);
  if(test==0)for(auto r:records)successful_depth=successful_depth||(r.proof.state==CERTIFIED_ELIGIBLE&&r.proof.depth>0);
  all=all&&passed;cases<<"{\"case\":"<<test<<",\"PASS\":"<<(passed?"true":"false")<<",\"selected_faces\":"<<selected_faces<<",\"empty_optimum\":"<<(empty_optimum?"true":"false")<<",\"infeasible_status_verified\":"<<(infeasible?"true":"false")<<",\"fixture_only_positive_selection\":"<<(require_nonempty?"true":"false")<<'}';
 }
 // Pure classifier limit and index-spoof tests do not execute another solver.
 Triangles t={T(P(0,0,0),P(20,0,0),P(0,20,0))};Tree tree(t.begin(),t.end());total_queries=QUERY_LIMIT;
 auto exhausted=cell(tree,t[0],0);all=all&&exhausted.state==UNRESOLVED;
 auto warped=polygon(tree,{P(0,0,0),P(20,0,0),P(0,20,2e-10)},K::Plane_3(0,0,1,0));
 bool warp_rejected=warped.state==UNRESOLVED&&warped.reason=="NONPLANAR_CANDIDATE";all=all&&warp_rejected&&successful_depth;
 Mesh bad;auto a=bad.add_vertex(P(0,0,0)),b=bad.add_vertex(P(20,0,0)),c=bad.add_vertex(P(0,20,0));auto d=bad.add_vertex(P(30,0,0)),e=bad.add_vertex(P(50,0,0)),f=bad.add_vertex(P(30,20,0));
 bad.add_face(a,b,c);bad.add_face(d,e,f);auto indices=bad.add_property_map<Mesh::Face_index,size_t>("f:index",0).first;
 K::Plane_3 plane(P(0,0,0),P(20,0,0),P(0,20,0));plane_groups.clear();plane_groups[&plane]=0;
 auto groups=bad.add_property_map<Mesh::Face_index,const K::Plane_3*>("f:supp_plane",&plane).first;bool duplicate=false;
 try{modly_classify(bad,indices,groups);}catch(const std::runtime_error& error){duplicate=std::string(error.what())=="Candidate index/key ambiguity";}
 all=all&&duplicate;
 records.clear();size_t index=0;for(auto face:bad.faces()){indices[face]=index++;records.push_back(snapshot(bad,indices,groups,face));}expected_variables=2;bool spoof=false;
 try{modly_selection(bad,indices,groups,{1.,0.},2);}catch(const std::runtime_error& error){spoof=std::string(error.what())=="Excluded face selected";}all=all&&spoof;
 cases<<']';exclusive(base+"/fixtures/result.json",std::string("{\"classification\":\"")+(all?"PASS":"FAIL")+"\",\"cases\":"+cases.str()+",\"zero_budget_is_unresolved\":"+(exhausted.state==UNRESOLVED?"true":"false")+",\"duplicate_index_rejected\":"+(duplicate?"true":"false")+",\"spoofed_excluded_selection_rejected\":"+(spoof?"true":"false")+",\"nonplanar_polygon_rejected\":"+(warp_rejected?"true":"false")+",\"successful_subdivision_depth_recorded\":"+(successful_depth?"true":"false")+"}\n");return all?0:11;
}
int main(int argc,char** argv){
 try{
  if(argc!=5)return 2;std::string mode=argv[1];base=argv[2];prior=argv[3]; // Last arg is a fixed protocol token.
  if(std::string(argv[4])!="finite-patch-v1")return 3;
  if(mode=="fixtures")return fixtures();if(mode!="reconstruct")return 4;
  fixture=false;reset(base+"/owned-session");std::filesystem::create_directory(out+"/candidate");
  std::ifstream input(prior+"/augmented/train.ply");
  if(!CGAL::IO::read_PLY_with_properties(input,std::back_inserter(input_points),CGAL::IO::make_ply_point_reader(PM()),CGAL::IO::make_ply_normal_reader(NM()),std::make_pair(IM(),CGAL::IO::PLY_property<int>("segment_index")))||input_points.empty()||input_points.size()>120000)return 5;
  CGAL::Polygonal_surface_reconstruction<K> algo(input_points,PM(),NM(),IM());Mesh model;bool ok=algo.reconstruct<Solver>(model);
  if(!ok){exclusive(base+"/native-result.json",std::string("{\"classification\":\"")+(empty_optimum?"NO_GO_EMPTY_OPTIMUM":insufficient?"NO_GO_INSUFFICIENT_ELIGIBLE":"NO_GO_OPTIMISATION")+"\",\"candidate_created\":false}\n");return 0;}
  std::ostringstream mesh;mesh<<std::setprecision(17);CGAL::IO::write_OFF(mesh,model);exclusive(out+"/candidate/raw-reconstruction.off",mesh.str());
  exclusive(base+"/native-result.json","{\"classification\":\"CANDIDATE_REQUIRES_FULL_VALIDATION\",\"vertices\":"+std::to_string(model.number_of_vertices())+",\"polygons\":"+std::to_string(model.number_of_faces())+",\"candidate_created\":true}\n");return 0;
 }catch(const std::exception& error){std::cerr<<error.what()<<std::endl;return 10;}
}
