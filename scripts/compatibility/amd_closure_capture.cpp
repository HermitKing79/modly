// Phase16: actual fans/witnesses; optional one optimistic diagnostic; NO export.
#include <vector>
template<class M,class I,class G,class A>bool phase16_capture(M&,I,G,const A&);
template<class M,class I,class G,class S,class Vars>void phase16_veto(M&,I,G,S&,const Vars&);
template<class M,class I,class G>bool phase16_selection(M&,I,G,const std::vector<double>&,size_t);
void phase16_before_solve();
#include "phase15-common.inc"
static bool mapping_only=false;
static bool diagnostic=false;

template<class M,class I,class G,class A>bool phase16_capture(M& mesh,I index,G group,const A& adjacency){
 records.clear();std::set<std::string> keys;
 for(auto face:mesh.faces()){
  auto r=snapshot(mesh,index,group,face);
  if(r.variable!=records.size()||!keys.insert(r.key).second)throw std::runtime_error("Live face mapping ambiguous");
  records.push_back(r);
 }
 if(records.size()!=1105)throw std::runtime_error("Historical arrangement count differs");
 std::ostringstream geometry,rawfans;geometry<<std::setprecision(17);
 for(auto r:records){geometry<<"{\"variable\":"<<r.variable<<",\"descriptor\":"<<r.descriptor<<",\"group\":"<<r.group<<",\"key\":\""<<r.key<<"\",\"vertices\":[";
  for(size_t k=0;k<r.vertices.size();++k){if(k)geometry<<',';auto p=r.vertices[k];geometry<<'['<<p.x()<<','<<p.y()<<','<<p.z()<<']';}geometry<<"]}\n";}
 size_t rawid=0;std::vector<size_t> rawoccurrences(records.size(),0);
 for(const auto& fan:adjacency){rawfans<<"{\"fan\":"<<rawid++<<",\"faces\":[";std::set<size_t> unique;
  for(size_t j=0;j<fan.size();++j){auto f=mesh.face(fan[j]);if(f==M::null_face())throw std::runtime_error("Null fan face");size_t id=index[f];if(id>=records.size()||!unique.insert(id).second)throw std::runtime_error("Fan ambiguity");if(j)rawfans<<',';rawfans<<id;++rawoccurrences[id];}rawfans<<"]}\n";}
 for(size_t i=0;i<records.size();++i)if(rawoccurrences[i]!=records[i].vertices.size())throw std::runtime_error("Incomplete raw edge coverage");
 exclusive(base+"/actual-polygons.jsonl",geometry.str());exclusive(base+"/actual-fans.jsonl",rawfans.str());
 if(mapping_only){exclusive(base+"/native-result.json","{\"classification\":\"IDENTITY_FORENSICS_ONLY\",\"solver_calls\":0,\"labels_transferred\":false,\"candidate_exported\":false}\n");return false;}
 struct Prior {size_t variable,group;int state;P witness;double distance;std::vector<P> vertices;};
 std::map<std::string,Prior> old;std::set<size_t> ids;
 std::ifstream in(base+(diagnostic?"/prior-polygons.txt":"/prior-faces.txt"));size_t n=0;if(!(in>>n)||n!=records.size())throw std::runtime_error("Prior count");
 for(size_t i=0;i<n;++i){Prior p;std::string key;double x,y,z;
  if(!(in>>p.variable>>p.group>>p.state>>x>>y>>z>>p.distance>>key)||p.variable>=n||!ids.insert(p.variable).second||p.group>=plane_coeffs.size()||p.state<0||p.state>2||!std::isfinite(x+y+z+p.distance))throw std::runtime_error("Prior identity record");
  p.witness=P(x,y,z);
  if(diagnostic){size_t nv=0;if(!(in>>nv)||nv<3||nv>1000)throw std::runtime_error("Prior vertex count");for(size_t j=0;j<nv;++j){if(!(in>>x>>y>>z)||!std::isfinite(x+y+z))throw std::runtime_error("Prior vertex");p.vertices.emplace_back(x,y,z);}}
  if(!old.emplace(key,p).second)throw std::runtime_error("Prior key duplicate");
 }
 std::string extra;if(in>>extra)throw std::runtime_error("Trailing prior input");
 std::map<size_t,Prior> matched;std::set<size_t> used_prior;std::ostringstream correspondence;correspondence<<std::setprecision(17);
 for(auto& r:records){const Prior* found=nullptr;double residual=0.;
  if(!diagnostic){auto it=old.find(r.key);if(it!=old.end())found=&it->second;}
  else for(auto& pair:old){auto& p=pair.second;if(p.group!=r.group||p.vertices.size()!=r.vertices.size())continue;
   double best=1e100;size_t nv=p.vertices.size();
   for(size_t start=0;start<nv;++start)for(int sign:{-1,1}){double worst=0;for(size_t j=0;j<nv;++j)worst=std::max(worst,std::sqrt(CGAL::to_double(CGAL::squared_distance(r.vertices[j],p.vertices[(start+nv+sign*int(j))%nv]))));best=std::min(best,worst);}
   if(best<=1e-10){if(found)throw std::runtime_error("Ambiguous cyclic polygon correspondence");found=&p;residual=best;}
  }
  if(!found||found->group!=r.group||!used_prior.insert(found->variable).second)throw std::runtime_error("Live polygon identity differs");
  auto p=plane_coeffs[r.group];K::Plane_3 plane(p[0],p[1],p[2],-p[3]);CGAL::Polygon_2<K> projected;
  for(auto v:r.vertices){if(CGAL::squared_distance(v,plane)>1e-20)throw std::runtime_error("Live polygon nonplanar");projected.push_back(plane.to_2d(v));}
  if(!projected.is_simple()||!projected.is_convex()||projected.area()==0)throw std::runtime_error("Live polygon not simple convex");
  matched[r.variable]=*found;r.proof.state=State(found->state);r.proof.witness=found->witness;r.proof.distance=found->distance;
  correspondence<<"{\"variable\":"<<r.variable<<",\"prior_variable\":"<<found->variable<<",\"residual_native\":"<<residual<<",\"eligible_certificate_transferred\":false}\n";
 }
 exclusive(base+"/live-correspondence.jsonl",correspondence.str());
 // Reuse authenticated COMPLETE compatible source domains, not training points.
 Source source=load_source(prior);std::ifstream domains(prior+"/finite-domain-input.bin",std::ios::binary);
 uint64_t np=0,nr=0;domains.read((char*)&np,8);domains.read((char*)&nr,8);
 if(!domains||np!=plane_coeffs.size()||nr>10000000)throw std::runtime_error("Domain dimensions");
 read(domains,domain_coeffs,np);std::vector<std::array<uint32_t,2>> rows;read(domains,rows,nr);domain_rows.assign(np,{});
 for(auto row:rows){if(row[0]>=np||row[1]>=source.triangles.size())throw std::runtime_error("Domain index");domain_rows[row[0]].push_back(row[1]);}
 for(size_t i=0;i<np;++i){auto a=canonical(plane_coeffs[i]),b=canonical(domain_coeffs[i]);for(int k=0;k<4;++k)if(std::abs(a[k]-b[k])>1e-10)throw std::runtime_error("Domain plane changed");}
 std::ostringstream witness;witness<<std::setprecision(17);size_t verified=0;
 for(size_t g=0;g<np;++g){
  Triangles domain;domain.reserve(domain_rows[g].size());for(auto row:domain_rows[g])domain.push_back(source.triangles[row]);
  if(domain.empty())throw std::runtime_error("Empty domain");Tree tree(domain.begin(),domain.end());tree.accelerate_distance_queries();
  for(auto& r:records)if(r.group==g&&r.proof.state==CONFIRMED_UNSUPPORTED){
   Triangles polygon;for(size_t k=1;k+1<r.vertices.size();++k)if(!CGAL::collinear(r.vertices[0],r.vertices[k],r.vertices[k+1]))polygon.emplace_back(r.vertices[0],r.vertices[k],r.vertices[k+1]);
   if(polygon.empty())throw std::runtime_error("Degenerate witness polygon");Tree face(polygon.begin(),polygon.end());
   double displacement=distance(face,r.proof.witness),d=distance(tree,r.proof.witness);
   // A true point on this face exists within displacement. Lipschitz lower bound
   // retains the full original margin, even for roundoff in saved centroids.
   if(!std::isfinite(displacement+d)||displacement>1e-10||std::abs(d-r.proof.distance)>1e-8||d-displacement<=EPS+MARGIN)throw std::runtime_error("Unsupported witness not reproduced");
   witness<<"{\"variable\":"<<r.variable<<",\"prior_variable\":"<<matched.at(r.variable).variable<<",\"group\":"<<g<<",\"distance_native\":"<<d<<",\"footprint_displacement_native\":"<<displacement<<",\"unsupported_lower_bound_native\":"<<d-displacement<<",\"complete_domain_triangles\":"<<domain.size()<<",\"PASS\":true}\n";++verified;
  }
 }
 if(verified!=719)throw std::runtime_error("Unsupported witness count differs");
 exclusive(base+"/witness-verification.jsonl",witness.str());verify_snapshot(mesh,index,group);
 std::ostringstream live,fans;live<<std::setprecision(17);
 for(auto r:records)live<<"{\"variable\":"<<r.variable<<",\"prior_variable\":"<<matched.at(r.variable).variable<<",\"group\":"<<r.group<<",\"key\":\""<<r.key<<"\",\"state\":\""<<(r.proof.state==CONFIRMED_UNSUPPORTED?"CONFIRMED_UNSUPPORTED":"POTENTIALLY_ELIGIBLE")<<"\",\"prior_state\":\""<<name(r.proof.state)<<"\"}\n";
 size_t fan_id=0;std::vector<size_t> occurrences(records.size(),0);
 for(const auto& fan:adjacency){fans<<"{\"fan\":"<<fan_id++<<",\"faces\":[";std::set<size_t> unique;
  for(size_t j=0;j<fan.size();++j){auto f=mesh.face(fan[j]);if(f==M::null_face())throw std::runtime_error("Null fan face");size_t id=index[f];
   if(id>=records.size()||!unique.insert(id).second)throw std::runtime_error("Fan identity ambiguity");if(j)fans<<',';fans<<id;++occurrences[id];}
  fans<<"],\"halfedge_descriptors\":[";for(size_t j=0;j<fan.size();++j){if(j)fans<<',';fans<<fan[j].idx();}fans<<"]}\n";
 }
 for(size_t i=0;i<records.size();++i)if(occurrences[i]!=records[i].vertices.size())throw std::runtime_error("Incomplete polygon edge coverage");
 exclusive(base+"/live-faces.jsonl",live.str());exclusive(base+"/live-fans.jsonl",fans.str());
 exclusive(base+"/capture-result.json","{\"classification\":\"AUTHENTICATED_ADJACENCY_CAPTURE\",\"faces\":1105,\"unsupported_witnesses_verified\":719,\"all_face_edges_accounted\":true,\"candidate_exported\":false}\n");
 if(!diagnostic)exclusive(base+"/native-result.json",contents(base+"/capture-result.json"));
 return diagnostic;
}
template<class M,class I,class G,class S,class Vars>void phase16_veto(M& mesh,I index,G group,S& solver,const Vars& variables){
 if(!diagnostic)throw std::runtime_error("Only diagnostic mode can allocate MIP");verify_snapshot(mesh,index,group);expected_variables=variables.size();
 if(expected_variables<records.size())throw std::runtime_error("Variable count");size_t vetoes=0;
 for(auto r:records)if(r.proof.state==CONFIRMED_UNSUPPORTED){auto c=solver.create_constraint(0.,0.);c->add_coefficient(variables[r.variable],1.);++vetoes;}
 if(vetoes!=719)throw std::runtime_error("Unsupported veto count");auto nonempty=solver.create_constraint(1.,1e20);for(size_t i=0;i<records.size();++i)nonempty->add_coefficient(variables[i],1.);
 exclusive(base+"/diagnostic-constraints.json","{\"unsupported_vetoes\":719,\"optimistically_allowed_faces\":386,\"nonempty_constraint\":true,\"original_objective_and_constraints_unchanged\":true,\"total_variables\":"+std::to_string(expected_variables)+"}\n");
}
void phase16_before_solve(){
 if(!diagnostic)throw std::runtime_error("Diagnostic mode required");exclusive(base+"/DIAGNOSTIC-READY.json",contents(base+"/capture-result.json"));
 std::string token;std::getline(std::cin,token);
 if(token=="STOP"&&std::cin.peek()==EOF)throw std::runtime_error("HOST_GRAPH_PROVED_EMPTY");
 if(token!="DIAGNOSTIC_GO"||std::cin.peek()!=EOF)throw std::runtime_error("Exact diagnostic GO and EOF required");
 exclusive(base+"/DIAGNOSTIC-MIP-started.txt","ONE_OPTIMISTIC_NONEMPTY_DIAGNOSTIC_ONLY\n");
}
template<class M,class I,class G>bool phase16_selection(M& mesh,I index,G group,const std::vector<double>& x,size_t total){
 verify_snapshot(mesh,index,group);if(x.size()!=total||total!=expected_variables)throw std::runtime_error("Incomplete diagnostic solution");
 for(auto v:x)if(!std::isfinite(v)||(v!=0&&v!=1))throw std::runtime_error("Nonbinary diagnostic solution");
 selected_faces=0;std::ostringstream s;s<<"{\"values\":[";for(size_t i=0;i<x.size();++i){if(i)s<<',';s<<x[i];}s<<"],\"selected_faces\":[";
 for(auto r:records){if(r.proof.state==CONFIRMED_UNSUPPORTED&&x[r.variable]!=0)throw std::runtime_error("Unsupported diagnostic face selected");if(x[r.variable]){if(selected_faces)s<<',';s<<r.variable;++selected_faces;}}
 if(!selected_faces)throw std::runtime_error("Diagnostic nonempty requirement violated");s<<"],\"candidate_exported\":false,\"not_a_supported_asset\":true}\n";exclusive(base+"/diagnostic-selection.json",s.str());return false;
}
int main(int argc,char** argv){try{
 if(argc!=4)return 2;std::string token=argv[3];if(token!="closure-capture-v1"&&token!="closure-mapping-v1"&&token!="closure-diagnostic-v1")return 2;mapping_only=token=="closure-mapping-v1";diagnostic=token=="closure-diagnostic-v1";base=argv[1];prior=argv[2];fixture=false;reset(base+"/owned-session");
 std::ifstream input(prior+"/augmented/train.ply");
 if(!CGAL::IO::read_PLY_with_properties(input,std::back_inserter(input_points),CGAL::IO::make_ply_point_reader(PM()),CGAL::IO::make_ply_normal_reader(NM()),std::make_pair(IM(),CGAL::IO::PLY_property<int>("segment_index")))||input_points.empty()||input_points.size()>120000)return 3;
 CGAL::Polygonal_surface_reconstruction<K> algo(input_points,PM(),NM(),IM());Mesh ignored;
 if(diagnostic){std::ostringstream captured;auto saved=std::cout.rdbuf(captured.rdbuf());
  try{if(algo.reconstruct<Solver>(ignored))throw std::runtime_error("Diagnostic unexpectedly exported");}catch(...){std::cout.rdbuf(saved);std::cout<<captured.str();exclusive(base+"/solver-stdout.txt",captured.str());throw;}
  std::cout.rdbuf(saved);std::cout<<captured.str();exclusive(base+"/solver-stdout.txt",captured.str());
  bool infeasible=captured.str().find("MODLY_SCIP_STATUS optimal=0 code="+std::to_string(int(SCIP_STATUS_INFEASIBLE))+" ")!=std::string::npos;
  bool optimal=captured.str().find("MODLY_SCIP_STATUS optimal=1 code="+std::to_string(int(SCIP_STATUS_OPTIMAL))+" ")!=std::string::npos;
  std::string classification=infeasible?"PROVEN_OPTIMISTIC_ARRANGEMENT_INFEASIBLE":optimal&&selected_faces?"OPTIMISTIC_NONEMPTY_SELECTION_EXISTS":"INCONCLUSIVE_SOLVER_STATUS";
  exclusive(base+"/native-result.json","{\"classification\":\""+classification+"\",\"diagnostic_solves\":1,\"production_solves\":0,\"candidate_exported\":false,\"selected_faces\":"+std::to_string(selected_faces)+"}\n");return 0;
 }
 if(algo.reconstruct<Solver>(ignored))throw std::runtime_error("Capture unexpectedly reconstructed");
 if(!std::filesystem::exists(base+"/native-result.json"))throw std::runtime_error("Capture did not complete");return 0;
 }catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 10;}}
