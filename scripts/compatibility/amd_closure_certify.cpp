// Classify every frozen live-arrangement polygon; no reconstruction or MIP code.
#define main unused_previous_validation_main
#include "amd_mesh_aware_validate.cpp"
#undef main
#include <CGAL/Polygon_2.h>
#include <map>
#include <set>
#include <sstream>
#include <chrono>
static size_t terminal_cells=0;
static double terminal_radius=0,terminal_slack=1e100;
#include "certificate-core.inc"

int main(int argc,char** argv){try{
 if(argc!=4)return 2;bool refine=std::string(argv[3])=="closure-refine-v1";if(!refine&&std::string(argv[3])!="closure-certify-v1")return 2;base=argv[1];prior=argv[2];
 std::ifstream in(base+"/frozen-polygons.txt");size_t nf=0,np=0;if(!(in>>nf>>np)||nf!=1105||np!=27)throw std::runtime_error("Frozen dimensions");
 for(size_t i=0;i<np;++i){size_t id;std::array<double,4> p;if(!(in>>id>>p[0]>>p[1]>>p[2]>>p[3])||id!=i)throw std::runtime_error("Frozen plane");plane_coeffs.push_back(p);}
 std::set<std::string> keys;
 for(size_t i=0;i<nf;++i){Face r;size_t nv;if(!(in>>r.variable>>r.group>>r.key>>nv)||r.variable!=i||r.group>=np||nv<3||nv>1000||!keys.insert(r.key).second)throw std::runtime_error("Frozen face identity");
  for(size_t j=0;j<nv;++j){double x,y,z;if(!(in>>x>>y>>z)||!std::isfinite(x+y+z))throw std::runtime_error("Frozen vertex");r.vertices.emplace_back(x,y,z);}
  if(geometry_key(r.group,r.vertices)!=r.key)throw std::runtime_error("Frozen exact geometry key differs");records.push_back(r);
 }
 std::string extra;if(in>>extra)throw std::runtime_error("Trailing frozen input");
 std::set<size_t> selected;
 if(refine){std::ifstream targets(base+"/refine-ids.txt");size_t count,id;std::string key;
  if(!(targets>>count)||count!=2||DEPTH_LIMIT!=28)throw std::runtime_error("Targeted refinement contract");
  for(size_t j=0;j<count;++j)if(!(targets>>id>>key)||id>=nf||key!=records[id].key||!selected.insert(id).second)throw std::runtime_error("Targeted face identity");
  if(targets>>extra)throw std::runtime_error("Trailing targets");
 }
 Source source=load_source(prior);std::ifstream domains(prior+"/finite-domain-input.bin",std::ios::binary);uint64_t dp=0,nr=0;domains.read((char*)&dp,8);domains.read((char*)&nr,8);
 if(!domains||dp!=np||nr>10000000)throw std::runtime_error("Domain dimensions");read(domains,domain_coeffs,np);std::vector<std::array<uint32_t,2>> rows;read(domains,rows,nr);domain_rows.assign(np,{});
 for(auto row:rows){if(row[0]>=np||row[1]>=source.triangles.size())throw std::runtime_error("Domain index");domain_rows[row[0]].push_back(row[1]);}
 for(size_t i=0;i<np;++i){auto a=canonical(plane_coeffs[i]),b=canonical(domain_coeffs[i]);for(int k=0;k<4;++k)if(!std::isfinite(a[k]+b[k])||std::abs(a[k]-b[k])>1e-10)throw std::runtime_error("Domain plane differs");}
 if(std::filesystem::exists(base+"/certificates.jsonl")||std::filesystem::exists(base+"/progress.jsonl"))throw std::runtime_error("Consumed certificate evidence");
 std::ofstream log(base+"/certificates.jsonl"),progress(base+"/progress.jsonl");log<<std::setprecision(17);progress<<std::setprecision(17);
 size_t eligible=0,unsupported=0,unresolved=0,resource_incomplete=0,done=0;auto start=std::chrono::steady_clock::now();
 for(size_t g=0;g<np;++g){Triangles domain;for(auto row:domain_rows[g])domain.push_back(source.triangles[row]);if(domain.empty())throw std::runtime_error("Empty domain");Tree tree(domain.begin(),domain.end());tree.accelerate_distance_queries();
  auto p=plane_coeffs[g];K::Plane_3 plane(p[0],p[1],p[2],-p[3]);std::vector<size_t> work;for(size_t i=0;i<nf;++i)if(records[i].group==g)work.push_back(i);
  std::sort(work.begin(),work.end(),[](size_t a,size_t b){return records[a].key<records[b].key;});
  for(auto i:work){if(refine&&!selected.count(i))continue;auto& r=records[i];bool unexamined=total_queries>=QUERY_LIMIT;terminal_cells=0;terminal_radius=0;terminal_slack=1e100;r.proof=polygon(tree,r.vertices,plane);auto q=r.proof;
   eligible+=q.state==CERTIFIED_ELIGIBLE;unsupported+=q.state==CONFIRMED_UNSUPPORTED;unresolved+=q.state==UNRESOLVED;
   std::string coverage=q.state!=UNRESOLVED?"CONCLUSIVE":q.reason=="QUERY_LIMIT"?(unexamined?"UNEXAMINED_RESOURCE_EXHAUSTION":"PARTIALLY_EXAMINED_RESOURCE_EXHAUSTION"):"EXAMINED_NUMERICAL_DEPTH_UNRESOLVED";
   resource_incomplete+=q.reason=="QUERY_LIMIT";
   log<<"{\"variable\":"<<i<<",\"group\":"<<g<<",\"key\":\""<<r.key<<"\",\"state\":\""<<name(q.state)<<"\",\"reason\":\""<<q.reason<<"\",\"coverage\":\""<<coverage<<"\",\"queries\":"<<q.queries<<",\"depth\":"<<q.depth<<",\"witness_distance_native\":"<<q.distance<<",\"terminal_depth_cells\":"<<terminal_cells<<",\"terminal_max_radius_native\":"<<terminal_radius<<",\"terminal_min_slack_native\":"<<(terminal_cells?terminal_slack:0)<<",\"witness\":["<<q.witness.x()<<','<<q.witness.y()<<','<<q.witness.z()<<"]}\n";++done;
  }
  log.flush();progress<<"{\"group\":"<<g<<",\"faces_completed\":"<<done<<",\"queries\":"<<total_queries<<",\"elapsed_seconds\":"<<std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()<<"}\n";progress.flush();if(!log||!progress)throw std::runtime_error("Evidence write failure");
 }
 if(done!=(refine?2:nf))throw std::runtime_error("Incomplete classification");
 exclusive(base+"/result.json","{\"classification\":\""+std::string(resource_incomplete?"INCOMPLETE_RESOURCE_LIMIT":refine?"ALL_TARGETED_FACES_EXAMINED":"ALL_FACES_EXAMINED")+"\",\"faces\":"+std::to_string(done)+",\"eligible\":"+std::to_string(eligible)+",\"unsupported\":"+std::to_string(unsupported)+",\"unresolved\":"+std::to_string(unresolved)+",\"resource_incomplete\":"+std::to_string(resource_incomplete)+",\"queries\":"+std::to_string(total_queries)+",\"query_limit\":"+std::to_string(QUERY_LIMIT)+",\"historical_and_locked_source_geometry_consumed\":true,\"solver_calls\":0}\n");return 0;
 }catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 10;}}
