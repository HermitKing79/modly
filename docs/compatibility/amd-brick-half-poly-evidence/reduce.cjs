// One fixed installed CPU/WASM simplifier call. No weld, prune or retry.
'use strict';
const fs=require('fs'),path=require('path'),crypto=require('crypto');
const root=path.resolve(__dirname,'../../..');
const P=path.resolve(root,'../artifacts/amd-brick-half-poly-5705f9d-corrected');
const out=path.join(P,'reduce');
const sha=b=>crypto.createHash('sha256').update(b).digest('hex');
function need(x,msg){if(!x)throw Error(msg);}
function plain(p){
 need(path.isAbsolute(p)&&path.resolve(p)===p,'Canonical absolute path');
 for(let q=p;;q=path.dirname(q)){const s=fs.lstatSync(q);need(!s.isSymbolicLink(),'No symlinks');if(s.isFile())need(s.nlink===1,'Single-link file');if(path.dirname(q)===q)break;}
 return p;
}
function read(p,expected){plain(p);const b=fs.readFileSync(p);need(sha(b)===expected,'Authenticated file changed: '+p);return b;}
function write(name,data){need(path.basename(name)===name,'Owned output basename');const fd=fs.openSync(path.join(out,name),'wx');try{fs.writeFileSync(fd,data);fs.fsyncSync(fd);}finally{fs.closeSync(fd);}}
function json(name,value){write(name,JSON.stringify(value,null,2)+'\n');}
function npy(p,pin,descr,rows){
 const b=read(p,pin);need(b.subarray(0,8).equals(Buffer.from([147,78,85,77,80,89,1,0])),'Original NPY1.0');
 const n=b.readUInt16LE(8),offset=10+n;need(n<4096&&offset%4===0,'Bounded aligned NPY header');
 const header=b.toString('latin1',10,offset);
 need(header.includes("'descr': '"+descr+"'")&&header.includes("'fortran_order': False")&&new RegExp("'shape': \\("+rows+", 3\\)").test(header),'Exact original NPY schema');
 need(b.length===offset+rows*12,'Exact original NPY payload');
 return {b,array:descr==='<f4'?new Float32Array(b.buffer,b.byteOffset+offset,rows*3):new Uint32Array(b.buffer,b.byteOffset+offset,rows*3)};
}
async function main(){
 let chunks=[],size=0;for await(const chunk of process.stdin){size+=chunk.length;need(size<=64,'Bounded binary GO');chunks.push(chunk);}need(Buffer.concat(chunks).equals(Buffer.from('GO\n')),'Exact owned binary GO');
 const began=performance.now();let calls=0;
 try{
  need(process.execArgv.includes('--wasm-max-mem-pages=32768')&&process.execArgv.includes('--max-old-space-size=1536'),'Original fixed CPU memory caps');
  const baseline=JSON.parse(fs.readFileSync(plain(path.join(P,'baseline/result.json'))));need(baseline.classification==='PASS','Baseline must PASS');
  const planBytes=read(path.join(P,'baseline/plan.json'),baseline.private_plan_SHA256),plan=JSON.parse(planBytes);
  need(plan.target_triangles===2086792&&plan.error===.005&&JSON.stringify(plan.flags)==='["LockBorder"]','Frozen single candidate');
  need(plan.output===out&&plan.WASM_max_pages===32768&&plan.V8_heap_MiB===1536,'Exact output/caps');
  need(new Uint8Array(new Uint32Array([1]).buffer)[0]===1,'Little endian host');
  const v=npy(plan.positions.path,plan.positions.sha256,'<f4',1957891),f=npy(plan.indices.path,plan.indices.sha256,'<i4',4173584);
  for(let i=0;i<v.array.length;i++)need(Number.isFinite(v.array[i]),'Finite original positions');
  for(let i=0;i<f.array.length;i++)need(f.array[i]<1957891,'Original index range');
  const implementation=read(plan.implementation.path,plan.implementation.sha256);
  need(implementation.length===33612,'Pinned installed implementation');
  const MeshoptSimplifier=require(plan.implementation.path);
  need(MeshoptSimplifier&&typeof MeshoptSimplifier.simplify==='function'&&typeof MeshoptSimplifier.ready?.then==='function','Genuine direct CommonJS simplifier export');
  await MeshoptSimplifier.ready;
  need(MeshoptSimplifier.supported&&MeshoptSimplifier.useExperimentalFeatures===false,'Installed stable CPU path');
  const before=process.memoryUsage(),start=performance.now();calls++;
  const [indices,error]=MeshoptSimplifier.simplify(f.array,v.array,3,plan.target_triangles*3,plan.error,['LockBorder']);
  const seconds=(performance.now()-start)/1000;
  need(indices instanceof Uint32Array&&indices.length>0&&indices.length%3===0&&indices.length<=f.array.length,'Actual candidate triangle buffer');
  need(Number.isFinite(error)&&error>=0&&error<=Math.fround(plan.error),'Reported approximate F32 error');
  for(let i=0;i<indices.length;i++)need(indices[i]<1957891,'Output original vertex IDs');
  need(sha(v.b)===plan.positions.sha256&&sha(f.b)===plan.indices.sha256,'Original in-memory inputs unchanged');
  const bytes=Buffer.from(indices.buffer,indices.byteOffset,indices.byteLength);write('original-index-faces.bin',bytes);
  const again=fs.readFileSync(path.join(out,'original-index-faces.bin'));need(again.equals(bytes),'Actual entire binary readback');
  json('result.json',{classification:'PASS',candidate_calls:calls,CPU_only:true,method:'meshoptimizer0.22 simplify',target_triangles:2086792,actual_triangles:indices.length/3,error_relative_approximate:error,flags:['LockBorder'],no_weld_prune_or_retry:true,positions_and_colour_tables_not_changed:true,plan_SHA256:sha(planBytes),implementation_SHA256:plan.implementation.sha256,simplification_seconds:seconds,elapsed_seconds:(performance.now()-began)/1000,Node:process.version,execArgv:process.execArgv,memory_before:before,memory_after:process.memoryUsage(),files:{'original-index-faces.bin':{bytes:bytes.length,sha256:sha(bytes)}}});
  console.log('ONE_HALF_POLY_CANDIDATE_COMPLETE');
 }catch(e){json('failure.json',{classification:'BLOCKED',candidate_calls:calls,error:String(e),stack:e.stack,elapsed_seconds:(performance.now()-began)/1000});throw e;}
}
main().catch(e=>{console.error(e.stack);process.exitCode=1;});
