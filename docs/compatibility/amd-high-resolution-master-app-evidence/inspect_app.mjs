// Local, disposable installed-app diagnostics. All actual UI input uses sky.
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
const E=path.dirname(fileURLToPath(import.meta.url));
const P=path.resolve(E,'../../../../artifacts/amd-high-resolution-master-app-64d07f5/desktop-session');
const running=JSON.parse(await fs.readFile(path.join(P,'app-running.json'),'utf8'));
if(!running.attached_before_debugger_resume)throw Error('Owned app required');
const targets=await(await fetch(`http://127.0.0.1:${running.inspector_port}/json/list`)).json();
if(targets.length!==1||targets[0].type!=='node')throw Error('Exact owned Node inspector');
const ws=new WebSocket(targets[0].webSocketDebuggerUrl);let serial=0;const waits=new Map(),events=[];
await new Promise((ok,no)=>{ws.addEventListener('open',ok,{once:true});ws.addEventListener('error',no,{once:true});});
ws.addEventListener('message',event=>{const m=JSON.parse(event.data);if(m.id){const p=waits.get(m.id);if(p){waits.delete(m.id);clearTimeout(p.timer);m.error?p.no(Error(JSON.stringify(m.error))):p.ok(m.result);}}else events.push(m);});
const send=(method,params={})=>new Promise((ok,no)=>{const id=++serial;const timer=setTimeout(()=>{waits.delete(id);no(Error('15s diagnostic deadline: '+method));},15000);waits.set(id,{ok,no,timer});ws.send(JSON.stringify({id,method,params}));});
const value=result=>{if(result.exceptionDetails)throw Error(JSON.stringify(result.exceptionDetails));return result.result.value;};
async function save(name,row){if(!/^[a-z0-9-]+\.json$/.test(name))throw Error('Private receipt basename');await fs.writeFile(path.join(P,name),JSON.stringify(row,null,2)+'\n',{flag:'wx'});}
try{
  await send('Runtime.enable');
  if(process.argv[2]==='bootstrap'){
    await send('Debugger.enable');await send('Runtime.runIfWaitingForDebugger');
    const start=Date.now();let pause;
    while(!(pause=events.find(e=>e.method==='Debugger.paused'))){if(Date.now()-start>15000)throw Error('Initial main pause deadline');await new Promise(r=>setTimeout(r,20));}
    const frame=pause.params.callFrames[0],script=events.find(e=>e.method==='Debugger.scriptParsed'&&e.params.scriptId===frame.location.scriptId);
    const url=frame.url||script?.params.url;
    const main='C:/Users/wattz/AppData/Local/Programs/Modly/resources/app.asar/out/main/index.js';
    const resolved=url?.startsWith('file:')?fileURLToPath(url):url;
    if(!resolved||path.resolve(resolved).toLowerCase()!==path.resolve(main).toLowerCase()||frame.location.lineNumber>1)throw Error('Must pause before first installed main statement');
    const config={mainFile:main,privateRoot:P,profileDir:path.join(P,'profile'),sessionDir:path.join(P,'session')};
    const source=await fs.readFile(path.join(E,'startup_probe.js'),'utf8');
    const initial=value(await send('Debugger.evaluateOnCallFrame',{callFrameId:frame.callFrameId,expression:`(${source})(${JSON.stringify(config)})`,returnByValue:true}));
    if(initial.appReady||!initial.updaterMethodsIntact||initial.autoInstallOnAppQuit!==false)throw Error('Effective isolation before resume');
    const observer=await fs.readFile(path.join(E,'renderer_observer.js'),'utf8');
    const observed=value(await send('Debugger.evaluateOnCallFrame',{callFrameId:frame.callFrameId,expression:`(${observer})()`,returnByValue:true}));
    if(observed.installed!==true||observed.ready!==false||observed.errors?.length)throw Error('Read-only observer installation before resume');
    await save('bootstrap-before-resume.json',{classification:'PASS',pid:running.pid,pause:{url,lineNumber:frame.location.lineNumber},initial,observer:observed});
    await send('Debugger.resume');console.log(JSON.stringify({classification:'PASS',resumed:true,initial,observer:observed}));
  }else{
    let expression='';for await(const chunk of process.stdin){expression+=chunk;if(expression.length>100000)throw Error('Bounded diagnostic expression');}
    if(process.argv[2]==='renderer')expression=`process.mainModule.require('electron').BrowserWindow.getAllWindows()[0].webContents.executeJavaScript(${JSON.stringify(expression)},false)`;
    else if(process.argv[2]!=='main')throw Error('Read main/renderer or bootstrap only');
    const result=value(await send('Runtime.evaluate',{expression,awaitPromise:true,returnByValue:true}));
    if(process.argv[3])await save(process.argv[3],result);
    console.log(JSON.stringify(result));
  }
}finally{ws.close();}
