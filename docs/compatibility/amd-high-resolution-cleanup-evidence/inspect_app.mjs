// Local, disposable installed-app diagnostics. All actual UI input uses sky.
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
const E=path.dirname(fileURLToPath(import.meta.url));
const P=path.resolve(E,'../../../../artifacts/amd-high-resolution-cleanup-bc9515f');
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
async function installCleanup(source) {
      const {BrowserWindow}=process.mainModule.require('electron'), windows=BrowserWindow.getAllWindows();
      if(windows.length!==1)throw Error('Exact isolated app window');
      const d=windows[0].webContents.debugger, group='modly-cleanup-qualification';
      if(!d.isAttached()||!globalThis.__modlyObserver?.ready)throw Error('Owned renderer observer ready');
      const cmd=(method,params)=>d.sendCommand(method,params);
      const remote=async expression=>{const r=await cmd('Runtime.evaluate',{expression,objectGroup:group,returnByValue:false});
        if(r.exceptionDetails||!r.result.objectId)throw Error('Expected actual installed function handle');return r.result.objectId;};
      const scopeNames=async (id,wanted)=>{
        const p=await cmd('Runtime.getProperties',{objectId:id,ownProperties:true});
        const scopes=p.internalProperties?.find(x=>x.name==='[[Scopes]]')?.value?.objectId;
        if(!scopes)throw Error('Installed function scopes unavailable');
        const list=await cmd('Runtime.getProperties',{objectId:scopes,ownProperties:true}), found={};
        for(const s of list.result){if(!/^\d+$/.test(s.name)||!s.value?.objectId||!(/^(Module|Closure)/.test(s.value.description||'')))continue;
          const props=await cmd('Runtime.getProperties',{objectId:s.value.objectId,ownProperties:true});
          for(const x of props.result)if(wanted.includes(x.name)&&x.value?.objectId){
            if(found[x.name]&&found[x.name]!==x.value.objectId)throw Error('Ambiguous installed closure '+x.name);found[x.name]=x.value.objectId;
          }
        }
        for(const n of wanted)if(!found[n])throw Error('Missing installed closure '+n);return found;
      };
      try {
        const viewer=await remote(`(() => {const a=[];for(const ref of window.__modlyRoots||[]){const r=ref.deref();if(!r)continue;
          const stack=[r.current];while(stack.length){const f=stack.pop();if(!f)continue;
            if(f.type?.name==='Viewer3D'&&!a.includes(f.type))a.push(f.type);if(f.child)stack.push(f.child);if(f.sibling)stack.push(f.sibling);}}
          if(a.length!==1)throw Error('One actual installed Viewer3D');return a[0];})()`);
        const module=await scopeNames(viewer,['globalCache','GLTFLoader']);
        const render=await remote(`(() => {const all=[];for(const ref of window.__modlyRoots||[]){const r=ref.deref();if(!r)continue;const st=[r.current];
          while(st.length){const f=st.pop();if(!f)continue;all.push(f);if(f.child)st.push(f.child);if(f.sibling)st.push(f.sibling);}}
          const canvases=all.filter(f=>f.type?.name==='CanvasCapture').map(f=>f.memoizedProps.domRef.current),stores=[];
          for(const f of all){const store=f.memoizedProps?.store;if(!store?.getState)continue;const s=store.getState();
            if(canvases.includes(s.gl?.domElement)&&s.controls?.object===s.camera&&!stores.includes(store))stores.push(store);}
          if(stores.length!==1)throw Error('One actual main renderer');window.__modlyCleanupStore=new WeakRef(stores[0]);
          return stores[0].getState().gl.renderBufferDirect;})()`);
        const renderer=await scopeNames(render,['attributes']);
        const r=await cmd('Runtime.callFunctionOn',{objectId:module.globalCache,
          functionDeclaration:`function(cache,Loader,attributes){if(window.__modlyCleanup)throw Error('Observer already installed');
            window.__modlyCleanup=(${source})(window.__modlyRoots,cache,Loader,attributes);return window.__modlyCleanup.snapshot();}`,
          arguments:[{objectId:module.globalCache},{objectId:module.GLTFLoader},{objectId:renderer.attributes}],returnByValue:true,objectGroup:group});
        if(r.exceptionDetails)throw Error(JSON.stringify(r.exceptionDetails));
        return {classification:'INSTALLED_EMPTY_ONLY',scopeHandlesReleased:true,snapshot:r.result.value};
      }finally{await cmd('Runtime.releaseObjectGroup',{objectGroup:group});}

}
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
    const shutdownSource=await fs.readFile(path.join(E,'shutdown_control.js'),'utf8');
    const shutdown=value(await send('Debugger.evaluateOnCallFrame',{callFrameId:frame.callFrameId,
      expression:`(${shutdownSource})(${JSON.stringify({privateRoot:P,startedEpoch:running.started_epoch,shutdownEpoch:running.shutdown_no_later_epoch})})`,returnByValue:true}));
    if(shutdown.requested||shutdown.shutdownEpoch-running.started_epoch>221)throw Error('Normal shutdown reserve armed before resume');
    const observer=await fs.readFile(path.join(E,'renderer_observer.js'),'utf8');
    const observed=value(await send('Debugger.evaluateOnCallFrame',{callFrameId:frame.callFrameId,expression:`(${observer})()`,returnByValue:true}));
    if(observed.installed!==true||observed.ready!==false||observed.errors?.length)throw Error('Read-only observer installation before resume');
    await save('bootstrap-before-resume.json',{classification:'PASS',pid:running.pid,pause:{url,lineNumber:frame.location.lineNumber},initial,observer:observed,shutdown});
    await send('Debugger.resume');console.log(JSON.stringify({classification:'PASS',resumed:true,initial,observer:observed}));
  }else if(process.argv[2]==='install-cleanup'){
    const source=await fs.readFile(path.join(E,'cleanup_observer.js'),'utf8');
    const expression=`(${installCleanup.toString()})(${JSON.stringify(source)})`;
    const result=value(await send('Runtime.evaluate',{expression,awaitPromise:true,returnByValue:true}));
    await save('cleanup-installed.json',result);console.log(JSON.stringify(result));
  }else{
    let expression='';for await(const chunk of process.stdin){expression+=chunk;if(expression.length>100000)throw Error('Bounded diagnostic expression');}
    if(process.argv[2]==='renderer')expression=`process.mainModule.require('electron').BrowserWindow.getAllWindows()[0].webContents.executeJavaScript(${JSON.stringify(expression)},false)`;
    else if(process.argv[2]!=='main')throw Error('Read main/renderer or bootstrap only');
    const result=value(await send('Runtime.evaluate',{expression,awaitPromise:true,returnByValue:true}));
    if(process.argv[3])await save(process.argv[3],result);
    console.log(JSON.stringify(result));
  }
}finally{ws.close();}
