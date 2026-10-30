"""One read-only ordinary-user capacity query; no children or model imports."""
import json,os,pwd,re,resource,signal,sys
sys.dont_write_bytecode=True
resource.setrlimit(resource.RLIMIT_AS,(512*1024**2,512*1024**2))
resource.setrlimit(resource.RLIMIT_CPU,(10,10));signal.alarm(15)
token=sys.argv[1]
if not re.fullmatch('[a-f0-9]{32}',token):raise ValueError('Exact nonce required')
if os.getuid()==0 or pwd.getpwuid(os.getuid()).pw_name!='modly':raise ValueError('Ordinary modly user required')
def audit(event,args):
    if event in {'subprocess.Popen','os.system','os.fork','os.posix_spawn','os.exec','socket.connect','socket.bind','os.kill','os.killpg'}:raise RuntimeError('No children/network/signals')
    if event=='import' and args[0].split('.')[0] in {'torch','triton','numpy','trellis2','ctypes','bpy'}:raise ImportError('No native/model imports')
    if event=='open' and (any(x in str(args[1] or '') for x in 'wax+') or args[2]&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND)):raise RuntimeError('Read-only capacity query')
    if event in {'os.mkdir','os.remove','os.rmdir','os.rename','os.link','os.symlink','os.truncate','os.utime','os.chmod','os.chown'}:raise RuntimeError('No filesystem mutations')
sys.addaudithook(audit)
def read(path,limit=65536):
    with open(path,encoding='utf-8') as f:text=f.read(limit+1)
    if len(text)>limit:raise ValueError('Metadata text limit')
    return text
def start_ticks():return int(read('/proc/self/stat').rsplit(')',1)[1].split()[19])

def root_writable(fstype,mount_options,super_options,flags):
    return fstype=='ext4' and 'rw' in mount_options.split(',') and 'rw' in super_options.split(',') and not flags&os.ST_RDONLY

owner=dict(token=token,pid=os.getpid(),start_ticks=start_ticks())
print(json.dumps(dict(phase='READY',**owner)),flush=True)
if sys.stdin.buffer.read(128)!=b'GO\n':raise RuntimeError('Exact GO/EOF required')
v=os.statvfs('/');dev=os.stat('/').st_dev;device=str(os.major(dev))+':'+str(os.minor(dev))
roots=[]
for line in read('/proc/self/mountinfo').splitlines():
    left,right=line.split(' - ',1);a=left.split();b=right.split()
    if a[4]=='/':roots.append((a,b))
if len(roots)!=1:raise ValueError('Unique root mount required')
a,b=roots[0]
if a[2]!=device:raise ValueError('Root device mismatch')
row=dict(**owner,available_bytes=v.f_bavail*v.f_frsize,usable_bytes=v.f_blocks*v.f_frsize,
         root_device_bytes=int(read('/sys/dev/block/'+device+'/size',1024).strip())*512,
         ext4_rw=root_writable(b[0],a[5],b[2],v.f_flag),root_source=b[1],root_options=a[5],
         root_super_options=b[2],filesystem_flags=v.f_flag,
         peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
if start_ticks()!=owner['start_ticks']:raise RuntimeError('Process identity changed')
print(json.dumps(row),flush=True)
