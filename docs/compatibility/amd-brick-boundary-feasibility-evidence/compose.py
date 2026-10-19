"""One private A1 crop and exact projected graph; no render or mesh write."""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
import hashlib
import json
E=Path(__file__).resolve().parent
ROOT=E.parents[2]
P=ROOT.parent/'artifacts/amd-brick-boundary-feasibility-677d741'
PREV=ROOT/'docs/compatibility/amd-brick-defect-diagnosis-evidence'
Q=P/'recovery/qualification.json'
Q_SHA='2b11f15b22ba3c663c68278d585ec53c144a9ca09fae564062e8ab615d55cd6b'
PNG=ROOT.parent/'artifacts/amd-brick-pbr-exposure-b0b7be1/renders/view-pos-x-ev2.png'
PNG_SHA='74d1655d53d5fe7d262756c7e23c0a695aa42b7ee3c43e075007a59fc3bf7f54'
def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
    if sys.stdin.buffer.readline(64)!=b'GO\n':raise ValueError('Owned-host GO required')
    if sha(Q)!=Q_SHA or sha(PNG)!=PNG_SHA:raise ValueError('Authentic measurements/image required')
    sys.path.insert(0,str(PREV))
    import diagnose as auth
    auth.P=P
    sys.addaudithook(auth.runtime_guard)
    from PIL import Image,ImageDraw,ImageFont
    data=json.loads(Q.read_text());case=data['cases'][0]
    assert case['case']=='A1' and data['originals_unchanged']
    comp=data['components']['1014637']
    assert comp['edge_count']==2 and not comp['closed_simple_graph_cycle']
    nodes={}
    for face in case['hit_face_and_full_vertex_star']:
        for vertex,xy in zip(face['original_vertex_ids'],face['projected_vertices']):
            if vertex in nodes:assert nodes[vertex]==xy
            nodes[vertex]=xy
    letters=dict(zip('ABCDE',[1143327,1143329,1143348,1154379,1154395]))
    points=[nodes[v] for v in letters.values()]
    lower=[min(p[i] for p in points)-.1 for i in range(2)]
    upper=[max(p[i] for p in points)+.1 for i in range(2)]
    scale=420/max(upper[i]-lower[i] for i in range(2))
    def xy(vertex):return tuple([640,130][i]+(nodes[vertex][i]-lower[i])*scale for i in range(2))
    image=Image.new('RGB',(1160,820),'#f2f4f7');draw=ImageDraw.Draw(image)
    fonts={}
    def text(p,s,size=18,bold=False,color='#202830'):
        key=(size,bold)
        if key not in fonts:fonts[key]=ImageFont.truetype('C:/Windows/Fonts/segoeuib.ttf' if bold else 'C:/Windows/Fonts/segoeui.ttf',size)
        draw.text(p,s,font=fonts[key],fill=color)
    text((24,14),'A1: open boundary chain, not a qualified hole rim',27,True)
    text((24,53),'Original +2 EV image and original topology only. No repair, relighting or new surface render.',18)
    box=(514,352,555,393)
    with Image.open(PNG) as original:
        original.load();crop=original.convert('RGB').crop(box).resize((492,492),Image.Resampling.NEAREST)
    image.paste(crop,(24,112))
    text((24,84),'Untouched 12x crop; A1 pixel is at the panel center',17,True)
    draw.rectangle((22,110,517,605),outline='#2473bb',width=2)
    draw.line((264,104,276,104),fill='#2473bb',width=2)
    text((24,615),'Pixel (534,372); exact ray samples (534.5,372.5).',17)
    text((620,84),'Projected original graph (uniform enlargement)',17,True)
    # Only original edges, with the original image projection/orientation.
    face=next(f for f in case['hit_face_and_full_vertex_star'] if f['face_id']==2369396)
    facepoints=[xy(v) for v in face['original_vertex_ids']]
    draw.line(facepoints+[facepoints[0]],fill='#aaaeb2',width=2)
    for a,b in [(1143327,1143329),(1143329,1143348)]:
        draw.line((xy(a),xy(b)),fill='#d16b00',width=4)
    for a,b in comp['nonmanifold_edge_witnesses']:
        aa,bb=xy(a),xy(b)
        for k in range(0,20,2):
            draw.line(tuple(tuple(aa[i]+(bb[i]-aa[i])*t/20 for i in range(2)) for t in (k,k+1)),fill='#be2588',width=3)
    for label,vertex in letters.items():
        x,y=xy(vertex);draw.ellipse((x-4,y-4,x+4,y+4),fill='#182737')
        text((x+7,y-16),label,20,True)
    px=640+(534.5-lower[0])*scale;py=130+(372.5-lower[1])*scale
    draw.line((px-6,py,px+6,py),fill='#2473bb',width=2);draw.line((px,py-6,px,py+6),fill='#2473bb',width=2)
    text((620,575),'Orange: two incidence-1 edges, A-B-C (open).',17)
    text((620,601),'Magenta: incidence-3 edges A-D and C-E.',17)
    text((620,627),'Grey: original face 2369396. Blue: A1 ray.',17)
    text((620,653),'Image axes preserved: right +Y, up +Z; view +X.',16)
    text((24,672),'A=1143327   B=1143329   C=1143348   D=1154379   E=1154395',18,True)
    text((24,707),'Both chain endpoints meet non-manifold junctions. No closed loop, opening area or safe patch is established.',17)
    text((24,742),'Projected graph shows these original edges at their recorded depths; it is not an occlusion/lighting rendering.',17)
    out=P/'a1-boundary-closeup.png'
    with out.open('xb') as f:image.save(f,format='PNG')
    with Image.open(out) as check:
        check.load();assert check.size==(1160,820) and check.mode=='RGB'
        assert check.crop((24,112,516,604)).tobytes()==crop.tobytes()
    assert sha(PNG)==PNG_SHA and sha(Q)==Q_SHA
    record=dict(classification='PASS_IMAGE_READBACK',path=str(out),bytes=out.stat().st_size,
                sha256=sha(out),dimensions=[1160,820],source_png_sha256=PNG_SHA,
                qualification_sha256=Q_SHA,original_crop_box=box,nearest_neighbor_enlargement=12,
                unannotated_crop_pixel_exact=True,vertex_letters=letters,
                original_projected_points={str(v):nodes[v] for v in letters.values()},
                original_face=2369396,boundary_edges=[[1143327,1143329],[1143329,1143348]],
                nonmanifold_edges=comp['nonmanifold_edge_witnesses'],repair_approval=False)
    with (P/'image-metadata.json').open('x') as f:json.dump(record,f,indent=2);f.write('\n')
    print(json.dumps(record),flush=True)
if __name__=='__main__':main()
