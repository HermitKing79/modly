"""Fresh CPU image reopening and private labelled contextual comparison."""
import sys
sys.dont_write_bytecode=True
import json
from pathlib import Path
E=Path(__file__).resolve().parent;sys.path.insert(0,str(E))
from inputs import *
def main():
    require(sys.stdin.buffer.readline(64)==b'GO\n','Owned binary GO')
    auth=load_auth();auth.P=P;sys.addaudithook(auth.runtime_guard)
    from PIL import Image,ImageDraw,ImageStat
    out=P/'inspect';r=json.loads((P/'render/result.json').read_text());require(r['classification']=='PASS' and r['render_count']==1,'One actual CPU view')
    path=P/'render/view-pos-x-ev2.png';require(digest(plain(path))==r['files'][path.name]['sha256'],'Actual render identity')
    require(digest(plain(REFERENCE))==PINS[REFERENCE],'Original reference preview')
    with Image.open(path) as image:
        image.load();require(image.format=='PNG' and image.mode=='RGB' and image.size==(640,640),'Actual640RGB PNG');require(max(ImageStat.Stat(image).stddev)>5,'Nonblank image')
        current=image.copy()
    with Image.open(REFERENCE) as image:
        image.load();require(image.mode=='RGB' and image.size==(1024,1024),'Original high-res PBR reference');prior=image.resize((640,640),Image.Resampling.LANCZOS)
    sheet=Image.new('RGB',(1280,720),'white');sheet.paste(prior,(0,60));sheet.paste(current,(640,60));d=ImageDraw.Draw(sheet)
    d.text((12,10),'Original direct-row PBR / gamma2.2 / +2 EV',fill='black');d.text((652,10),'Imported source-sampled vertex RGB / +2 EV',fill='black')
    d.text((12,32),'Contextual comparison: sampling and shading semantics differ; geometry unchanged',fill='black')
    comparison=out/'comparison-pos-x.png';sheet.save(comparison)
    with Image.open(comparison) as image:image.load();require(image.size==(1280,720),'Independent comparison reopening')
    save(out/'result.json',dict(classification='PASS',PNG_verified=True,geometry_or_material_modified=False,files={path.name:dict(bytes=path.stat().st_size,sha256=digest(path)),comparison.name:dict(bytes=comparison.stat().st_size,sha256=digest(comparison))},interpretation='No pixel-perfect equivalence or new material-quality claim; independent human/agent visual inspection remains required'))
if __name__=='__main__':main()
