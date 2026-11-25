"""Turn the FMG stand into an FD stand in every image of a folder.

Instead of pasting a stand photo (which looks soft and cut-out), this edits the
stand that is already in each image: the printed panels are painted plain black
and a crisp FD logo is added. Lighting, edges and sharpness stay native.

Usage:  python replace_stands.py INPUT_FOLDER OUTPUT_FOLDER
Needs:  pip install opencv-python pillow numpy
Images the script is unsure about are saved with '_CHECK' in the name and listed
in OUTPUT_FOLDER/needs_review.txt."""
import cv2, numpy as np, os, sys
from PIL import Image
A=os.path.join(os.path.dirname(os.path.abspath(__file__)),'assets')
REF=cv2.imread(f'{A}/stand_ref.png',0); BH,BW=REF.shape            # reference stand box (740x790)
TMPL=(REF[int(BH*.62):,:]<60).astype(np.uint8)*255                  # legs + crossbar silhouette
LOGO=cv2.imread(f'{A}/fd_logo_alpha.png',0).astype(np.float32)/255; LH,LW=LOGO.shape
MIN_SCORE=0.70
TOP=[(158,74),(586,74),(590,132),(154,132)]; TOP_O=[(146,66),(598,66),(602,142),(142,142)]
BOT=[(92,510),(140,504),(220,496),(300,490),(380,489),(460,490),(540,495),(580,501),(620,505),(642,509)]
def _mid(dx,dy): return [(112-dx,264),(618+dx,264)]+[(x+(dx if x>370 else -dx),y-dy) for x,y in reversed(BOT)]
MID=_mid(0,3); MID_O=_mid(14,-3); MID_G=_mid(18,-26)
EXT=('.jpg','.jpeg','.png','.webp')

def smooth(x,a,b): t=np.clip((x-a)/(b-a),0,1); return t*t*(3-2*t)
def pmask(shape,poly,x0,y0,S,feather):
    m=np.zeros(shape,np.float32); pts=np.array([[x0+u*S,y0+v*S] for u,v in poly],np.float32)
    cv2.fillPoly(m,[np.round(pts*16).astype(np.int32)],1.0,lineType=cv2.LINE_AA,shift=4)
    return cv2.GaussianBlur(m,(0,0),feather) if feather>0 else m

def detect(rgb,thr=60):
    g=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY); H,W=g.shape
    k=1200/max(H,W) if max(H,W)>1200 else 1.0
    m=(cv2.resize(g,None,fx=k,fy=k,interpolation=cv2.INTER_AREA)<thr).astype(np.uint8)*255
    best=(-1,0,(0,0))
    for s in np.linspace(0.05,0.45,81):
        t=cv2.resize(TMPL,None,fx=s,fy=s,interpolation=cv2.INTER_AREA)
        if t.shape[0]>=m.shape[0] or t.shape[1]>=m.shape[1] or t.shape[1]<15: continue
        _,sc,_,loc=cv2.minMaxLoc(cv2.matchTemplate(m,t,cv2.TM_CCOEFF_NORMED))
        if sc>best[0]: best=(sc,s,loc)
    sc,s,(x,y)=best; S=s/k
    return sc,S,(int(x/k),int(y/k-BH*.62*S))

def process(path):
    src=Image.open(path).convert('RGBA'); orig=np.array(src)
    bg=Image.new('RGBA',src.size,(255,255,255,255)); bg.alpha_composite(src)
    comp=np.array(bg.convert('RGB')); arr=comp.astype(np.float32)
    gray=cv2.cvtColor(comp,cv2.COLOR_RGB2GRAY).astype(np.float32)
    sc,S,(x0,y0)=detect(comp); fe=max(0.6,0.6*S)
    core=np.maximum(pmask(gray.shape,TOP,x0,y0,S,fe),pmask(gray.shape,MID,x0,y0,S,fe))
    outer=np.maximum(pmask(gray.shape,TOP_O,x0,y0,S,fe),pmask(gray.shape,MID_O,x0,y0,S,fe))
    zone=pmask(gray.shape,[(175,165),(565,165),(565,235),(175,235)],x0,y0,S,0)>0.99
    dk=zone&(gray<70); base=np.median(arr[dk],axis=0) if dk.sum()>50 else np.array([20,20,20],np.float32)
    base=np.maximum(base,14)
    G=np.maximum(pmask(gray.shape,TOP_O,x0,y0,S,0),pmask(gray.shape,MID_G,x0,y0,S,0))
    bright=((gray>105)&(G>0.5)).astype(np.uint8)
    _,lab=cv2.connectedComponents(bright,connectivity=4)
    hit=np.unique(lab[(core>0.5)&(bright>0)]); hit=hit[hit>0]
    k=max(3,int(round(S*5))|1)
    blob=cv2.dilate(np.isin(lab,hit).astype(np.uint8),np.ones((k,k),np.uint8)).astype(np.float32)
    blob=cv2.GaussianBlur(blob,(0,0),max(0.6,0.5*S))*(G>0.5)
    w=np.maximum.reduce([core,outer*smooth(gray,55,115),blob])
    out=arr*(1-w[...,None])+base*w[...,None]
    lw=int(350*S); lh=int(lw*LH/LW)
    if lw>8:
        la=cv2.resize(LOGO,(lw,lh),interpolation=cv2.INTER_AREA)
        lx,ly=int(x0+365*S-lw/2),int(y0+392*S-lh/2); H,W=gray.shape
        xa,ya,xb,yb=max(lx,0),max(ly,0),min(lx+lw,W),min(ly+lh,H)
        a=la[ya-ly:yb-ly,xa-lx:xb-lx,None]
        out[ya:yb,xa:xb]=out[ya:yb,xa:xb]*(1-a)+np.array([238,238,238],np.float32)*a
    out=np.clip(out,0,255).astype(np.uint8)
    changed=(np.abs(out.astype(int)-comp.astype(int)).sum(2)>0)[...,None]
    res=orig.copy(); res[...,:3]=np.where(changed,out,orig[...,:3])    # keep original alpha/pixels elsewhere
    return sc,Image.fromarray(res,'RGBA')

def save(img,dst,ext):
    if ext in ('.jpg','.jpeg'): img.convert('RGB').save(dst,quality=95,subsampling=0)
    elif ext=='.webp': img.save(dst,quality=95,method=6)
    else: img.save(dst)

if __name__=='__main__':
    if len(sys.argv)<3: sys.exit('Usage: python replace_stands.py INPUT_FOLDER OUTPUT_FOLDER')
    src,dst=os.path.abspath(sys.argv[1]),os.path.abspath(sys.argv[2])
    print('Input folder :',src); print('Output folder:',dst)
    if not os.path.isdir(src): sys.exit('ERROR: input folder does not exist. Check the path.')
    files=sorted(os.path.join(r,n) for r,_,ns in os.walk(src) for n in ns if n.lower().endswith(EXT))
    print('Images found :',len(files))
    if not files: sys.exit('ERROR: no .jpg/.jpeg/.png/.webp images found in the input folder.')
    os.makedirs(dst,exist_ok=True); review=[]; ok=0
    for i,p in enumerate(files,1):
        rel=os.path.relpath(p,src); stem,ext=os.path.splitext(rel); stem=stem.replace(os.sep,'__'); ext=ext.lower()
        try: sc,res=process(p)
        except Exception as e:
            review.append(f'{rel}  ERROR {e}'); print(f'[{i}/{len(files)}] ERROR {rel}: {e}'); continue
        if sc<MIN_SCORE: review.append(f'{rel}  score {sc:.2f}'); stem+='_CHECK'
        save(res,os.path.join(dst,stem+ext),ext); ok+=1
        print(f'[{i}/{len(files)}] {rel}  score {sc:.2f}')
    open(os.path.join(dst,'needs_review.txt'),'w').write('\n'.join(review))
    print(f'Done. Saved {ok} of {len(files)} images. Need review: {len(review)}')
