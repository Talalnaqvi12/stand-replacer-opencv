"""Compare the input folder with the output folder.
Usage:  python count_outputs.py INPUT_FOLDER OUTPUT_FOLDER"""
import os, sys
EXT=('.jpg','.jpeg','.png','.webp')
if len(sys.argv)<3: sys.exit('Usage: python count_outputs.py INPUT_FOLDER OUTPUT_FOLDER')
src,dst=os.path.abspath(sys.argv[1]),os.path.abspath(sys.argv[2])
for p,l in ((src,'input'),(dst,'output')):
    if not os.path.isdir(p): sys.exit(f'ERROR: {l} folder does not exist: {p}')
exp={os.path.splitext(os.path.relpath(os.path.join(r,n),src))[0].replace(os.sep,'__'):os.path.relpath(os.path.join(r,n),src)
     for r,_,ns in os.walk(src) for n in ns if n.lower().endswith(EXT)}
outs=[f for f in os.listdir(dst) if f.lower().endswith(EXT)]
done=set(); flagged=[]
for f in outs:
    b=os.path.splitext(f)[0]
    if b.endswith('_CHECK'): flagged.append(f); b=b[:-6]
    done.add(b)
missing=sorted(rel for k,rel in exp.items() if k not in done); extra=sorted(d for d in done if d not in exp)
print(f'Input images    : {len(exp)}\nOutput images   : {len(outs)}\nFlagged (_CHECK): {len(flagged)}\nMissing outputs : {len(missing)}')
if flagged: print('\nFlagged - check these by hand:'); [print('  ',x) for x in sorted(flagged)]
if missing: print('\nMissing - no output saved for:'); [print('  ',x) for x in missing]
if extra: print('\nOutput files with no matching input:'); [print('  ',x) for x in extra]
if not missing: print('\nOK: every input image has an output file.')
