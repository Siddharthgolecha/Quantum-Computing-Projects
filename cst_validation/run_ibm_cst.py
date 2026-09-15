#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, hashlib, json, math, os, pickle, re, shutil, tarfile, zipfile
from pathlib import Path
from urllib.request import Request, urlopen
import numpy as np
import pandas as pd

RECORD_ID='17291855'; L=12; DIM=1<<L; QTOT=6
API=f'https://zenodo.org/api/records/{RECORD_ID}'
POS=re.compile(r'xxz|sigma.?z|z.?basis|bit|string|prob|pec|bootstrap|boot|mitig|config',re.I)
NEG=re.compile(r'exact|theory|ideal|x.?basis|tfi|ising',re.I)
MASK6=np.array([i.bit_count()==QTOT for i in range(DIM)])

def get_json(url):
    req=Request(url,headers={'User-Agent':'CST-realdata-validation/1.0'})
    with urlopen(req,timeout=180) as r:return json.load(r)

def download(url,path):
    req=Request(url,headers={'User-Agent':'CST-realdata-validation/1.0'})
    path.parent.mkdir(parents=True,exist_ok=True)
    with urlopen(req,timeout=180) as r,open(path,'wb') as f:shutil.copyfileobj(r,f,1024*1024)

def md5(path):
    h=hashlib.md5()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def extract(path,out):
    out.mkdir(parents=True,exist_ok=True)
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as z:z.extractall(out)
        return True
    try:
        if tarfile.is_tarfile(path):
            with tarfile.open(path) as t:t.extractall(out)
            return True
    except Exception:pass
    return False

def fetch_record(root:Path):
    meta=get_json(API); (root/'zenodo_record.json').write_text(json.dumps(meta,indent=2))
    manifest=[]
    for f in meta.get('files',[]):
        key=f.get('key') or f.get('filename') or 'unnamed'; links=f.get('links',{})
        url=links.get('content') or links.get('self') or links.get('download')
        dest=root/'downloads'/key
        if url:
            print('DOWNLOAD',key,f.get('size'))
            download(url,dest)
            chk=f.get('checksum') or ''
            if chk.startswith('md5:'):
                got=md5(dest); exp=chk.split(':',1)[1]
                if got.lower()!=exp.lower(): raise RuntimeError(f'checksum mismatch {key}')
            try: extract(dest,root/'extracted'/Path(key).name)
            except Exception as e: print('extract warning',e)
        manifest.append({'key':key,'size':f.get('size'),'checksum':f.get('checksum'),'url':url})
    pd.DataFrame(manifest).to_csv(root/'download_manifest.csv',index=False)

def score_name(s):return 2*len(POS.findall(s))-3*len(NEG.findall(s))

def normalize_array(label,obj):
    try:a=np.asarray(obj)
    except Exception:return []
    if not np.issubdtype(a.dtype,np.number):return []
    a=a.astype(float)
    if a.ndim==1 and a.size==DIM:return [(label,a[None,:])]
    if a.ndim==2:
        if a.shape[1]==DIM:return [(label,a)]
        if a.shape[0]==DIM:return [(label,a.T)]
    return []

def json_walk(label,obj,depth=0):
    out=[]
    if depth>7:return out
    if isinstance(obj,dict):
        keys=list(obj)
        bitkeys=[]
        for k in keys:
            s=str(k).replace(' ','').replace('_','')
            if len(s)==L and not(set(s)-set('01')):bitkeys.append(k)
        if len(bitkeys)>=min(100,len(keys)):
            v=np.zeros(DIM)
            try:
                for k in bitkeys:v[int(str(k).replace(' ','').replace('_',''),2)]=float(obj[k])
                out.append((label,v[None,:]))
            except Exception:pass
        for k,v in obj.items():
            if isinstance(v,(dict,list)):out.extend(json_walk(label+':'+str(k),v,depth+1))
    elif isinstance(obj,list):
        out.extend(normalize_array(label,obj))
        for i,v in enumerate(obj[:100]):
            if isinstance(v,(dict,list)):out.extend(json_walk(label+f':{i}',v,depth+1))
    return out

def discover(path:Path):
    ext=path.suffix.lower(); out=[]
    try:
        if ext=='.npy':out+=normalize_array(path.name,np.load(path,allow_pickle=True))
        elif ext=='.npz':
            z=np.load(path,allow_pickle=True)
            for k in z.files:out+=normalize_array(path.name+':'+k,z[k])
        elif ext=='.json':out+=json_walk(path.name,json.loads(path.read_text(errors='ignore')))
        elif ext in {'.pkl','.pickle'}:
            with open(path,'rb') as f:o=pickle.load(f)
            out+=normalize_array(path.name,o); out+=json_walk(path.name,o) if isinstance(o,(dict,list)) else []
        elif ext=='.csv':
            df=pd.read_csv(path); low={str(c).lower():c for c in df.columns}
            bc=next((low[x] for x in ['bitstring','bit_string','state','configuration','config'] if x in low),None)
            pc=next((low[x] for x in ['probability','prob','p','weight','quasi_probability','quasiprobability'] if x in low),None)
            if bc is not None and pc is not None:
                v=np.zeros(DIM); good=True
                for b,p in zip(df[bc],df[pc]):
                    s=str(b).replace(' ','').replace('_','').zfill(L)
                    if len(s)!=L or set(s)-set('01'):good=False;break
                    v[int(s,2)]=float(p)
                if good:out.append((path.name,v[None,:]))
            bcols=[c for c in df.columns if len(str(c))==L and not(set(str(c))-set('01'))]
            if len(bcols)>=DIM//2:
                arr=np.zeros((len(df),DIM))
                for c in bcols:arr[:,int(str(c),2)]=pd.to_numeric(df[c],errors='coerce').fillna(0)
                out.append((path.name+':wide',arr))
    except Exception as e:
        print('discover warning',path,e)
    return out

def physicalize(v):
    v=np.asarray(v,float).copy(); v[~MASK6]=0.0
    neg=v[MASK6 & (v<0)]; cutoff=float(np.sqrt(np.mean(neg**2))) if neg.size else 0.0
    v[v<cutoff]=0.0; s=float(v.sum())
    if not np.isfinite(s) or s<=0:return None,cutoff
    return v/s,cutoff

def bit_matrix(A):
    idx=np.arange(DIM,dtype=np.uint16)
    return np.stack([((idx>>(L-1-j))&1).astype(float) for j in range(A)],axis=1)

BM={A:bit_matrix(A) for A in range(1,L)}

def analyze_vec(v,A):
    X=BM[A]; q=X.sum(axis=1)
    mu=float(v@q); dq=q-mu; var=float(v@(dq*dq))
    qvals=np.arange(A+1); pq=np.array([v[q==k].sum() for k in qvals],float)
    q0=int(np.argmin(np.abs(qvals-mu))); p0=float(pq[q0]); Iobs=float(-math.log(p0)) if p0>0 else math.inf
    r=1 if var>1e-12 else 0; logdet=math.log(var) if r else 0.0; V=1.0
    Icst=0.5*logdet-math.log(V)+0.5*r*math.log(2*math.pi)
    R=Iobs-Icst if np.isfinite(Iobs) else math.inf
    m=v@X; centered=X-m
    C=(centered*v[:,None]).T@centered
    eig=np.linalg.eigvalsh((C+C.T)/2)[::-1]
    return {'A':A,'mu':mu,'variance_Q':var,'rank_Q':r,'q0':q0,'p0':p0,'I_obs':Iobs,'I_CST':Icst,'R':R,
            'charge_distribution':json.dumps({int(k):float(p) for k,p in zip(qvals,pq)}),
            'site_cov_eigenvalues':json.dumps([float(x) for x in eig]),
            'site_cov_rank':int(np.sum(eig>max(1e-12,(eig[0] if eig.size else 0)*1e-8)))}

def fit_slope(metrics):
    m=[x for x in metrics if 2<=x['A']<=L//2 and np.isfinite(x['R'])]
    if len(m)<2:return {'slope_logA':math.nan,'intercept_logA':math.nan,'slope_A':math.nan,'intercept_A':math.nan}
    A=np.array([x['A'] for x in m],float); R=np.array([x['R'] for x in m],float)
    blog,alog=np.polyfit(np.log(A),R,1); bA,aA=np.polyfit(A,R,1)
    return {'slope_logA':float(blog),'intercept_logA':float(alog),'slope_A':float(bA),'intercept_A':float(aA)}

def summarize_candidate(label,arr,outdir):
    reps=[]; cut=[]
    for row in arr:
        v,c=physicalize(row)
        if v is not None:reps.append(v);cut.append(c)
    if not reps:return None
    reps=np.asarray(reps); meanv=reps.mean(axis=0);meanv/=meanv.sum()
    base=[analyze_vec(meanv,A) for A in range(1,L)]
    boot=[[analyze_vec(v,A) for A in range(1,L)] for v in reps]
    trends=[fit_slope(m) for m in boot]
    safe=re.sub(r'[^A-Za-z0-9_.-]+','_',label)[:150]; d=outdir/safe;d.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(base).to_csv(d/'cst_metrics.csv',index=False)
    rows=[]
    for bi,mm in enumerate(boot):
        for x in mm:rows.append({'bootstrap':bi,**x})
    pd.DataFrame(rows).to_csv(d/'native_bootstrap_metrics.csv',index=False)
    pd.DataFrame(trends).to_csv(d/'native_bootstrap_trends.csv',index=False)
    summary=[]
    for A in range(1,L):
        rr=np.array([m[A-1]['R'] for m in boot],float)
        summary.append({'A':A,'R_mean':float(np.mean(rr)),'R_sd':float(np.std(rr,ddof=1)) if len(rr)>1 else math.nan,
                        'R_ci025':float(np.quantile(rr,.025)) if len(rr)>1 else math.nan,'R_ci975':float(np.quantile(rr,.975)) if len(rr)>1 else math.nan})
    pd.DataFrame(summary).to_csv(d/'residual_bootstrap_summary.csv',index=False)
    tv=np.array([t['slope_logA'] for t in trends],float)
    primary=fit_slope(base)
    return {'label':label,'n_native_replicates':len(reps),'mean_cutoff':float(np.mean(cut)),'slope_logA_point':primary['slope_logA'],
            'slope_logA_boot_mean':float(np.nanmean(tv)),'slope_logA_ci025':float(np.nanquantile(tv,.025)) if len(tv)>1 else math.nan,
            'slope_logA_ci975':float(np.nanquantile(tv,.975)) if len(tv)>1 else math.nan,
            'falsify_growing_residual':bool(len(tv)>1 and np.nanquantile(tv,.025)>0),'output_dir':str(d)}

def ghz_control(out):
    rows=[]
    for A in [2,4,8,16,32,64,128]:
        var=A*A/4; Iobs=math.log(2); Icst=.5*math.log(2*math.pi*var)
        rows.append({'A':A,'variance_Q':var,'p0':.5,'I_obs':Iobs,'I_CST':Icst,'R':Iobs-Icst})
    pd.DataFrame(rows).to_csv(out/'ghz_non_gaussian_control.csv',index=False)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',default='ibm_xxz_zenodo');ap.add_argument('--output',default='cst_results');ap.add_argument('--top',type=int,default=12);ap.add_argument('--skip-download',action='store_true')
    a=ap.parse_args(); root=Path(a.root);out=Path(a.output);root.mkdir(parents=True,exist_ok=True);out.mkdir(parents=True,exist_ok=True)
    if not a.skip_download:fetch_record(root)
    candidates=[]
    for p in root.rglob('*'):
        if not p.is_file() or p.suffix.lower() not in {'.npy','.npz','.json','.csv','.pkl','.pickle'}:continue
        for label,arr in discover(p):
            if arr.shape[1]!=DIM:continue
            sc=score_name(str(p.relative_to(root))+' '+label); sums=arr.sum(axis=1); mass=[]
            for row,s in zip(arr[:20],sums[:20]):mass.append(float(row[MASK6].sum()/s) if s else math.nan)
            candidates.append({'score':sc,'path':p,'label':label,'arr':arr,'sector6_mass':float(np.nanmean(mass)),
                               'negative_fraction':float(np.mean(arr<0)),'row_sum_mean':float(np.mean(sums))})
    candidates.sort(key=lambda x:(x['score'],x['sector6_mass'],x['arr'].shape[0]),reverse=True)
    inv=[{'score':c['score'],'path':str(c['path'].relative_to(root)),'label':c['label'],'shape':str(c['arr'].shape),'sector6_mass':c['sector6_mass'],
          'negative_fraction':c['negative_fraction'],'row_sum_mean':c['row_sum_mean']} for c in candidates]
    pd.DataFrame(inv).to_csv(out/'candidate_distributions.csv',index=False)
    print('TOP CANDIDATES');print(pd.DataFrame(inv[:30]).to_string(index=False) if inv else 'NONE')
    results=[]
    for c in candidates[:a.top]:
        r=summarize_candidate(f"{c['path'].relative_to(root)}::{c['label']}",c['arr'],out/'analyses')
        if r:r.update(score=c['score'],sector6_mass=c['sector6_mass']);results.append(r)
    pd.DataFrame(results).to_csv(out/'validation_summary.csv',index=False)
    ghz_control(out)
    rows=[]
    for p in root.rglob('*'):
        if p.is_file():rows.append({'path':str(p.relative_to(root)),'size_bytes':p.stat().st_size,'suffix':p.suffix})
    pd.DataFrame(rows).to_csv(out/'file_inventory.csv',index=False)
    (out/'run_manifest.json').write_text(json.dumps({'record_id':RECORD_ID,'L':L,'global_charge':QTOT,'primary_A':[2,3,4,5,6],
        'criterion':'lower 95% native-bootstrap CI of slope of R vs log(A) > 0 flags growing residual over measured range',
        'n_candidates':len(candidates),'n_analyzed':len(results)},indent=2))
    print('\nSUMMARY');print(pd.DataFrame(results).to_string(index=False) if results else 'NO 4096-STATE DISTRIBUTION AUTO-DISCOVERED')

if __name__=='__main__':main()
