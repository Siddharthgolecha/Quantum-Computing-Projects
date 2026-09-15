#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, math, shutil, tarfile, zipfile
from pathlib import Path
from urllib.request import Request, urlopen
import numpy as np
import pandas as pd

RECORD_ID='17291855'; L=12; DIM=1<<L; QTOT=L//2
API=f'https://zenodo.org/api/records/{RECORD_ID}'
IDX=np.arange(DIM,dtype=np.uint16)
BITS=np.stack([((IDX>>i)&1).astype(float) for i in range(L)],axis=1)
ZMAGS=BITS.sum(axis=1).astype(int); MASK=ZMAGS==QTOT

def get_json(url):
    req=Request(url,headers={'User-Agent':'CST-realdata-validation/2.0'})
    with urlopen(req,timeout=180) as r:return json.load(r)

def download(url,path):
    req=Request(url,headers={'User-Agent':'CST-realdata-validation/2.0'})
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

def fetch(root):
    meta=get_json(API); (root/'zenodo_record.json').write_text(json.dumps(meta,indent=2))
    rows=[]
    for f in meta.get('files',[]):
        key=f.get('key') or f.get('filename') or 'unnamed'; links=f.get('links',{})
        url=links.get('content') or links.get('self') or links.get('download'); dest=root/'downloads'/key
        if url:
            print('DOWNLOAD',key,f.get('size')); download(url,dest)
            chk=f.get('checksum') or ''
            if chk.startswith('md5:') and md5(dest).lower()!=chk.split(':',1)[1].lower():raise RuntimeError('checksum mismatch '+key)
            extract(dest,root/'extracted'/Path(key).name)
        rows.append({'key':key,'size':f.get('size'),'checksum':f.get('checksum'),'url':url})
    pd.DataFrame(rows).to_csv(root/'download_manifest.csv',index=False)

def find_z_file(root):
    hits=list(root.rglob('XXZ_12_periodic_z_bootstrapped_bitstrings.csv'))
    if not hits:raise FileNotFoundError('XXZ_12_periodic_z_bootstrapped_bitstrings.csv not found')
    return hits[0]

def paper_physicalize_all(raw):
    min_neg=np.array([row[MASK].min() for row in raw],float)
    cutoff=float(np.sqrt(np.mean(np.square(min_neg))))
    out=[]
    for row in raw:
        v=np.asarray(row,float).copy(); v[~MASK]=0.; v[v<cutoff]=0.
        den=float(np.sum(np.abs(v)))
        if den<=0:raise RuntimeError('physicalized bootstrap has zero mass')
        out.append(v/den)
    return np.asarray(out),cutoff,min_neg

def central_shell(A):
    if A%2==0:return [A//2]
    return [A//2,A//2+1]

def metrics(v,A):
    pqs=[]; Cs=[]
    for start in range(L):
        sites=[(start+j)%L for j in range(A)]
        X=BITS[:,sites]; q=X.sum(axis=1).astype(int)
        pq=np.bincount(q,weights=v,minlength=A+1); pqs.append(pq)
        m=v@X; Y=X-m; Cs.append((Y*v[:,None]).T@Y)
    pq=np.mean(pqs,axis=0); Csite=np.mean(Cs,axis=0)
    qvals=np.arange(A+1,dtype=float); mu=float(qvals@pq); var=float(((qvals-mu)**2)@pq)
    shell=central_shell(A); V=float(len(shell)); p0=float(pq[shell].sum()); Iobs=-math.log(p0)
    r=int(var>1e-12); logdet=math.log(var) if r else 0.0
    Icst=.5*logdet-math.log(V)+.5*r*math.log(2*math.pi); R=Iobs-Icst
    if r:
        gmass=sum(math.exp(-(q-mu)**2/(2*var)) for q in shell)/math.sqrt(2*math.pi*var)
        Icst_lat=-math.log(gmass); R_lat=Iobs-Icst_lat
    else:Icst_lat=-math.log(V);R_lat=Iobs-Icst_lat
    eig=np.linalg.eigvalsh((Csite+Csite.T)/2)[::-1]
    return {'A':A,'shell':'|'.join(map(str,shell)),'V':V,'mean_Q':mu,'rank_C':r,'detprime_C':var if r else 1.0,
            'charge_cov_spectrum':json.dumps([var] if r else []),'site_cov_spectrum':json.dumps([float(x) for x in eig]),
            'p0':p0,'I_obs':Iobs,'I_CST':Icst,'R':R,'I_CST_lattice_diagnostic':Icst_lat,'R_lattice_diagnostic':R_lat,
            'charge_distribution':json.dumps({str(q):float(pq[q]) for q in range(A+1)})}

def fit(ms,key='R',even=False):
    sel=[m for m in ms if 2<=m['A']<=6 and (not even or m['A']%2==0) and np.isfinite(m[key])]
    if len(sel)<2:return {'slope_logA':math.nan,'slope_A':math.nan}
    A=np.array([m['A'] for m in sel],float); y=np.array([m[key] for m in sel],float)
    return {'slope_logA':float(np.polyfit(np.log(A),y,1)[0]),'slope_A':float(np.polyfit(A,y,1)[0])}

def quant(x,q):return float(np.quantile(np.asarray(x,float),q))

def ghz(out):
    rows=[]
    for A in [2,4,8,16,32,64,128]:
        var=A*A/4; Iobs=math.log(2); Icst=.5*math.log(2*math.pi*var)
        rows.append({'A':A,'p0':.5,'detprime_C':var,'I_obs':Iobs,'I_CST':Icst,'R':Iobs-Icst})
    pd.DataFrame(rows).to_csv(out/'ghz_non_gaussian_control.csv',index=False)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',default='ibm_xxz_zenodo');ap.add_argument('--output',default='cst_results');ap.add_argument('--top',type=int,default=16);a=ap.parse_args()
    root=Path(a.root);out=Path(a.output);root.mkdir(parents=True,exist_ok=True);out.mkdir(parents=True,exist_ok=True)
    fetch(root); src=find_z_file(root); print('SOURCE',src)
    raw=np.loadtxt(src,delimiter=',')
    if raw.shape!=(20,DIM):print('WARNING expected (20,4096), got',raw.shape)
    phys,cutoff,min_neg=paper_physicalize_all(raw); meanv=phys.mean(axis=0);meanv/=meanv.sum()
    base=[metrics(meanv,A) for A in range(1,7)]
    boots=[[metrics(v,A) for A in range(1,7)] for v in phys]
    pd.DataFrame(base).to_csv(out/'ibm_xxz_z_metrics.csv',index=False)
    br=[]
    for b,ms in enumerate(boots):
        for m in ms:br.append({'bootstrap':b,**m})
    pd.DataFrame(br).to_csv(out/'ibm_xxz_z_native_bootstrap_metrics.csv',index=False)
    sr=[]
    for A in range(1,7):
        row={'A':A}
        for key in ['detprime_C','p0','I_obs','I_CST','R','R_lattice_diagnostic']:
            x=[ms[A-1][key] for ms in boots]
            row.update({f'{key}_mean':float(np.mean(x)),f'{key}_sd':float(np.std(x,ddof=1)),f'{key}_ci025':quant(x,.025),f'{key}_ci975':quant(x,.975)})
        sr.append(row)
    pd.DataFrame(sr).to_csv(out/'ibm_xxz_z_bootstrap_summary.csv',index=False)
    trend_rows=[]
    for key in ['R','R_lattice_diagnostic']:
        for even in [False,True]:
            fs=[fit(ms,key,even) for ms in boots]; point=fit(base,key,even)
            slog=[x['slope_logA'] for x in fs]; sA=[x['slope_A'] for x in fs]
            trend_rows.append({'residual':key,'range':'A=2,4,6' if even else 'A=2..6','point_slope_logA':point['slope_logA'],
                'bootstrap_slope_logA_mean':float(np.mean(slog)),'bootstrap_slope_logA_ci025':quant(slog,.025),'bootstrap_slope_logA_ci975':quant(slog,.975),
                'point_slope_A':point['slope_A'],'bootstrap_slope_A_mean':float(np.mean(sA)),'bootstrap_slope_A_ci025':quant(sA,.025),'bootstrap_slope_A_ci975':quant(sA,.975),
                'falsified_by_positive_log_slope_95pct':bool(quant(slog,.025)>0)})
    pd.DataFrame(trend_rows).to_csv(out/'ibm_xxz_z_residual_trends.csv',index=False)
    pd.DataFrame({'bootstrap':np.arange(len(min_neg)),'most_negative_half_filled':min_neg}).to_csv(out/'paper_cutoff_inputs.csv',index=False)
    ghz(out)
    manifest={'zenodo_record':RECORD_ID,'source_file':str(src.relative_to(root)),'raw_shape':list(raw.shape),'paper_global_cutoff':cutoff,
        'global_sector':'Hamming weight 6','windows':'average over all 12 cyclic connected windows for each A',
        'primary_formula':'I_CST=0.5 log detprime(C)-log(V)+r/2 log(2pi)','primary_sizes':[2,3,4,5,6],
        'falsification_rule':'systematically growing R(A); operational flag here is lower 95% native-bootstrap CI of slope R vs log(A) > 0',
        'note':'lattice diagnostic includes Gaussian displacement of odd-A shell points and is secondary only'}
    (out/'run_manifest.json').write_text(json.dumps(manifest,indent=2))
    print('\nMETRICS');print(pd.DataFrame(base)[['A','V','detprime_C','p0','I_obs','I_CST','R','R_lattice_diagnostic']].to_string(index=False))
    print('\nTRENDS');print(pd.DataFrame(trend_rows).to_string(index=False))

if __name__=='__main__':main()
