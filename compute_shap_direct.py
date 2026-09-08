#!/usr/bin/env python3
"""Reconstruct archived-fold tree models from frozen selections and compute SHAP directly."""
import json
import numpy as np,pandas as pd,shap
from scipy.cluster.hierarchy import fcluster,linkage
from scipy.spatial.distance import squareform
import final_locked_analysis_pipeline as c
def main():
 d=pd.read_csv(c.SOURCE/'analysis_matrix.csv');fixed=pd.read_csv(c.SOURCE/'fixed_folds.csv');X=d.drop(columns=['Author Name','Class']);y=d.Class.to_numpy(int);g=fixed.identity_group.to_numpy();fold=fixed.outer_fold.to_numpy(int);cfg=pd.read_csv(c.CK/'repeat_01_configs.csv');arrays={m:np.zeros(X.shape) for m in ['HGB','RF','XGB']};recon=[]
 for name in arrays:
  for f in range(1,6):
   tr=np.flatnonzero(fold!=f);te=np.flatnonzero(fold==f);row=cfg[(cfg.model==name)&(cfg.fold==f)].iloc[0];p=json.loads(row.parameters);a,b,imp,sc=c.prep(name,X.iloc[tr],X.iloc[te]);m=c.model(name,p,c.MASTER_SEED+100+f);m.fit(a,y[tr]);ex=shap.TreeExplainer(m);v=np.asarray(ex.shap_values(b));base=np.asarray(ex.expected_value)
   if v.ndim==3:v=v[:,:,-1];base=base[-1]
   arrays[name][te]=v;base=float(base.reshape(-1)[0]);target=m.predict_proba(b)[:,1] if name=='RF' else (m.decision_function(b) if name=='HGB' else m.predict(b,output_margin=True));err=float(np.max(np.abs(v.sum(1)+base-target)));recon.append({'model':name,'fold':f,'max_abs_error':err});assert err<1e-4;print(name,f,err,flush=True)
 corr=X.corr(method='spearman').abs();dist=1-corr;np.fill_diagonal(dist.values,0);Z=linkage(squareform(dist.values,checks=False),method='average');np.save(c.OUT/'linkage.npy',Z);corr.to_csv(c.OUT/'absolute_spearman.csv');mr=[]
 for cut in c.FAMILY_CUTS:
  labels=fcluster(Z,cut,criterion='distance');mp={old:f'F{i:02d}' for i,old in enumerate(dict.fromkeys(labels),1)};mr += [{'cut_distance':cut,'equivalent_abs_rho':1-cut,'indicator':col,'family':mp[lab]} for col,lab in zip(X.columns,labels)]
 mem=pd.DataFrame(mr);c.save(mem,'14_family_membership_all_thresholds.csv');c.save(pd.DataFrame(recon),'15_shap_reconstruction.csv');ir=[];fr=[]
 for name,a in arrays.items():
  ir += [{'model':name,'indicator':col,'mean_abs_SHAP':ma,'mean_signed_SHAP':ms} for col,ma,ms in zip(X.columns,np.abs(a).mean(0),a.mean(0))]
  for cut in c.FAMILY_CUTS:
   sub=mem[mem.cut_distance==cut]
   for fam,q in sub.groupby('family'):
    ids=[list(X.columns).index(col) for col in q.indicator];v=np.abs(a[:,ids]).sum(1);fr.append({'model':name,'cut_distance':cut,'family':fam,'count':len(ids),'members':' | '.join(q.indicator),'mean_summed_abs_SHAP':v.mean(),'SD':v.std(ddof=1),'mean_per_indicator':v.mean()/len(ids)})
 c.save(pd.DataFrame(ir),'16_individual_treeshap.csv');c.save(pd.DataFrame(fr),'17_family_treeshap_all_thresholds.csv');c.raw_sensitivity(y);c.manifest(c.grids());print('DIRECT SHAP COMPLETE',flush=True)
if __name__=='__main__':main()
