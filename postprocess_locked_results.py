#!/usr/bin/env python3
import itertools
from pathlib import Path
import numpy as np,pandas as pd
import final_locked_analysis_pipeline as c
OUT=c.OUT
def composite_sensitivity():
 d=pd.read_csv(c.SOURCE/'analysis_matrix.csv');fixed=pd.read_csv(c.SOURCE/'fixed_folds.csv');X=d.drop(columns=['Author Name','Class']);y=d.Class.to_numpy(int);folds=c.assignments(d,fixed);settings=list(itertools.product((8,10,12),(.60,.70,.80,.90),(.85,.90,.95)));sel=[];perf=[]
 for r,fold in enumerate(folds,1):
  for L,pair,red in settings:
   score=np.full(len(y),np.nan)
   for f in range(1,6):
    tr=np.flatnonzero(fold!=f);te=np.flatnonzero(fold==f);s,m=c.composite(X.iloc[tr],y[tr],X.iloc[te],L,pair,red);score[te]=s;sel.append({'repeat':r,'fold':f,'shortlist':L,'pair_rho':pair,'redundancy_rho':red,**m})
   perf.append({'repeat':r,'shortlist':L,'pair_rho':pair,'redundancy_rho':red,**c.metrics(y,c.fold_percentile(score,fold))})
 c.save(pd.DataFrame(sel),'20_composite_sensitivity_selections.csv');c.save(pd.DataFrame(perf),'21_composite_sensitivity_performance.csv')
def reduced():
 d=pd.read_csv(c.SOURCE/'analysis_matrix.csv');fixed=pd.read_csv(c.SOURCE/'fixed_folds.csv');y=d.Class.to_numpy(int);g=fixed.identity_group.to_numpy();s=pd.read_csv(OUT/'08_primary_consensus_scores.csv');a=s.HGB.to_numpy();b=s.HGB5.to_numpy();gv,inv,by,strata=c.resample_structure(y,g);rng=np.random.default_rng(c.MASTER_SEED+12000);obs=c.top(y,a,.1)['NDCG']-c.top(y,b,.1)['NDCG'];bd=[]
 for _ in range(2000):
  ix=np.concatenate([by[i] for q in strata.values() for i in rng.choice(q,len(q),replace=True)]);bd.append(c.top(y[ix],a[ix],.1)['NDCG']-c.top(y[ix],b[ix],.1)['NDCG'])
 ext=0
 for _ in range(10000):
  sw=(rng.random(len(gv))<.5)[inv];u=np.where(sw,b,a);v=np.where(sw,a,b);ext+=abs(c.top(y,u,.1)['NDCG']-c.top(y,v,.1)['NDCG'])>=abs(obs)-1e-15
 c.save(pd.DataFrame([{'comparison':'HGB64 minus HGB5','difference':obs,'ci_low':np.quantile(bd,.025),'ci_high':np.quantile(bd,.975),'randomization_p':(ext+1)/10001,'bootstrap_replicates':2000,'randomization_draws':10000}]),'22_reduced_representation_inference.csv')
def summaries():
 p=pd.read_csv(OUT/'06_panel5_selections.csv');counts={}
 for z in p.features:
  for q in z.split(' | '):counts[q]=counts.get(q,0)+1
 pd.DataFrame([{'indicator':k,'selected_folds':v,'selection_frequency':v/len(p)} for k,v in counts.items()]).sort_values(['selected_folds','indicator'],ascending=[False,True]).to_csv(OUT/'23_panel5_feature_frequencies.csv',index=False)
 s=pd.read_csv(OUT/'20_composite_sensitivity_selections.csv');s['specification']=s.a+' + '+s.b+' | '+s.function;z=s.groupby('specification').size().rename('count').reset_index();z['frequency']=z['count']/len(s);z.sort_values(['count','specification'],ascending=[False,True]).to_csv(OUT/'24_composite_specification_frequencies.csv',index=False)
def main():composite_sensitivity();reduced();summaries();print('ADDITIONAL ANALYSES COMPLETE',flush=True)
if __name__=='__main__':main()
