#!/usr/bin/env python3
"""Locked repeated-validation analysis for the scientometric ranking manuscript."""
from __future__ import annotations
import argparse, hashlib, itertools, json, math, platform, sys
from pathlib import Path
import numpy as np, pandas as pd, scipy, shap, sklearn, xgboost as xgb
from joblib import Parallel, delayed
from scipy.stats import rankdata
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import ParameterSampler, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

ROOT=Path(__file__).resolve().parent; SOURCE=ROOT/'revision_20260827'; OUT=ROOT/'final_locked_results'; CK=OUT/'checkpoints'
MASTER_SEED=20260819; N_REPEATS=10; N_OUTER=5; N_INNER=3; N_CONFIGS=12
CUTOFFS=(.10,.20,.30,.40,.50); PREVALENCES=(.05,.10,.20,.30,.50); FAMILY_CUTS=(.05,.10,.15,.20)
AGG_FUNCS=('Arithmetic','Harmonic','Geometric','Quadratic','Cubic','Contra-harmonic','Logarithmic')

def save(df,name): df.to_csv(OUT/name,index=False)
def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def top(y,s,p=.1):
 y=np.asarray(y,float);s=np.asarray(s,float);k=math.ceil(len(y)*p);o=np.argsort(-s,kind='stable');z=s[o]
 starts=np.r_[0,np.flatnonzero(z[1:]!=z[:-1])+1];counts=np.diff(np.r_[starts,len(y)])
 rel=np.repeat(np.add.reduceat(y[o],starts)/counts,counts)[:k];disc=1/np.log2(np.arange(2,k+2));ideal=disc[:min(k,int(y.sum()))].sum()
 return {'NDCG':float(rel@disc/ideal),'Precision':float(rel.mean()),'Recall':float(rel.sum()/y.sum()),'k':k,'ties':int(len(s)-len(np.unique(s)))}
def metrics(y,s):
 d={'ROC-AUC':float(roc_auc_score(y,s)),'AP':float(average_precision_score(y,s))}
 for p in CUTOFFS:
  z=top(y,s,p);q=int(p*100);d.update({f'NDCG{q}':z['NDCG'],f'Precision{q}':z['Precision'],f'Recall{q}':z['Recall']})
 d['tie_count']=int(len(s)-len(np.unique(s)));return d
def fold_percentile(s,fold):
 s=np.asarray(s,float);out=np.empty(len(s))
 for f in range(1,6):
  ix=np.flatnonzero(fold==f);out[ix]=(rankdata(s[ix],method='average')-1)/(len(ix)-1)
 return out
def ecdf(train,v):
 t=np.sort(np.asarray(train,float));return np.searchsorted(t,np.asarray(v,float),side='right')/len(t)
def agg(a,b,n):
 a=np.asarray(a,float);b=np.asarray(b,float)
 if n=='Arithmetic':return (a+b)/2
 if n=='Harmonic':return np.divide(2*a*b,a+b,out=np.zeros_like(a),where=(a+b)!=0)
 if n=='Geometric':return np.sqrt(a*b)
 if n=='Quadratic':return np.sqrt((a*a+b*b)/2)
 if n=='Cubic':return np.cbrt((a**3+b**3)/2)
 if n=='Contra-harmonic':return np.divide(a*a+b*b,a+b,out=np.zeros_like(a),where=(a+b)!=0)
 out=np.zeros_like(a);same=np.isclose(a,b,rtol=1e-10,atol=1e-14);out[same]=(a[same]+b[same])/2;valid=(~same)&(a>0)&(b>0);out[valid]=(a[valid]-b[valid])/(np.log(a[valid])-np.log(b[valid]));return out
def rank_indicators(X,y):
 rows=[]
 for c in X.columns:
  z=top(y,X[c],.1);rows.append((c,z['NDCG'],z['Precision']))
 return [r[0] for r in sorted(rows,key=lambda x:(-x[1],-x[2],x[0]))]
def select5(X,y):
 order=rank_indicators(X,y);corr=X.corr(method='spearman').abs();chosen=[]
 for c in order:
  if all(corr.loc[c,d]<.90 for d in chosen):chosen.append(c)
  if len(chosen)==5:break
 return chosen
def composite(Xtr,ytr,Xte,L=10,pair=.70,red=.90):
 order=rank_indicators(Xtr,ytr)[:L];corr=Xtr.corr(method='spearman').abs();kept=[]
 for c in order:
  if all(corr.loc[c,d]<red for d in kept):kept.append(c)
 cand=[]
 for a,b in itertools.combinations(kept,2):
  if corr.loc[a,b]<=pair:
   pa,pb=ecdf(Xtr[a],Xtr[a]),ecdf(Xtr[b],Xtr[b])
   for fn in AGG_FUNCS:
    z=top(ytr,agg(pa,pb,fn),.1);cand.append((z['NDCG'],z['Precision'],a,b,fn))
 if not cand:raise RuntimeError('no composite candidate')
 best=sorted(cand,key=lambda z:(-z[0],-z[1],z[2],z[3],z[4]))[0];_,_,a,b,fn=best
 s=agg(ecdf(Xtr[a],Xte[a]),ecdf(Xtr[b],Xte[b]),fn)
 return s,{'a':a,'b':b,'function':fn,'train_NDCG10':best[0],'train_Precision10':best[1],'train_rho':float(corr.loc[a,b]),'eligible':len(cand)}
def panel(Xtr,ytr,Xte):
 cols=select5(Xtr,ytr);return np.mean([ecdf(Xtr[c],Xte[c]) for c in cols],axis=0),cols
def grids():
 lr=[{'C':float(c)} for c in np.logspace(-3,2.5,12)]
 rf=[{'n_estimators':n,'max_features':mf,'min_samples_leaf':leaf,'max_depth':None} for n in (300,500,800) for mf in ('sqrt',.5) for leaf in (1,3)]
 hgb=[{'max_iter':300,'learning_rate':r,'max_leaf_nodes':l,'l2_regularization':z} for r in (.03,.05,.08) for l in (7,15) for z in (0.,1.)]
 xp=list(ParameterSampler({'n_estimators':[250,400,600,800],'max_depth':[2,3,4,5],'learning_rate':[.02,.04,.06,.10],'subsample':[.70,.85,1.],'colsample_bytree':[.60,.75,.90,1.],'min_child_weight':[1,3,5,8],'reg_alpha':[0.,.05,.20,.50],'reg_lambda':[.5,1.,2.,5.],'gamma':[0.,.05,.20]},n_iter=12,random_state=MASTER_SEED))
 return {'LR':lr,'RF':rf,'HGB':hgb,'XGB':xp}
def model(name,p,seed):
 if name=='LR':return LogisticRegression(C=p['C'],max_iter=5000,solver='lbfgs',class_weight=None,random_state=seed)
 if name=='RF':return RandomForestClassifier(**p,class_weight=None,random_state=seed,n_jobs=1)
 if name=='HGB':return HistGradientBoostingClassifier(**p,random_state=seed)
 return xgb.XGBClassifier(**p,objective='binary:logistic',eval_metric='logloss',tree_method='hist',scale_pos_weight=1,random_state=seed,n_jobs=1,verbosity=0)
def prep(name,A,B):
 imp=SimpleImputer(strategy='median');a=imp.fit_transform(A);b=imp.transform(B);sc=None
 if name=='LR':sc=StandardScaler();a=sc.fit_transform(a);b=sc.transform(b)
 return a,b,imp,sc
def inner_score(name,p,X,y,g,splits,seed,five=False):
 vals=[]
 for j,(tr,va) in enumerate(splits,1):
  cols=select5(X.iloc[tr],y[tr]) if five else list(X.columns);a,b,_,_=prep(name,X.iloc[tr][cols],X.iloc[va][cols]);m=model(name,p,seed+j);m.fit(a,y[tr]);vals.append(top(y[va],m.predict_proba(b)[:,1],.1)['NDCG'])
 return float(np.mean(vals))
def tune(name,grid,Xtr,ytr,gtr,Xte,seed,five=False):
 splits=list(StratifiedGroupKFold(3,shuffle=True,random_state=seed).split(Xtr,ytr,gtr))
 vals=Parallel(n_jobs=6)(delayed(inner_score)(name,p,Xtr,ytr,gtr,splits,seed*100+i*10,five) for i,p in enumerate(grid));chosen=sorted(range(12),key=lambda i:(-vals[i],i))[0]
 cols=select5(Xtr,ytr) if five else list(Xtr.columns);a,b,imp,sc=prep(name,Xtr[cols],Xte[cols]);m=model(name,grid[chosen],seed);m.fit(a,ytr)
 return m.predict_proba(b)[:,1],m,imp,sc,cols,chosen,vals
def assignments(data,fixed):
 y=data.Class.to_numpy();g=fixed.identity_group.to_numpy();out=[fixed.outer_fold.to_numpy(int)]
 for r in range(1,10):
  f=np.zeros(len(y),int);cv=StratifiedGroupKFold(5,shuffle=True,random_state=MASTER_SEED+r)
  for k,(_,te) in enumerate(cv.split(data,y,g),1):f[te]=k
  out.append(f)
 return out
def repeat_run(r,fold,X,y,g,gs):
 methods=('LR','RF','HGB','XGB');scores={m:np.full(len(y),np.nan) for m in methods};scores.update({'Composite':np.full(len(y),np.nan),'Panel5':np.full(len(y),np.nan),'HGB5':np.full(len(y),np.nan)})
 configs=[];inners=[];comps=[];panels=[];refs={}
 for f in range(1,6):
  tr=np.flatnonzero(fold!=f);te=np.flatnonzero(fold==f);seed=MASTER_SEED+r*100+f
  assert not set(g[tr])&set(g[te]);assert len(np.unique(y[tr]))==2 and len(np.unique(y[te]))==2
  s,meta=composite(X.iloc[tr],y[tr],X.iloc[te]);scores['Composite'][te]=s;comps.append({'repeat':r,'fold':f,**meta})
  s,cols=panel(X.iloc[tr],y[tr],X.iloc[te]);scores['Panel5'][te]=s;panels.append({'repeat':r,'fold':f,'features':' | '.join(cols)})
  for name in methods:
   s,m,imp,sc,cols,ch,vals=tune(name,gs[name],X.iloc[tr],y[tr],g[tr],X.iloc[te],seed);scores[name][te]=s
   configs.append({'repeat':r,'fold':f,'model':name,'candidate':ch+1,'inner_NDCG10':vals[ch],'parameters':json.dumps(gs[name][ch],sort_keys=True,default=str)})
   inners += [{'repeat':r,'fold':f,'model':name,'candidate':i+1,'mean_inner_NDCG10':v,'parameters':json.dumps(gs[name][i],sort_keys=True,default=str)} for i,v in enumerate(vals)]
   if r==1:refs[(name,f)]=(m,imp,sc,cols,te)
  s,m,imp,sc,cols,ch,vals=tune('HGB',gs['HGB'],X.iloc[tr],y[tr],g[tr],X.iloc[te],seed+50000,True);scores['HGB5'][te]=s
  configs.append({'repeat':r,'fold':f,'model':'HGB5','candidate':ch+1,'inner_NDCG10':vals[ch],'parameters':json.dumps(gs['HGB'][ch],sort_keys=True),'features':' | '.join(cols)})
  if r==1:refs[('HGB5',f)]=(m,imp,sc,cols,te)
  print(f'repeat {r}/10 fold {f}/5',flush=True)
 score_rows=[];perf=[]
 for name,s in scores.items():
  assert np.isfinite(s).all();norm=fold_percentile(s,fold)
  score_rows += [{'repeat':r,'record':i,'fold':int(fold[i]),'method':name,'raw_score':s[i],'normalized_score':norm[i]} for i in range(len(y))]
  perf.append({'repeat':r,'method':name,**metrics(y,norm)})
  for f in range(1,6):
   ix=fold==f;perf.append({'repeat':r,'method':name,'fold':f,**metrics(y[ix],s[ix])})
 pd.DataFrame(score_rows).to_csv(CK/f'repeat_{r:02d}_scores.csv',index=False);pd.DataFrame(perf).to_csv(CK/f'repeat_{r:02d}_performance.csv',index=False);pd.DataFrame(configs).to_csv(CK/f'repeat_{r:02d}_configs.csv',index=False);pd.DataFrame(inners).to_csv(CK/f'repeat_{r:02d}_inner.csv',index=False);pd.DataFrame(comps).to_csv(CK/f'repeat_{r:02d}_composite.csv',index=False);pd.DataFrame(panels).to_csv(CK/f'repeat_{r:02d}_panel.csv',index=False)
 (CK/f'repeat_{r:02d}.done').write_text('complete\n');return refs
def fit_all(data,fixed):
 X=data.drop(columns=['Author Name','Class']);y=data.Class.to_numpy(int);g=fixed.identity_group.to_numpy();folds=assignments(data,fixed);gs=grids();refs={}
 for r,fold in enumerate(folds,1):
  if (CK/f'repeat_{r:02d}.done').exists():print(f'repeat {r}/10 checkpoint found',flush=True);continue
  q=repeat_run(r,fold,X,y,g,gs)
  if q:refs=q
 kinds={'scores':'01_repeated_oof_scores.csv','performance':'02_repeated_performance.csv','configs':'03_selected_model_configurations.csv','inner':'04_inner_search_results.csv','composite':'05_composite_selections.csv','panel':'06_panel5_selections.csv'}
 for k,name in kinds.items():save(pd.concat([pd.read_csv(CK/f'repeat_{r:02d}_{k}.csv') for r in range(1,11)],ignore_index=True),name)
 return X,y,g,gs
def summarize(y):
 s=pd.read_csv(OUT/'01_repeated_oof_scores.csv');rows=[];wide={'record':np.arange(len(y)),'Class':y}
 for m in s.method.unique():
  z=s[s.method==m].pivot(index='record',columns='repeat',values='normalized_score').sort_index().mean(axis=1).to_numpy();wide[m]=z;rows.append({'method':m,**metrics(y,z)})
 save(pd.DataFrame(rows).sort_values('NDCG10',ascending=False),'07_primary_consensus_performance.csv');save(pd.DataFrame(wide),'08_primary_consensus_scores.csv')
 p=pd.read_csv(OUT/'02_repeated_performance.csv');p=p[p['fold'].isna()];z=p.groupby('method').agg({c:['mean','std','min','max'] for c in ['NDCG10','Precision10','Recall10','ROC-AUC','AP']});z.columns=[f'{a}_{b}' for a,b in z.columns];save(z.reset_index(),'09_repeat_summary.csv')
def individual(X,y,g,B=2000):
 rows=[{'indicator':c,**metrics(y,X[c]),'rank_biserial':2*roc_auc_score(y,X[c])-1} for c in X];gv,inv=np.unique(g,return_inverse=True);by=[np.flatnonzero(inv==i) for i in range(len(gv))];strata={}
 for i,ix in enumerate(by):strata.setdefault((len(ix),int(y[ix[0]])),[]).append(i)
 rng=np.random.default_rng(MASTER_SEED+8000);a=X.to_numpy();boot=np.empty((B,X.shape[1]))
 for b in range(B):
  ix=np.concatenate([by[i] for q in strata.values() for i in rng.choice(q,len(q),replace=True)]);yy=y[ix];n1=yy.sum();n0=len(yy)-n1;r=rankdata(a[ix],axis=0);U=r[yy==1].sum(0)-n1*(n1+1)/2;boot[b]=2*U/(n1*n0)-1
 ci=np.quantile(boot,[.025,.975],axis=0);d=pd.DataFrame(rows);d['ci_low']=ci[0];d['ci_high']=ci[1];save(d.sort_values(['NDCG10','Precision10','indicator'],ascending=[False,False,True]),'10_individual_indicator_results.csv')
def resample_structure(y,g):
 gv,inv=np.unique(g,return_inverse=True);by=[np.flatnonzero(inv==i) for i in range(len(gv))];strata={}
 for i,ix in enumerate(by):strata.setdefault((len(ix),int(y[ix[0]])),[]).append(i)
 return gv,inv,by,strata
def holm(d):
 o=np.argsort(d.raw_p);a=np.empty(len(d));a[o]=np.minimum(1,np.maximum.accumulate(d.raw_p.to_numpy()[o]*(len(d)-np.arange(len(d)))));d['holm_p']=a;return d
def inference(y,g):
 s=pd.read_csv(OUT/'08_primary_consensus_scores.csv');methods=['LR','RF','HGB','XGB','Composite'];pairs=list(itertools.combinations(methods,2));gv,inv,by,strata=resample_structure(y,g);rb=np.random.default_rng(MASTER_SEED+9000);rp=np.random.default_rng(MASTER_SEED+9001);rows=[]
 for a,b in pairs:
  sa=s[a].to_numpy();sb=s[b].to_numpy();obs=top(y,sa,.1)['NDCG']-top(y,sb,.1)['NDCG'];bd=[]
  for _ in range(2000):
   ix=np.concatenate([by[i] for q in strata.values() for i in rb.choice(q,len(q),replace=True)]);bd.append(top(y[ix],sa[ix],.1)['NDCG']-top(y[ix],sb[ix],.1)['NDCG'])
  ext=0
  for _ in range(10000):
   sw=(rp.random(len(gv))<.5)[inv];u=np.where(sw,sb,sa);v=np.where(sw,sa,sb);ext+=abs(top(y,u,.1)['NDCG']-top(y,v,.1)['NDCG'])>=abs(obs)-1e-15
  rows.append({'method_a':a,'method_b':b,'cutoff':10,'difference':obs,'ci_low':np.quantile(bd,.025),'ci_high':np.quantile(bd,.975),'raw_p':(ext+1)/10001})
 save(holm(pd.DataFrame(rows)),'11_primary_pairwise_inference.csv');rows=[]
 for a,b in pairs:
  sa=s[a].to_numpy();sb=s[b].to_numpy()
  for p in CUTOFFS[1:]:
   obs=top(y,sa,p)['NDCG']-top(y,sb,p)['NDCG'];ext=0
   for _ in range(10000):
    sw=(rp.random(len(gv))<.5)[inv];u=np.where(sw,sb,sa);v=np.where(sw,sa,sb);ext+=abs(top(y,u,p)['NDCG']-top(y,v,p)['NDCG'])>=abs(obs)-1e-15
   rows.append({'method_a':a,'method_b':b,'cutoff':int(p*100),'difference':obs,'raw_p':(ext+1)/10001})
 save(holm(pd.DataFrame(rows)),'12_secondary_pairwise_inference.csv')
def prevalence(y):
 s=pd.read_csv(OUT/'08_primary_consensus_scores.csv');pos=np.flatnonzero(y==1);neg=np.flatnonzero(y==0);rng=np.random.default_rng(MASTER_SEED+10000);rows=[]
 for m in ['LR','RF','HGB','XGB']:
  for p in PREVALENCES:
   np1=min(len(pos),round(p/(1-p)*len(neg)));vals=[]
   for _ in range(1000):
    ix=np.r_[neg,rng.choice(pos,np1,replace=False)];vals.append(metrics(y[ix],s[m].to_numpy()[ix]))
   d=pd.DataFrame(vals);r={'method':m,'target_prevalence':p,'realized_prevalence':np1/(np1+len(neg)),'n_awardee':np1,'n_comparison':len(neg)}
   for c in ['NDCG10','Precision10','Recall10','ROC-AUC','AP']:r.update({c:d[c].mean(),c+'_low':d[c].quantile(.025),c+'_high':d[c].quantile(.975)})
   rows.append(r)
 save(pd.DataFrame(rows),'13_prevalence_all_models.csv')
def raw_sensitivity(y):
 s=pd.read_csv(OUT/'01_repeated_oof_scores.csv');s=s[s.repeat==1].pivot(index='record',columns='method',values='raw_score').sort_index();save(pd.DataFrame([{'method':m,**metrics(y,s[m])} for m in s]),'18_fixed_fold_raw_score_sensitivity.csv')
def manifest(gs):
 d={'status':'FROZEN','master_seed':MASTER_SEED,'outer_repeats':10,'outer_folds':5,'inner_folds':3,'configurations_per_model':12,'primary_score':'mean within-outer-fold percentile-normalized held-out score across ten repeats','class_weighting':'none for every model','cutoffs':CUTOFFS,'prevalences':PREVALENCES,'family_cuts':FAMILY_CUTS,'grids':gs,'versions':{'python':sys.version,'platform':platform.platform(),'numpy':np.__version__,'pandas':pd.__version__,'scipy':scipy.__version__,'sklearn':sklearn.__version__,'xgboost':xgb.__version__,'shap':shap.__version__},'source_hashes':{'analysis_matrix.csv':sha(SOURCE/'analysis_matrix.csv'),'fixed_folds.csv':sha(SOURCE/'fixed_folds.csv')}}
 (OUT/'19_frozen_manifest.json').write_text(json.dumps(d,indent=2,default=str))
def main():
 OUT.mkdir(exist_ok=True);CK.mkdir(exist_ok=True);data=pd.read_csv(SOURCE/'analysis_matrix.csv');fixed=pd.read_csv(SOURCE/'fixed_folds.csv');assert len(data)==1184 and data.Class.sum()==609 and data.shape[1]==66 and fixed.identity_group.nunique()==1154;assert not fixed.groupby('identity_group').outer_fold.nunique().gt(1).any();assert np.isfinite(data.drop(columns=['Author Name','Class']).to_numpy()).all()
 X,y,g,gs=fit_all(data,fixed);summarize(y);individual(X,y,g);inference(y,g);prevalence(y);raw_sensitivity(y);manifest(gs);print('CORE PIPELINE COMPLETE; RUN compute_shap_direct.py NEXT',flush=True)
if __name__=='__main__':main()
