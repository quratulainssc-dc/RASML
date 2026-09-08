#!/usr/bin/env python3
from pathlib import Path
import math,numpy as np,pandas as pd
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch,Rectangle,Circle,Ellipse
from scipy.cluster.hierarchy import dendrogram
R=Path(__file__).resolve().parent;O=R/'final_locked_results';F=R/'final_locked_figures';F.mkdir(exist_ok=True)
B='#23679E';DB='#174C78';LB='#EAF3FA';G='#27866B';DG='#17634E';LG='#EAF6F1';N='#667085';INK='#111827';GRID='#D7DEE7'
plt.rcParams.update({'font.family':'DejaVu Serif','font.size':9.2,'axes.labelsize':9.2,'axes.titlesize':10.2,'xtick.labelsize':8.2,'ytick.labelsize':8.2,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':600})
def nm(s):return {'h2upper_index':'h²-upper','h2 lower_index':'h²-lower','h2 center_index':'h²-center','Ar index':'Ar-index','A index':'A-index','P index':'P-index','gf index':'gf-index','gF index':'gF-index','Cite/Paper':'Citations/paper','Cite/Year':'Citations/year','Author/paper':'Authors/paper','platinium_h_index':'platinum h-index','m_qoutient_index':'m-quotient index'}.get(s,str(s).replace('_',' '))
def done(fig,n):fig.savefig(F/f'Figure_{n}.png',dpi=600,bbox_inches='tight',pad_inches=.06);plt.close(fig)
def fig1():
 fig,ax=plt.subplots(figsize=(6.85,8.35));ax.set(xlim=(0,100),ylim=(0,126));ax.axis('off')
 def box(x,y,w,h,head,body='',c=B,fc=LB,fs=9.2):
  ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.25,rounding_size=1.2',ec=c,fc=fc,lw=1.35));ax.text(x+w/2,y+h-1.7,head,ha='center',va='top',fontweight='bold',color=c,fontsize=fs)
  if body:ax.text(x+w/2,y+(h-4)/2,body,ha='center',va='center',color=INK,fontsize=fs-.6,linespacing=1.22)
 def ar(x,a,b):ax.add_patch(FancyArrowPatch((x,a),(x,b),arrowstyle='-|>',mutation_scale=11,lw=1.6,color=INK))
 box(1,112,98,12,'Existing Scientometric Matrix And Grouped Validation','1,184 Records • 1,154 Identity Groups • 64 Indicators\nHistorical Award Status As External Relevance Benchmark');ar(50,112,108.5)
 box(1,102,98,5,'Ten Repeated Five-Fold Outer Evaluations',c=G,fc=LG);ar(25,102,98.5);ar(75,102,98.5)
 box(1,42,47,56,'Statistical Aggregation',c=G,fc=LG,fs=9.5);box(52,42,47,56,'Machine Learning',fs=9.5)
 box(3,84,43,8,'Individual Indicator Evaluation','Ranking Metrics And Distributional Effects',G,'white',7.6);box(54,84,43,8,'Training-Fitted Preprocessing','Median Imputation • LR Standardization',B,'white',7.6);ar(24.5,84,81);ar(75.5,84,81)
 box(3,69,43,11,'Training Selection And Normalization','Shortlist • Redundancy Screening\nPair Screening • Percentile Mapping',G,'white',7.35);box(54,69,43,11,'Equal-Budget Model Development','LR • Random Forest • HGB • XGBoost\nThree Inner Folds • 12 Settings Per Model',B,'white',7.35);ar(24.5,69,66);ar(75.5,69,66)
 box(3,52,43,13,'Seven Aggregation Functions','Arithmetic • Harmonic • Geometric\nQuadratic • Cubic • Contra-Harmonic\nLogarithmic',G,'white',7.8);box(54,52,43,13,'Held-Out Scoring And Explanation','Within-Fold Percentile Scores\nFold And Repeat Performance\nTreeSHAP • Four Family Thresholds',B,'white',7.35);ar(24.5,52,49);ar(75.5,52,49)
 box(3,43.5,43,5,'Selected Composite Scores',c=G,fc='white',fs=7.6);box(54,43.5,43,5,'Model Scores And Attributions',c=B,fc='white',fs=7.6);ar(24.5,43,39);ar(75.5,43,39)
 box(1,20,98,18,'Common Comparative Analysis','Consensus Rankings From Ten Repeats • Raw-Score Sensitivity\nNDCG • Precision • Recall • ROC-AUC • Average Precision\nGroup Bootstrap • Paired Randomization • Holm Adjustment\nAwardee Prevalence: 5%, 10%, 20%, 30%, And 50%',G,LG);ar(50,20,16.5)
 box(1,1,98,15,'Comparative Results And Interpretation','Ranking Performance And Uncertainty\nComposite And Five-Indicator Selection Stability\nCorrelation-Family Attribution And Threshold Sensitivity');done(fig,1)
def fig2():
 d=pd.read_csv(R/'revision_20260827/analysis_matrix.csv');ind=pd.read_csv(O/'10_individual_indicator_results.csv');fig,axs=plt.subplots(2,2,figsize=(6.85,4.9))
 for ax,c in zip(axs.flat,ind.indicator.head(4)):
  z=np.sign(d[c])*np.log1p(abs(d[c]));e=np.histogram_bin_edges(z,bins=25)
  for q,col,lab in [(0,G,'Comparison'),(1,B,'Awardee')]:ax.hist(z[d.Class==q],bins=e,density=True,histtype='step',lw=1.6,color=col,label=lab)
  ax.set_title(nm(c),fontweight='bold');ax.set_xlabel('Signed Log-Transformed Value');ax.set_ylabel('Density')
 axs[0,0].legend(frameon=False);fig.tight_layout();done(fig,2)
def fig3():
 d=pd.read_csv(O/'10_individual_indicator_results.csv');fig,ax=plt.subplots(figsize=(6.85,8.8));v=d.NDCG10.to_numpy();y=np.arange(len(d));ax.scatter(v,y,c=v,cmap='Blues',s=24);ax.set_yticks(y,[nm(x) for x in d.indicator],fontsize=7.4);ax.invert_yaxis();ax.set_xlabel('NDCG At Top 10%');ax.tick_params(axis='y',length=0);ax.spines['left'].set_visible(False);ax.grid(axis='x',color=GRID,lw=.5);fig.tight_layout();done(fig,3)
def fig4():
 d=pd.read_csv(O/'21_composite_sensitivity_performance.csv').groupby(['shortlist','pair_rho','redundancy_rho'],as_index=False).NDCG10.mean();fig,axs=plt.subplots(1,3,figsize=(6.85,2.65),sharey=True);lo,hi=d.NDCG10.min(),d.NDCG10.max();lo,hi=(lo-.005,hi+.005) if math.isclose(lo,hi) else (lo,hi)
 for ax,r in zip(axs,(.85,.90,.95)):
  z=d[d.redundancy_rho==r].pivot(index='shortlist',columns='pair_rho',values='NDCG10');im=ax.imshow(z,cmap='Blues',vmin=lo,vmax=hi,aspect='auto');ax.set_xticks(range(4),['.60','.70','.80','.90']);ax.set_yticks(range(3),['8','10','12']);ax.set_title(f'Redundancy Limit {r:.2f}',fontweight='bold',fontsize=9);ax.set_xlabel('Pair-Correlation Limit')
  for i in range(3):
   for j in range(4):ax.text(j,i,f'{z.iloc[i,j]:.3f}',ha='center',va='center',fontsize=7,color='white' if z.iloc[i,j]>(lo+hi)/2 else INK)
 axs[0].set_ylabel('Shortlist Size');fig.subplots_adjust(left=.08,right=.9,bottom=.21,top=.84,wspace=.16);fig.colorbar(im,ax=axs,fraction=.025,pad=.02,label='Mean NDCG At Top 10%');done(fig,4)
def fig5():
 p=pd.read_csv(O/'07_primary_consensus_performance.csv').set_index('method');r=pd.read_csv(O/'09_repeat_summary.csv').set_index('method');methods=['LR','RF','HGB','XGB','Composite'];fig,axs=plt.subplots(1,3,figsize=(6.85,3))
 for ax,m,lab in zip(axs,['NDCG10','Precision10','ROC-AUC'],['NDCG At Top 10%','Precision At Top 10%','ROC-AUC']):ax.barh(methods,p.loc[methods,m],xerr=r.loc[methods,m+'_std'],color=[B,G,DB,DG,N],capsize=2);ax.invert_yaxis();ax.set_xlim(0,1);ax.set_xlabel(lab);ax.grid(axis='x',color=GRID,lw=.5)
 fig.tight_layout();done(fig,5)
def fig6():
 mem=pd.read_csv(O/'14_family_membership_all_thresholds.csv');mem=mem[np.isclose(mem.cut_distance,.1)];corr=pd.read_csv(O/'absolute_spearman.csv',index_col=0);fams=sorted(mem.family.unique());mat=np.zeros((len(fams),len(fams)))
 for i,a in enumerate(fams):
  for j,b in enumerate(fams):mat[i,j]=np.median(corr.loc[mem[mem.family==a].indicator,mem[mem.family==b].indicator].to_numpy())
 np.fill_diagonal(mat,np.nan);fig=plt.figure(figsize=(6.85,6.8));gs=fig.add_gridspec(2,2,height_ratios=[1,2.6],width_ratios=[4.5,1.2],hspace=.42,wspace=.28);ax=fig.add_subplot(gs[0,:]);dendrogram(np.load(O/'linkage.npy'),ax=ax,no_labels=True,color_threshold=.1,above_threshold_color=N);ax.axhline(.1,color=G,ls='--',lw=1.3);ax.set_ylabel('Distance 1 − |ρ|');ax.set_xlabel('64 Indicators In Clustering Order');ax.text(-.05,1.03,'(a)',transform=ax.transAxes,fontweight='bold');ax=fig.add_subplot(gs[1,0]);cm=plt.cm.Blues.copy();cm.set_bad('white');im=ax.imshow(mat,vmin=0,vmax=1,cmap=cm);ax.set_xticks(range(len(fams)),fams,rotation=90);ax.set_yticks(range(len(fams)),fams);ax.set_xlabel('Correlation Family');ax.set_ylabel('Correlation Family');ax.text(-.09,1.02,'(b)',transform=ax.transAxes,fontweight='bold');ax=fig.add_subplot(gs[1,1]);counts=mem.groupby('family').size().reindex(fams);ax.barh(fams,counts,color=[B if i%2==0 else G for i in range(len(fams))]);ax.invert_yaxis();ax.set_xlabel('Indicators (n)');ax.text(-.28,1.02,'(c)',transform=ax.transAxes,fontweight='bold');cb=fig.colorbar(im,ax=fig.axes,fraction=.018,pad=.02);cb.set_label('Median Absolute Spearman Correlation');done(fig,6)
def fig7():
 d=pd.read_csv(O/'17_family_treeshap_all_thresholds.csv');p=d[(d.model=='HGB')&np.isclose(d.cut_distance,.1)].nlargest(10,'mean_summed_abs_SHAP');fig,ax=plt.subplots(figsize=(6.85,3.4));ax.barh([f"{r.family} (n={r['count']})" for _,r in p.iterrows()],p.mean_summed_abs_SHAP,color=[B if i%2==0 else G for i in range(len(p))]);ax.invert_yaxis();ax.set_xlabel('Mean Summed Absolute SHAP Contribution');ax.set_title('HGB At Primary Family Cut 0.10',fontweight='bold');fig.tight_layout();done(fig,7)
def fig8():
 d=pd.read_csv(O/'13_prevalence_all_models.csv');fig,axs=plt.subplots(1,3,figsize=(6.85,2.75));colors={'LR':B,'RF':G,'HGB':DB,'XGB':DG}
 for ax,m,lab in zip(axs,['Precision10','NDCG10','AP'],['Precision At Top 10%','NDCG At Top 10%','Average Precision']):
  for q in colors:
   z=d[d.method==q].sort_values('target_prevalence');x=z.realized_prevalence*100;ax.plot(x,z[m],marker='o',ms=3,lw=1.3,label=q,color=colors[q]);ax.fill_between(x,z[m+'_low'],z[m+'_high'],color=colors[q],alpha=.08)
  ax.set_xlabel('Awardee Prevalence (%)');ax.set_ylabel(lab);ax.set_ylim(0,1);ax.set_xticks([5,10,20,30,50]);ax.grid(color=GRID,lw=.5)
 axs[0].legend(frameon=False,ncol=2,fontsize=7.4);fig.tight_layout();done(fig,8)
def fig9():
 p=pd.read_csv(O/'07_primary_consensus_performance.csv').set_index('method');i=pd.read_csv(O/'10_individual_indicator_results.csv').iloc[0];labs=[f'Leading Individual: {nm(i.indicator)}','Selected Two-Indicator Composite','Redesigned Five-Indicator Panel','HGB With Five Selected Indicators','HGB With All 64 Indicators'];v=[i.NDCG10,p.loc['Composite','NDCG10'],p.loc['Panel5','NDCG10'],p.loc['HGB5','NDCG10'],p.loc['HGB','NDCG10']];fig,ax=plt.subplots(figsize=(6.85,3.15));ax.barh(labs,v,color=[N,G,DG,B,DB]);ax.invert_yaxis();ax.set_xlim(0,1);ax.set_xlabel('NDCG At Top 10%');ax.grid(axis='x',color=GRID,lw=.5);fig.tight_layout();done(fig,9)
def main():
 for f in [fig1,fig2,fig3,fig4,fig5,fig6,fig7,fig8,fig9]:f()
 print('NINE FIGURES GENERATED')
if __name__=='__main__':main()
