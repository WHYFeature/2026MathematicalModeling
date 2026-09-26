"""Analyze saved outputs, preserve unchanged figures, and write scientific vector plots."""
from pathlib import Path
import sys, os, json, csv, hashlib, shutil, re, struct
from collections import Counter, defaultdict
R=Path(__file__).resolve().parents[1]; O=R/'analysis'; OLD=R/'results_analysis_20260924'
sys.path.insert(0,str(R/'.analysis_deps')); os.environ['MPLCONFIGDIR']=str(O/'qa/mpl')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import spearmanr
for d in ['figures','previews','tables','qa']:(O/d).mkdir(exist_ok=True,parents=True)
COL=['#D47764','#B5A15D','#469A91']; MC=['#447BA8','#D99555','#469A91']; INK='#253746'
MOD=['text','audio','vision']; MODEL=['baseline','robust_noaug','robust']; ML=['Baseline','Robust no aug.','Robust']; CL=['Negative','Neutral','Positive']
plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','DejaVu Serif'],'font.size':10,'axes.labelsize':11,'axes.titlesize':12,'axes.titleweight':'bold','axes.edgecolor':'#8B949C','axes.labelcolor':INK,'text.color':INK,'xtick.color':INK,'ytick.color':INK,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.65,'grid.color':'#DDE2E5','grid.linewidth':.5,'legend.frameon':False,'legend.fontsize':9,'lines.linewidth':1.6,'pdf.fonttype':42,'svg.fonttype':'none','savefig.facecolor':'white'})
SRC={}; SECTIONS=[]; CATALOG=[]; STATS={}
def reg(p):
 p=Path(p);SRC[str(p.relative_to(R))]=hashlib.sha256(p.read_bytes()).hexdigest();return p
def js(p):return json.loads(reg(R/p).read_text(encoding='utf-8-sig'))
def rows(p):
 with reg(R/p).open(encoding='utf-8-sig',newline='') as f:rr=list(csv.DictReader(f))
 for row in rr:
  for k,v in row.items():
   if k not in ['id','condition_id','video_id','clip_id']:
    try:row[k]=float(v)
    except (ValueError,TypeError):pass
 return rr
def tab(name,rr):
 if not rr:return
 with (O/'tables'/f'{name}.csv').open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
def fmt(x):return f'{x:.4f}' if isinstance(x,(float,np.floating)) else str(x)
def mdtable(rr,fields):
 return '\n'.join(['| '+' | '.join(fields)+' |','| '+' | '.join(['---']*len(fields))+' |']+['| '+' | '.join(fmt(r.get(k,'')) for k in fields)+' |' for r in rr])
def select(rr,**kw):return [r for r in rr if all(r[k]==v for k,v in kw.items())]
def vals(rr,k):return np.array([r[k] for r in rr],float)
def ms(rr,k):
 a=vals(rr,k);return float(a.mean()),float(a.std(ddof=1)) if len(a)>1 else 0.
def title(ax,t):ax.set_title(t,loc='left',pad=12);ax.grid(axis='y');ax.set_axisbelow(True)
def grid(n=1,m=2,size=(11,4.2)):return plt.subplots(n,m,figsize=size,layout='constrained',squeeze=False)
def heat(ax,x,xt,yt,cmap='Blues',vmin=None,vmax=None,dec=2):
 im=ax.pcolormesh(np.arange(x.shape[1]+1)-.5,np.arange(x.shape[0]+1)-.5,x,cmap=cmap,vmin=vmin,vmax=vmax,rasterized=False,shading='flat');ax.set_ylim(x.shape[0]-.5,-.5)
 ax.set_xticks(range(len(xt)),xt);ax.set_yticks(range(len(yt)),yt)
 for i in range(len(yt)):
  for j in range(len(xt)):
   ax.text(j,i,f'{x[i,j]:.{dec}f}',ha='center',va='center',fontsize=8,color='white' if im.norm(x[i,j])>.65 else INK)
 cb=ax.figure.colorbar(im,ax=ax,shrink=.8);cb.solids.set_rasterized(False)
 if hasattr(cmap,'colors') and len(cmap.colors)==3:cb.set_ticks([0,1,2],labels=CL)
 return im
PDF=PdfPages(O/'qa/new_figures.pdf')
def save(fig,num,slug,cn,source,body,rr=None,fields=None):
 name=f'{num:02d}_{slug}';fig.savefig(O/'figures'/f'{name}.svg',bbox_inches='tight');fig.savefig(O/'previews'/f'{name}.png',dpi=160,bbox_inches='tight');PDF.savefig(fig,bbox_inches='tight');plt.close(fig)
 CATALOG.append({'number':num,'slug':name,'title':cn,'status':'new','sources':source})
 t=f'\n## 图{num} {cn}\n\n![图{num} {cn}](figures/{name}.svg)\n\n**数据来源与统计口径：**{source}\n\n'+body+'\n\n'
 if rr:
  tab(name,rr);t+=mdtable(rr,fields or list(rr[0]))+'\n\n'
 SECTIONS.append(t);print('FIG',num,cn,flush=True)

# Move exact originals, retaining byte-identical provenance. Safe to rerun.
diff=js('analysis/source_comparison.json');assert all(x['status']=='unchanged' for x in diff)
old=js('results_analysis_20260924/analysis.json')
with (OLD/'tables/figure_catalog.csv').open(encoding='utf-8-sig') as f:oldcat=list(csv.DictReader(f))
for c in oldcat:
 for folder,ext in [('figures','svg'),('previews','png')]:
  src=OLD/folder/f"{c['slug']}.{ext}";dest=O/folder/src.name
  if src.exists():
   if dest.exists():assert hashlib.sha256(src.read_bytes()).digest()==hashlib.sha256(dest.read_bytes()).digest()
   else:shutil.move(str(src),str(dest))
 CATALOG.append({'number':int(c['number']),'slug':c['slug'],'title':c['title'],'status':'unchanged_moved','sources':'上次15个源文件全部SHA-256一致；详见source_comparison.json'})
for p in (OLD/'tables').glob('*.csv'):
 if p.name!='figure_catalog.csv':shutil.copy2(p,O/'tables'/p.name)
# All analysis inputs are saved outputs. No model execution.
C=rows('problem2_experiment2/condition_metrics.csv'); A3=rows('problem2_experiment2/attachment3_predictions.csv'); cfg=js('problem2_experiment2/experiment_config.json')
assert len(C)==3*3*2*58
assert len({(r['model'],r['seed'],r['split'],r['condition_id']) for r in C})==len(C)
assert all(r['valid_sample_count']==(728 if r['split']=='valid' else 727) for r in C)
for m in MODEL:
 for s in [42,43,44]:
  for sp in ['valid','test']:
   one=select(C,model=m,seed=s,split=sp);b=select(one,group='baseline')[0];a=select(one,group='ablation',missing_modalities='')[0]
   assert all(abs(b[k]-a[k])<1e-10 for k in ['accuracy','macro_f1','mae'])
AGG=[]
for key in sorted({(r['model'],r['split'],r['condition_id']) for r in C}):
 rr=select(C,model=key[0],split=key[1],condition_id=key[2]);q={k:rr[0][k] for k in ['model','split','condition_id','group','missing_modalities','requested_rate','position','duration']}
 for k in ['accuracy','macro_f1','mae','rmse','pearson','f1_negative','f1_neutral','f1_positive','effective_missing_rate']:
  q[k+'_mean'],q[k+'_sd']=ms(rr,k)
 AGG.append(q)
tab('controlled_condition_statistics',AGG)
STATS['controlled_dimensions']={'rows':len(C),'conditions_per_run':58,'unique_conditions_excluding_complete_duplicate':57,'seeds':[42,43,44]}

# 21 design coverage
fig,ax=grid(size=(11,4));groups=['baseline','rate','position','duration','ablation'];ct=[len(select(C,model='baseline',seed=42,split='valid',group=g)) for g in groups]
ax[0,0].bar(groups,ct,color=MC[0]);title(ax[0,0],'(a) Conditions per model and seed');ax[0,0].set_ylabel('Condition count')
for i,n in enumerate(ct):ax[0,0].text(i,n+.3,str(n),ha='center')
heat(ax[0,1],np.array([[len(select(C,model=m,seed=s)) for s in [42,43,44]] for m in MODEL]),['42','43','44'],ML,vmin=0,vmax=116,dec=0);title(ax[0,1],'(b) Recorded rows across both splits')
save(fig,21,'experiment_coverage','新增鲁棒性实验的覆盖范围','condition_metrics.csv，1044行；3模型×3种子×2划分×58条件。','每个模型与种子都有58个条件：完整输入1个、缺失率28个、位置12个、时长9个、全模态消融8个。验证和测试分别使用728、727条带标签样本。图中每格116行是条件数乘以两个划分，不是116次独立训练。\n\n消融中还含一个完整输入对照，与baseline组重复，核对指标一致；汇总所有条件时将其去重，得到57个独特条件，其中56个为缺失条件。缺失率、位置和时长采用分组设计，并非全部因素的完整笛卡尔积，不能据此估计任意高阶交互。\n\n相较上次结果，当前已具备分析局部缺失规律和训练随机性的数据依据。误差线统一使用3个种子的样本标准差（ddof=1），不称为95%置信区间；同一种子下不同条件也不能当作独立样本扩大显著性。')

# 22 complete input seeds
fig,ax=grid(2,2,(11,7));complete=select(C,group='baseline');rr=[]
for a,k,label in zip(ax.flat,['accuracy','macro_f1','mae','pearson'],['Accuracy','Macro-F1','MAE','Pearson r']):
 for j,sp in enumerate(['valid','test']):
  for i,m in enumerate(MODEL):
   vv=vals(select(complete,split=sp,model=m),k);x=i+(-.15 if j==0 else .15);a.errorbar(x,vv.mean(),yerr=vv.std(ddof=1),fmt='o' if j==0 else 's',color=MC[i],capsize=4);a.scatter(x+np.linspace(-.045,.045,3),vv,s=16,color=MC[i],alpha=.55)
 a.set_xticks(range(3),['Base','No aug.','Robust']);title(a,label+' (circle: valid; square: test)')
for sp in ['valid','test']:
 for m in MODEL:
  q={'split':sp,'model':m}
  for k in ['accuracy','macro_f1','mae','pearson']:q[k],q[k+'_sd']=ms(select(complete,split=sp,model=m),k)
  rr.append(q)
save(fig,22,'complete_seed_stability','完整输入下的多种子性能与稳定性','完整输入，3个种子；点为单次训练，误差线为种子样本标准差。','本图将单次最优结果与跨种子表现分开。旧参考模型对应baseline的seed 42，不能把其71.80%测试准确率当作baseline跨种子均值。下表同时列出两个划分的四项赛题指标。\n\n比较Robust与No aug.可以观察加入训练缺失增强后的变化，比较No aug.与Baseline反映可用性感知架构及相关配置的整体差异。完整输入性能的提升或下降只说明完整输入代价；是否获得鲁棒性，需要结合后续缺失条件。三种子足以提供初步波动范围，但不足以证明普遍稳定。',rr)

# 23-26 rates, all combinations, valid/test metrics
combos=['text','audio','vision','text+audio','text+vision','audio+vision','text+audio+vision']
short=lambda x:x.replace('text','T').replace('audio','A').replace('vision','V') or 'None'
for num,sp,k,lab in [(23,'valid','macro_f1','Macro-F1'),(24,'test','macro_f1','Macro-F1'),(25,'valid','mae','MAE'),(26,'test','mae','MAE')]:
 fig,ax=grid(2,4,(14,7));summary=[]
 for a,co in zip(ax.flat,combos):
  for m,c,label in zip(MODEL,MC,ML):
   x=[0,.1,.2,.3,.4];means=[];sds=[]
   for rate in x:
    r=select(C,split=sp,model=m,group='baseline') if rate==0 else select(C,split=sp,model=m,group='rate',missing_modalities=co,requested_rate=rate)
    mu,sd=ms(r,k);means.append(mu);sds.append(sd)
   a.errorbar(x,means,yerr=sds,color=c,marker='o',ms=3,capsize=2,label=label)
   summary.append({'modalities':short(co),'model':m,'at_0':means[0],'at_0.4':means[-1],'change':means[-1]-means[0],'sd_at_0.4':sds[-1]})
  title(a,short(co));a.set_xlabel('Requested local missing rate');a.set_ylabel(lab)
 ax.flat[-1].axis('off');handles,labels=ax.flat[0].get_legend_handles_labels();ax.flat[-1].legend(handles,labels,loc='center');ax.flat[-1].text(.1,.2,'Mean +/- seed SD\n3 seeds; same saved splits',transform=ax.flat[-1].transAxes)
 b=[x for x in summary if x['model']=='baseline'];worst=max(b,key=lambda r:abs(r['change']));best=min([x for x in summary if x['modalities']==worst['modalities']],key=lambda r:r['at_0.4'] if k=='mae' else -r['at_0.4'])
 save(fig,num,f'rate_{sp}_{k}',f'{"验证集" if sp=="valid" else "测试集"}局部缺失率对{lab}的影响',f'rate组，7种模态组合×4个非零缺失率×3模型×3种子；零点取完整输入。',f'曲线从完整输入延伸至请求缺失率0.4，图中T/A/V分别表示文本/语音/视觉。误差线为跨种子标准差，保留非单调结果；没有对曲线作平滑或强制下降。基线端点变化幅度最大的组合是{worst["modalities"]}，{lab}由{worst["at_0"]:.4f}变为{worst["at_0.4"]:.4f}，差值{worst["change"]:+.4f}。该组合0.4端点表现最好的模型为{best["model"]}，值为{best["at_0.4"]:.4f}。\n\n应在同一模态组合、同一请求缺失率下比较模型，不能用不同面板的横向位置替代相同实际信息损失。全局缺失率的分母含三个模态全部有效bin，因此文本缺失30%并不等于总信息缺失30%。下表报告全部端点，帮助区分对文本损伤敏感与对音视频损伤敏感的情况。\n\n验证集用于建模选择，测试集用于描述最终泛化；本报告不据测试曲线重新挑选训练轮次或阈值。分类与回归曲线需共同判断，F1提升不自动意味着强度MAE下降。',summary)

# 27 position, 28 duration, exact same nominal rate
for num,g,levels in [(27,'position',['prefix','middle','suffix','random']),(28,'duration',['short','medium','long'])]:
 fig,ax=grid(2,3,(13,7));rr=[]
 for row,sp in enumerate(['valid','test']):
  for col,co in enumerate(MOD):
   a=ax[row,col]
   for m,c,label in zip(MODEL,MC,ML):
    v=[];sd=[]
    for lev in levels:
     one=select(C,split=sp,model=m,group=g,missing_modalities=co,**{g:lev});mu,s=ms(one,'macro_f1');v.append(mu);sd.append(s)
     rr.append({'split':sp,'modality':co,'model':m,g:lev,'macro_f1':mu,'sd':s,'mae':ms(one,'mae')[0],'effective_total_rate':ms(one,'effective_missing_rate')[0]})
    a.errorbar(range(len(levels)),v,yerr=sd,color=c,marker='o',ms=4,capsize=3,label=label)
   a.set_xticks(range(len(levels)),levels);a.set_ylabel('Macro-F1');title(a,f'{sp.title()}: {co.title()}')
 ax[0,0].legend(fontsize=8)
 spreads=[]
 for sp in ['valid','test']:
  for m in MODEL:
   for co in MOD:
    one=select(rr,split=sp,model=m,modality=co);hi=max(one,key=lambda x:x['macro_f1']);lo=min(one,key=lambda x:x['macro_f1']);spreads.append(f'{sp}/{m}/{co}：{lo[g]} {lo["macro_f1"]:.4f} 至 {hi[g]} {hi["macro_f1"]:.4f}（差{hi["macro_f1"]-lo["macro_f1"]:.4f}）')
 save(fig,num,f'missing_{g}',f'缺失{"位置" if g=="position" else "连续块时长"}对分类性能的影响','请求缺失率固定0.3；仅单模态；分别按位置或连续块时长分组；3种子。','这一组比较直接回应题目对缺失'+('位置' if g=='position' else '时长')+'的要求。横轴是离散实验设置，连线用于同一模型组内比较，不是连续函数。下表同时给出MAE和全局实际缺失率，便于检查名义上相同缺失率是否对应相近损伤。\n\n'+('前段、中段、后段与随机缺失表现的差异，只能说明当前条件下模型对时序分布敏感。未结合逐样本语义和真实词时间戳时，不能把“前段更重要”解释成所有视频的开头都含决定性情绪。' if g=='position' else 'short/medium/long表示缺失连续块的设置，不是视频秒数。总请求缺失率相同而块长度不同，体现缺失集中程度；缺乏物理时间映射时不换算为秒。时长效应可能与有效长度、离散取整和片段覆盖同时相关。')+'\n\n各组观察范围：\n\n'+'\n'.join('- '+s for s in spreads),rr)

# 29 denominators
fig,ax=grid(size=(11,4.5));rr=[]
for co,c in zip(combos,['#447BA8','#D99555','#469A91','#8875AE','#B5A15D','#707A84','#D47764']):
 x=[];y=[]
 for rate in [.1,.2,.3,.4]:
  r=select(C,split='test',model='baseline',group='rate',missing_modalities=co,requested_rate=rate);x.append(rate);y.append(ms(r,'effective_missing_rate')[0]);rr.append({'modalities':short(co),'requested':rate,'total_effective':y[-1],**{m:ms(r,'effective_missing_rate_'+m)[0] for m in MOD}})
 ax[0,0].plot(x,y,'o-',color=c,label=short(co),ms=4)
ax[0,0].plot([0,.4],[0,.4],'--',color='#999999',lw=1);title(ax[0,0],'(a) Global vs requested missing rate');ax[0,0].set_xlabel('Requested rate');ax[0,0].set_ylabel('Global effective rate');ax[0,0].legend(ncol=2)
matrix=np.array([[select(rr,modalities=short(co),requested=.3)[0][m] for m in MOD] for co in combos]);heat(ax[0,1],matrix,['Text','Audio','Vision'],[short(c) for c in combos],vmin=0,vmax=.4);title(ax[0,1],'(b) Per-modality rate at requested 0.30')
save(fig,29,'missing_rate_denominators','请求缺失率与实际缺失率的分母差异','测试集baseline，种子均值；读取effective_missing_rate及各模态对应字段。','请求缺失率针对被选中的模态有效序列；全局实际缺失率以全部模态有效bin总量作分母。只缺失一个模态时，全局比例自然小于请求值；同时缺失T+A+V时两者才较接近。离散bin数量、可用长度和取整会产生小幅偏差。\n\n热图将0.3设置分解到各模态，未被选中模态的缺失率应为零。该核对只检查保存结果的计数与实验条件是否对应，不审核实现代码。论文需要写清分母，不能把全局比例直接称为“文本缺失率”。\n\n本图有助于解释不同缺失组合之间的损伤强弱，但不能把被删除bin数量等同于语义信息量；文本、音频、视觉的一个bin并不具有相同信息价值。',rr)

# 30 ablation
ac=['','text','audio','vision','text+audio','text+vision','audio+vision','text+audio+vision'];fig,ax=grid(2,2,(12,9));rr=[]
for row,sp in enumerate(['valid','test']):
 for col,k in enumerate(['macro_f1','mae']):
  mat=[]
  for co in ac:
   mat.append([ms(select(C,split=sp,model=m,group='ablation',missing_modalities=co),k)[0] for m in MODEL])
  heat(ax[row,col],np.array(mat),['Base','No aug.','Robust'],[short(c) for c in ac],cmap='Blues' if k=='macro_f1' else 'Oranges');title(ax[row,col],f'{sp.title()}: {k}')
 for m in MODEL:
  for co in ac:
   one=select(C,split=sp,model=m,group='ablation',missing_modalities=co);rr.append({'split':sp,'model':m,'removed':short(co),**{k:ms(one,k)[0] for k in ['accuracy','macro_f1','mae','pearson']}})
save(fig,30,'whole_modality_ablation','整模态消融与极端信息丢失','ablation组，8种保留/删除组合；行标签表示被删除的模态，None表示完整输入。','整模态删除是极端压力测试，与题目主要关心的局部连续缺失不同，应作为补充消融呈现。移除T、A、V后的性能变化揭示模型对相应输入的依赖；文本删除后的下降不能独立证明文本在所有真实场景中都具有最高因果重要性。\n\n全部删除T+A+V时，即使分类准确率非零，也可能只是类先验或常数输出的表现。该条件下Pearson数值可能来自近常数浮点波动，不能解释为模型保留了实际强度预测能力。图中MAE与F1分开显示，避免不同量纲混合。\n\n本文消融使用固定已训练模型的输入删除结果，而非为每种保留模态重新训练专门模型。因此“只剩文本”的成绩应表述为该融合模型在只保留文本时的表现，不应写成重新训练的纯文本模型基线。',rr)

# 31 paired seed differences grouped by condition family
fig,ax=grid(2,2,(12,7));rr=[];gs=['baseline','rate','position','duration','ablation']
for ri,sp in enumerate(['valid','test']):
 for ci,k in enumerate(['macro_f1','mae']):
  a=ax[ri,ci]
  for pi,(left,right,label,c) in enumerate([('robust_noaug','baseline','Architecture',MC[0]),('robust','robust_noaug','Augmentation',MC[1])]):
   for gi,g in enumerate(gs):
    ds=[]
    for seed in [42,43,44]:
     aa=select(C,split=sp,seed=seed,model=left,group=g);bb={r['condition_id']:r for r in select(C,split=sp,seed=seed,model=right,group=g)}
     aa=[r for r in aa if not (g=='ablation' and not r['missing_modalities'])];ds.append(np.mean([r[k]-bb[r['condition_id']][k] for r in aa]))
    x=gi+(-.13 if pi==0 else .13);a.errorbar(x,np.mean(ds),yerr=np.std(ds,ddof=1),fmt='o',color=c,capsize=3,label=label if gi==0 else None);a.scatter(x+np.linspace(-.04,.04,3),ds,s=14,color=c,alpha=.5)
    rr.append({'split':sp,'metric':k,'comparison':left+' - '+right,'group':g,'mean_difference':np.mean(ds),'seed_sd':np.std(ds,ddof=1),'seed42':ds[0],'seed43':ds[1],'seed44':ds[2]})
  a.axhline(0,color='#777777',ls='--',lw=1);a.set_xticks(range(5),gs,rotation=20);title(a,f'{sp.title()}: delta {k}');a.legend()
save(fig,31,'paired_architecture_augmentation','架构与训练增强的配对差异','同一种子、同一条件相减后，先对条件族等权平均，再对3种子计算均值与标准差。','Architecture为No aug.减Baseline；Augmentation为Robust减No aug.。F1差值为正更好，MAE差值为负更好。图中的配对保留了种子编号和缺失条件，避免把大量相关条件误认为大量独立训练。\n\n每个条件族的权重由本次设计决定，rate组的均值并不代表真实业务缺失分布。对于ablation，去掉重复完整输入后仅汇总7个压力条件。均值接近零时应查看三个种子的符号是否一致，而不能只选择其中一次成功的结果写进论文。\n\n该设计比历史模型横向比较更能区分架构和增强作用，但模型/增强差异的解释仍应限定在保存配置、三个种子及当前条件范围内。',rr)

#32 average and worst stress; avoid treating redundant conditions independent
fig,ax=grid(size=(11,4.5));rr=[]
for sp in ['valid','test']:
 for m in MODEL:
  one=[r for r in AGG if r['split']==sp and r['model']==m and r['group']!='baseline' and not(r['group']=='ablation' and not r['missing_modalities'])]
  for scope in ['local','all_stress']:
   z=[r for r in one if scope=='all_stress' or r['group']!='ablation'];lo=min(z,key=lambda r:r['macro_f1_mean']);hi=max(z,key=lambda r:r['mae_mean']);rr.append({'split':sp,'model':m,'scope':scope,'conditions':len(z),'mean_f1':np.mean([r['macro_f1_mean'] for r in z]),'worst_f1':lo['macro_f1_mean'],'mean_mae':np.mean([r['mae_mean'] for r in z]),'worst_mae':hi['mae_mean'],'worst_f1_condition':lo['condition_id'],'worst_mae_condition':hi['condition_id']})
for j,(k,lab) in enumerate([('mean_f1','Mean local Macro-F1'),('worst_f1','Worst local Macro-F1')]):
 for si,sp in enumerate(['valid','test']):
  vs=[select(rr,split=sp,model=m,scope='local')[0][k] for m in MODEL];ax[0,j].bar(np.arange(3)+(si-.5)*.32,vs,width=.30,color=MC[si],label=sp)
 ax[0,j].set_xticks(range(3),['Base','No aug.','Robust']);ax[0,j].set_ylim(0,.8);title(ax[0,j],lab);ax[0,j].legend()
save(fig,32,'average_worst_conditions','局部缺失的平均性能与最差条件','对三种子均值再按条件等权汇总；local=49个局部缺失条件，all_stress=56个条件含整模态删除。','平均值描述实验网格的整体表现，最差值用于识别模型在何种条件下失效；两者没有相同含义。图中仅画局部缺失，避免全模态全删这种极端条件主导主要结论；下表额外列出全部压力条件供核对。\n\n最差条件是从固定测试网格中事后选出的观察值，不是有统计置信保证的性能下界。表中准确写出对应条件ID，也列出MAE最差条件，因为分类和回归最差情形未必重合。\n\n题目中的“稳定输出”可由缺失条件下仍有可用F1、MAE及跨种子波动共同支撑，不应仅凭程序产生数值便称为鲁棒。',rr)

#33 class F1 under text loss
fig,ax=grid(2,3,(12,7));rr=[]
for ri,sp in enumerate(['valid','test']):
 for ci,cl in enumerate(['negative','neutral','positive']):
  a=ax[ri,ci];k='f1_'+cl
  for m,c,label in zip(MODEL,MC,ML):
   mus=[];sds=[]
   for rate in [0,.1,.2,.3,.4]:
    z=select(C,split=sp,model=m,group='baseline') if rate==0 else select(C,split=sp,model=m,group='rate',missing_modalities='text',requested_rate=rate);mu,sd=ms(z,k);mus.append(mu);sds.append(sd);rr.append({'split':sp,'model':m,'class':cl,'rate':rate,'f1':mu,'sd':sd})
   a.errorbar([0,.1,.2,.3,.4],mus,yerr=sds,color=c,marker='o',capsize=2,ms=3,label=label)
  title(a,f'{sp.title()}: {cl.title()}');a.set_xlabel('Requested text missing rate');a.set_ylabel('Class F1');a.set_ylim(0,.9)
 ax[ri,0].legend(fontsize=8)
save(fig,33,'class_sensitivity','文本局部缺失对各情感类别的差异影响','保存的逐类F1；文本局部缺失0至0.4，两个划分、3模型、3种子。','Macro-F1的变化可能由一个类别主导，本图用相同纵轴显示三类F1。中性类既需要与正面也需要与负面区分，其完整输入表现已偏弱；文本局部丢失后，如果中性曲线先下降，整体准确率可能因为多数正面类别相对稳定而掩盖风险。\n\n各类F1结合precision与recall，不能单凭F1判定下降究竟来自漏识别还是误报。新增受控表只存逐类F1，未存各条件混淆矩阵；因此这里不伪造逐条件召回率或错误流向。\n\n观察Robust相对No aug.的作用时，应说明受益类别与可能牺牲类别，而不是只报告单个平均数。完整数值随图导出。',rr)

#34 multi-run training
fig,ax=grid(1,3,(13,4.3));rr=[]
for mi,m in enumerate(MODEL):
 a=ax[0,mi]
 for si,seed in enumerate([42,43,44]):
  h=rows(f'problem2_experiment2/{m}/seed_{seed}/training_history.csv');ep=vals(h,'epoch');a.plot(ep,vals(h,'train_macro_f1'),color=MC[si],ls='--',alpha=.65);a.plot(ep,vals(h,'valid_macro_f1'),color=MC[si],marker='o',ms=3,label=f'Seed {seed}')
  best=max(h,key=lambda r:r['valid_macro_f1']);rr.append({'model':m,'seed':seed,'recorded_epochs':len(h),'best_valid_epoch':best['epoch'],'best_valid_f1':best['valid_macro_f1'],'last_train_f1':h[-1]['train_macro_f1'],'last_valid_f1':h[-1]['valid_macro_f1'],'last_gap':h[-1]['train_macro_f1']-h[-1]['valid_macro_f1']})
 title(a,ML[mi]);a.set_xlabel('Epoch');a.set_ylabel('Macro-F1');a.set_ylim(.3,1);a.legend(fontsize=8)
save(fig,34,'controlled_training','新增九次训练的拟合与泛化过程','各模型seed_42/43/44的training_history.csv；实线验证，虚线训练。','新增训练记录与图16的30轮延长训练属于不同实验记录，不能无缝拼接成一条曲线。每个面板保留三个种子，直接观察早期提升、峰值轮次以及训练验证差距。\n\n下表按已存验证Macro-F1定位峰值，仅用于复核历史选择，不重新训练或重新选择测试模型。若训练F1持续增大而验证F1停滞，说明继续拟合训练集并未同步提升验证性能；这比只看训练loss下降更能说明模型选择的必要性。\n\n训练指标计算口径、正则化与增强可能影响训练验证数值差距，因此差距是诊断证据，不是对过拟合原因的唯一解释。',rr)

#35 all aligned a3
fig,ax=grid(size=(12,8));ar={m:sorted(select(A3,model=m,variant='aligned'),key=lambda x:x['id']) for m in MODEL};ids=[r['id'].replace('sample_','') for r in ar['baseline']]
cm=np.array([[r['predicted_class'] for r in ar[m]] for m in MODEL]).T
from matplotlib.colors import ListedColormap
heat(ax[0,0],cm,['Base','No aug.','Robust'],ids,cmap=ListedColormap(COL),vmin=0,vmax=2,dec=0);title(ax[0,0],'(a) Predicted class (0/1/2)')
for m,c,label in zip(MODEL,MC,ML):ax[0,1].plot(vals(ar[m],'regression_prediction'),range(30),'o-',color=c,ms=3,label=label)
ax[0,1].set_yticks(range(30),ids);ax[0,1].set_ylim(29.5,-.5);ax[0,1].axvline(0,color='#777777',ls='--');ax[0,1].set_xlabel('Predicted intensity');title(ax[0,1],'(b) Intensity by sample');ax[0,1].legend()
rr=[]
for i,id in enumerate(ids):
 q={'sample':id}
 for m in MODEL:q[m+'_class']=ar[m][i]['predicted_label'];q[m+'_intensity']=ar[m][i]['regression_prediction']
 rr.append(q)
save(fig,35,'attachment3_models','附件三aligned全量三模型输出','attachment3_predictions.csv，30个aligned样本×3模型；不是三种子平均。','左图0/1/2对应负面/中性/正面，右图保留每条样本的强度预测。下表列出全部30条结果，使正文图与提交文件逐条可对应。三模型分歧揭示模型选择对专项输出的影响，不能用多数票或高置信度来认定某个模型正确。\n\n新增表没有种子列，因此不能把这180行专项预测理解为三种子集成。aligned是本轮受控训练使用的特征版本，对应结果应作为主分析。unaligned在下一图作为额外输出一致性诊断；题目要求训练与专项测试维持同一特征版本。\n\n无真实标签时，类别分布、置信度和强度范围都是模型输出特征，不是性能评价。',rr)

#36 a3 distribution and same-id format
fig,ax=grid(1,3,(13,4.5));rr=[]
for mi,m in enumerate(MODEL):
 a=sorted(select(A3,model=m,variant='aligned'),key=lambda x:x['id']);b=sorted(select(A3,model=m,variant='unaligned'),key=lambda x:x['id']);assert [x['id'] for x in a]==[x['id'] for x in b]
 for j,rrr in enumerate([a,b]):
  counts=[sum(r['predicted_class']==k for r in rrr) for k in range(3)];bottom=0
  for k in range(3):ax[0,0].bar(mi*3+j,counts[k],bottom=bottom,color=COL[k],label=CL[k] if mi==0 and j==0 else None);bottom+=counts[k]
 eq=np.mean([x['predicted_class']==y['predicted_class'] for x,y in zip(a,b)]);da=vals(b,'regression_prediction')-vals(a,'regression_prediction');ax[0,1].bar(mi,eq,color=MC[mi]);ax[0,2].scatter(np.full(30,mi)+np.linspace(-.1,.1,30),da,color=MC[mi],s=18,alpha=.65)
 rr.append({'model':m,'agreement':eq,'mean_signed_intensity_difference':da.mean(),'mean_abs_intensity_difference':abs(da).mean(),'max_abs_difference':abs(da).max(),'aligned_counts':str(Counter(r['predicted_label'] for r in a)),'unaligned_counts':str(Counter(r['predicted_label'] for r in b))})
ax[0,0].set_xticks([.5,3.5,6.5],['Base A/U','No aug. A/U','Robust A/U']);ax[0,0].legend(fontsize=8,ncol=3,loc='upper center',bbox_to_anchor=(.5,1.20));title(ax[0,0],'(a) Predicted class counts')
for a in ax[0,1:]:a.set_xticks(range(3),['Base','No aug.','Robust'])
ax[0,1].set_ylim(0,1);title(ax[0,1],'(b) A/U class agreement');ax[0,2].axhline(0,color='#777777',ls='--');title(ax[0,2],'(c) Intensity: U minus A')
save(fig,36,'attachment3_formats','附件三类别分布与格式敏感性','180行专项输出；同编号aligned/unaligned描述性配对，每模型30对。','柱状分布可以识别模型是否倾向输出某一极性，一致率与差值则量化输入版本变化的敏感性。两种格式之间的一致率高并不等于准确率高，强度差小也不代表鲁棒性已得到带标签验证。\n\n本轮训练协议为aligned_50，unaligned输出不能替代同版本专项评价。图中A/U只表示输出来源，不能把60条版本记录当作60个独立真实样本。\n\n完整输入缺失结构的统计还显示附件三的有效bin比例低于1，但其中可能混有填充与局部缺失，不能将1减有效比例全部解释为人为缺失率。',rr)

# P3 labeled predictions, metrics recomputed without probabilities
P={s:rows(f'problem3_outputs/problem3_{s}_predictions.csv') for s in ['valid','test']};PM=js('problem3_outputs/problem3_metrics.json')
def metrics(z):
 y=vals(z,'true_class').astype(int);p=vals(z,'predicted_class').astype(int);cm=np.bincount(y*3+p,minlength=9).reshape(3,3);f1=2*np.diag(cm)/np.maximum(cm.sum(0)+cm.sum(1),1);err=vals(z,'regression_prediction')-vals(z,'true_regression')
 return {'accuracy':np.trace(cm)/cm.sum(),'macro_f1':f1.mean(),'mae':abs(err).mean(),'rmse':np.sqrt(np.mean(err**2)),'pearson':np.corrcoef(vals(z,'true_regression'),vals(z,'regression_prediction'))[0,1],'bias':err.mean()},cm
STATS['problem3']={}
for sp in P:
 met,cm=metrics(P[sp]);assert all(abs(met[k]-PM[sp][k])<1e-6 for k in ['accuracy','macro_f1','mae','rmse','pearson']);STATS['problem3'][sp]={'metrics':met,'confusion':cm.tolist()}

#37 main metric comparison with old ref clearly not architecture causality
fig,ax=grid(2,2,(11,7));rr=[]
for a,k in zip(ax.flat,['accuracy','macro_f1','mae','pearson']):
 for j,sp in enumerate(['valid','test']):
  vs=[old['main'][sp]['metrics'][k],STATS['problem3'][sp]['metrics'][k]];a.bar(np.arange(2)+(j-.5)*.3,vs,width=.28,color=MC[j],label=sp)
 a.set_xticks([0,1],['P2 reference','P3 explainable']);title(a,k);a.legend()
for sp in P:
 m=STATS['problem3'][sp]['metrics'];rr.append({'split':sp,**m,**{'delta_'+k:m[k]-old['main'][sp]['metrics'][k] for k in ['accuracy','macro_f1','mae','pearson']}})
save(fig,37,'p3_overall','问题三基础性能及与问题二参考结果的关系','问题三验证728、测试727条；与旧问题二固定参考的描述性比较。','问题三模型的目标包含解释输出，但可解释性能力不能代替预测准确性。本图同时报告赛题四项指标，数值由预测表重新计算并与metrics.json核对一致。\n\n问题三测试Accuracy为64.10%、Macro-F1为0.5516、MAE为0.6929、Pearson为0.6217；相较问题二固定参考，分类和回归表现均更弱。两者架构、训练与参数不同，不能将性能下降单独归因为“增加解释机制的代价”，也不能把解释输出存在视为可信性已经得到全面证明。\n\n验证和测试应分开叙述：测试准确率略高于验证，但Macro-F1更低，提示类别分布和类别间表现不均衡的影响。下面用混淆矩阵与回归诊断解释这种表面矛盾。',rr)

#38 confusion
fig,ax=grid(size=(11,4.5));rr=[]
for a,sp in zip(ax.flat,['valid','test']):
 cm=np.array(STATS['problem3'][sp]['confusion']);norm=cm/cm.sum(1,keepdims=True);heat(a,norm,CL,CL,vmin=0,vmax=1);title(a,sp.title());a.set_xlabel('Predicted class');a.set_ylabel('True class')
 for i,cl in enumerate(CL):
  pre=cm[i,i]/cm[:,i].sum();rec=norm[i,i];rr.append({'split':sp,'class':cl,'support':int(cm[i].sum()),'correct':int(cm[i,i]),'precision':pre,'recall':rec,'f1':2*pre*rec/(pre+rec),'predicted_count':int(cm[:,i].sum())})
save(fig,38,'p3_confusion','问题三的类别瓶颈与错误流向','问题三全部验证/测试预测；颜色为按真实类别归一的比例，计数见下表。','验证中性类184条只识别53条，召回28.80%；测试中性类158条只识别28条，召回17.72%。测试中性误判中64条流向负面、66条流向正面，说明问题并非单向偏好，而是中性边界整体不足。\n\n测试总误判261条，其中涉及真实或预测中性的错误为171条；直接负面与正面互换为90条。中性预测数量仅69条，明显小于真实158条。这样的混淆结构解释了Accuracy尚可而Macro-F1偏低。\n\n这是一种数值层面的错误归因，不能在没有逐样本素材复核时进一步归因为讽刺、转写错误或表情遮挡。',rr)

#39 regression
fig,ax=grid(2,2,(11,8));rr=[]
for j,sp in enumerate(['valid','test']):
 z=P[sp];y=vals(z,'true_regression');p=vals(z,'regression_prediction');slope,inter=np.polyfit(y,p,1);a=ax[0,j];a.scatter(y,p,s=12,alpha=.4,color=MC[j],edgecolors='none');a.plot([-3,3],[-3,3],'--',color='#999999');a.plot([-3,3],np.array([-3,3])*slope+inter,color=COL[0]);a.set_xlim(-3.1,3.1);a.set_ylim(-3.1,3.1);a.set_xlabel('True intensity');a.set_ylabel('Predicted intensity');title(a,f'{sp.title()}: slope {slope:.3f}')
 for lo,hi,label in [(0,0,'0'),(0,.5,'(0,.5]'),(.5,1,'(.5,1]'),(1,2,'(1,2]'),(2,3,'(2,3]')]:
  ix=(abs(y)==0) if hi==0 else ((abs(y)>lo)&(abs(y)<=hi));rr.append({'split':sp,'abs_true_band':label,'n':int(ix.sum()),'mae':np.mean(abs(p[ix]-y[ix])),'bias':np.mean(p[ix]-y[ix]),'accuracy':np.mean(vals(z,'predicted_class')[ix]==vals(z,'true_class')[ix])})
 one=select(rr,split=sp);a=ax[1,j];a.bar(range(5),[r['mae'] for r in one],color=MC[j]);a.set_xticks(range(5),[r['abs_true_band'] for r in one]);a.set_ylabel('MAE');a.set_xlabel('Absolute true intensity');title(a,sp.title()+' by intensity')
 for i,r in enumerate(one):a.text(i,r['mae']+.025,f'n={r["n"]}',ha='center',fontsize=9)
 STATS['problem3'][sp]['regression_slope']=slope
save(fig,39,'p3_regression','问题三的强度压缩与强情感误差','回归散点保留全部样本；直线为诊断最小二乘拟合，不修改预测；下排按真实绝对强度分层。','真实值越强而预测仍集中在中间区域，反映回归收缩。斜率仅用于描述预测对标签的响应幅度，不是模型训练参数。分层MAE揭示整体均值可能掩盖强情绪样本的大误差。\n\n零强度组与非零组的样本量不同；每个柱标注n，下表同时给出分类准确率，便于检查“极性正确但强度不准”的情况。强度高并不必然分类更困难，符号和幅值应分别评价。\n\n问题二也存在回归压缩，因此它是现有多模态方案共有的结果现象，但不能仅靠散点判定由损失权重、标签噪声或模型容量中的哪一项导致。',rr)

#40 confidence accuracy only, no fabricate multiclass probs
fig,ax=grid(size=(11,4.5));rr=[]
for sp,c in zip(['valid','test'],MC):
 z=P[sp];conf=vals(z,'confidence');ok=vals(z,'true_class')==vals(z,'predicted_class');ece=0;points=[]
 for low in np.arange(0,1,.1):
  ix=(conf>=low)&(conf<low+.1+1e-12);n=ix.sum()
  if n:
   mu=conf[ix].mean();acc=ok[ix].mean();ece+=n/len(z)*abs(mu-acc);points.append((mu,acc));rr.append({'split':sp,'bin_low':low,'n':int(n),'mean_confidence':mu,'accuracy':acc})
 ax[0,0].plot(*np.array(points).T,'o-',color=c,label=sp)
 thresholds=np.linspace(.34,.9,60);cov=[];risk=[]
 for t in thresholds:
  ix=conf>=t
  if ix.sum()>=15:cov.append(ix.mean());risk.append(1-ok[ix].mean())
 ax[0,1].plot(cov,risk,color=c,label=sp)
 STATS['problem3'][sp]['ece']=ece;STATS['problem3'][sp]['high_confidence_errors']=int(((conf>=.8)&~ok).sum())
ax[0,0].plot([.3,1],[.3,1],'--',color='#777777');ax[0,0].set_xlabel('Mean confidence');ax[0,0].set_ylabel('Observed accuracy');title(ax[0,0],'(a) Top-label reliability');ax[0,0].legend()
ax[0,1].set_xlabel('Coverage');ax[0,1].set_ylabel('Classification error rate');title(ax[0,1],'(b) Risk versus coverage');ax[0,1].legend()
save(fig,40,'p3_confidence','问题三概率置信度与高置信错误','预测表仅保存最大类置信度，使用10个等宽箱；风险曲线每点至少15样本。',f'验证集ECE={STATS["problem3"]["valid"]["ece"]:.4f}，测试集ECE={STATS["problem3"]["test"]["ece"]:.4f}。置信度≥0.8的错误分别为{STATS["problem3"]["valid"]["high_confidence_errors"]}与{STATS["problem3"]["test"]["high_confidence_errors"]}条。可靠性图能够说明置信水平与实际正确率是否一致，但ECE对分箱有依赖，也不能保证单例可靠。\n\n风险覆盖图描述拒绝低置信样本后保留集合的表现，覆盖率下降意味着更多样本没有得到被接受的预测。这里不基于测试结果选择部署阈值。\n\n由于问题三带标签预测表没有完整三类概率，本报告不生成该模型的逐类ROC/PR、Brier或多类NLL；仅凭最大概率无法还原另外两个类别的概率。',rr)

#41 p3 training
H=rows('problem3_outputs/training_history.csv');fig,ax=grid(1,3,(13,4.3));ep=vals(H,'epoch')
for a,k in zip(ax.flat,['loss','macro_f1','accuracy']):
 for sp,c in [('train',MC[0]),('valid',MC[1])]:a.plot(ep,vals(H,sp+'_'+k),color=c,label=sp)
 a.axvline(PM['best_epoch'],color=COL[2],ls='--',label='Selected epoch');title(a,k);a.set_xlabel('Epoch');a.legend(fontsize=8)
save(fig,41,'p3_training','问题三50轮训练与早期验证峰值','training_history.csv与problem3_metrics.json；保存选择轮为1。',f'保存模型在第{PM["best_epoch"]}轮被选中，后续训练没有得到更高的已记录验证Macro-F1。第1轮训练/验证F1为{H[0]["train_macro_f1"]:.4f}/{H[0]["valid_macro_f1"]:.4f}，最后一轮为{H[-1]["train_macro_f1"]:.4f}/{H[-1]["valid_macro_f1"]:.4f}。训练loss由{H[0]["train_loss"]:.4f}变为{H[-1]["train_loss"]:.4f}，验证loss由{H[0]["valid_loss"]:.4f}变为{H[-1]["valid_loss"]:.4f}。\n\n后期训练拟合改善而验证停滞说明继续训练并不能保证泛化。报告应使用已保存选中模型的预测，不能把第50轮训练成绩写成最终验证成绩。该问题三实验只提供seed 42，不能引用问题二的三种子波动来替代其训练稳定性证据。\n\n历史中第1轮valid_mae与最终metrics存在约十万分之几量级差异，应以最终保存的逐样本预测为本报告评价口径，不将微小浮点或评价过程差异解释为实质提升。', [{'epoch':r['epoch'],'train_loss':r['train_loss'],'valid_loss':r['valid_loss'],'train_f1':r['train_macro_f1'],'valid_f1':r['valid_macro_f1'],'valid_mae':r['valid_mae']} for r in H if r['epoch'] in [1,2,5,10,20,30,40,50]])

# Explanations
EX=rows('problem3_outputs/attachment4_explanations.csv');LC=rows('problem3_outputs/primary_modality_local_importance_samples.csv');EC=[js(str(p.relative_to(R))) for p in sorted((R/'problem3_outputs/evidence_cards').glob('*.json'))]
assert len(EX)==len(EC)==40 and len(LC)==2000
EKEY={(r['variant'],r['id']):r for r in EC}
for r in EX:
 e=EKEY[r['variant'],r['id']];assert int(r['predicted_class'])==e['predicted_class'];assert abs(r['regression_prediction']-e['regression_prediction'])<1e-7

#42 all predictions a4
fig,ax=grid(size=(12,7.5));rr=[]
for a,variant in zip(ax.flat,['aligned','unaligned']):
 z=sorted(select(EX,variant=variant),key=lambda r:r['id'])
 for i,r in enumerate(z):a.hlines(i,0,r['regression_prediction'],color=COL[int(r['predicted_class'])]);a.scatter(r['regression_prediction'],i,color=COL[int(r['predicted_class'])],s=20+50*r['confidence'])
 a.set_yticks(range(20),[r['id'] for r in z]);a.invert_yaxis();a.axvline(0,color='#777777',ls='--');a.set_xlim(-1.8,1.8);a.set_xlabel('Predicted intensity');title(a,variant.title());rr+=z
save(fig,42,'attachment4_predictions','附件四全部预测与置信度','40行=20个编号×两种版本；颜色为极性，圆大小随置信度变化。','该图逐条呈现附件四的强度和极性，不删除边界样本。表中保留所有预测及主要参考模态，满足全量展示需要；原始详细证据字符串由专项CSV提供。\n\n附件四没有真实标签，因此不可计算准确率，也不能把高置信预测称为正确解释。两种版本对应相同编号，只能作为配对版本记录，不能当作40条独立随机抽样。\n\n问题三训练使用aligned版本，因此aligned输出作为主要结果，unaligned结果作描述性补充。强度靠近零且分类概率不集中的样本应被明确标为边界预测，但本次不修改其类别或引入新的中性阈值。',rr,['variant','id','predicted_label','regression_prediction','confidence','main_modality'])

#43 per sample contributions
fig,ax=grid(size=(12,7.5));rr=[]
for a,v in zip(ax.flat,['aligned','unaligned']):
 z=sorted(select(EX,variant=v),key=lambda r:r['id']);left=np.zeros(20)
 for m,c in zip(MOD,MC):
  vv=vals(z,m+'_contribution');a.barh(range(20),vv,left=left,color=c,label=m.title());left+=vv
 a.set_yticks(range(20),[r['id'] for r in z]);a.invert_yaxis();a.set_xlim(0,1);a.set_xlabel('Normalized contribution');title(a,v.title());a.legend(ncol=3,fontsize=8,loc='upper center',bbox_to_anchor=(.5,1.13))
 for m in MOD:rr.append({'variant':v,'modality':m,'main_count':sum(r['main_modality']==m for r in z),'mean_contribution':vals(z,m+'_contribution').mean(),'median_contribution':np.median(vals(z,m+'_contribution')),'zero_count':int((vals(z,m+'_contribution')==0).sum())})
save(fig,43,'modality_contributions','附件四逐样本三模态作用程度','attachment4_explanations.csv，保存的归一化模态作用程度；每行三模态合计约1。','40份版本记录中36份以文本为主要参考模态，语音与视觉各2份。该比例说明此模型在当前样本的决策证据明显偏向文本，不能被描述为三模态贡献均衡。\n\n贡献值是归一化相对量：当其他模态遮挡变化很小时，文本占比可能接近1，但并不意味着文本单独贡献了100%的预测正确性。某模态贡献为0也不能证明它在真实世界没有情绪信息，应结合有符号遮挡变化和归一化规则分析。\n\n本图对每一条记录展示分布，避免均值抹平少数音频/视觉主导案例。主导模态的差异为选取解释卡提供依据，不应只挑选文本主导的成功例子。',rr)

#44 local heat maps
fig,ax=grid(size=(13,7));rr=[]
for a,v in zip(ax.flat,['aligned','unaligned']):
 mat=[];labels=[]
 for id in sorted({r['id'] for r in LC if r['variant']==v}):
  z=sorted(select(LC,variant=v,id=id),key=lambda r:r['aligned_bin']);assert len(z)==50;imp=vals(z,'normalized_local_importance');mat.append(imp);labels.append(id+' '+z[0]['main_modality'][0].upper());rr.append({'variant':v,'id':id,'main_modality':z[0]['main_modality'],'sum_importance':imp.sum(),'positive_bins':int((imp>0).sum()),'top_bin':int(imp.argmax()),'top5_mass':np.sort(imp)[-5:].sum(),'signed_drop_sum':vals(z,'occlusion_drop').sum()})
 im=a.pcolormesh(np.arange(51)-.5,np.arange(21)-.5,np.array(mat),cmap='Blues',vmin=0,vmax=max(.3,max(r['normalized_local_importance'] for r in LC)),rasterized=False,shading='flat');a.set_ylim(19.5,-.5);a.set_yticks(range(20),labels);a.set_xlabel('Aligned bin index (0-49)');title(a,v.title());cb=fig.colorbar(im,ax=a,shrink=.8);cb.solids.set_rasterized(False)
STATS['local_concentration']=rr
save(fig,44,'local_importance_heatmap','主要参考模态内50个局部位置的重要性分布','primary_modality_local_importance_samples.csv，40记录×50bin；T/A/V标注主导模态。','每行展示当前样本主要参考模态的完整50位置重要性；并非三个模态共150位置的全量矩阵。该CSV共2000行，足以覆盖主导模态的全局分布，不能扩称为6000条三模态全位置观测。\n\n正向遮挡下降值经归一化后绘图。深色块表示在该样本内部相对集中，不适合直接比较不同样本的绝对概率下降幅度。下表提供前5位置累计质量、正值bin数量与最高位置，便于量化集中程度。\n\n横轴严格称为bin索引，暂不标注真实秒数。零重要性既可能意味着遮挡没有降低当前预测类别概率，也可能来自负下降值被截为零；这种零值不等于该片段没有信息。',rr)

#45 aggregate distribution stratified main modality
fig,ax=grid(1,3,(13,4.4));rr=[]
for a,m,c in zip(ax.flat,MOD,MC):
 samples=[x for x in STATS['local_concentration'] if x['main_modality']==m];curves=[]
 for s in samples:curves.append(vals(sorted(select(LC,variant=s['variant'],id=s['id']),key=lambda r:r['aligned_bin']),'normalized_local_importance'))
 mat=np.array(curves);mu=mat.mean(0)
 for v in mat:a.plot(range(50),v,color=c,alpha=.15,lw=.8)
 a.plot(range(50),mu,color=c,lw=2);title(a,f'{m.title()} (n={len(samples)} records)');a.set_xlabel('Aligned bin index');a.set_ylabel('Normalized importance')
 rr.append({'modality':m,'records':len(samples),'mean_top5_mass':np.mean([s['top5_mass'] for s in samples]),'mean_positive_bins':np.mean([s['positive_bins'] for s in samples]),'peak_of_mean_bin':int(mu.argmax()),'peak_of_mean_value':mu.max()})
save(fig,45,'local_importance_aggregate','按主导模态聚合的局部重要性与样本不均衡','文本36份、音频2份、视觉2份版本记录；淡线为单记录，粗线为均值。','不同模态组的样本量极不平衡。文本曲线可以概括36份记录，但音频和视觉均只有2份，均值高度受个例影响，不能比较曲线形状后宣布某模态具有稳定的时间规律。\n\nbin索引相同并不保证对应同一语义阶段或相同绝对秒数，跨样本平均主要用于描述索引空间中的集中位置。两种版本记录也存在同编号依赖，因此这里不画把40条当独立样本得到的置信区间。\n\n下表的前5位置累计质量越大，说明单样本解释越集中；集中并不自动意味着解释更正确，需要与原始素材的真实定位和遮挡效应共同核验。',rr)

#46 signed occlusion values
fig,ax=grid(size=(11,4.5));rr=[]
for i,m in enumerate(MOD):
 z=select(LC,main_modality=m);v=vals(z,'occlusion_drop');ax[0,0].boxplot(v,positions=[i],widths=.5,showfliers=False,patch_artist=True,boxprops={'facecolor':MC[i],'alpha':.5});ax[0,0].scatter(np.full(len(v),i)+np.random.default_rng(42).uniform(-.15,.15,len(v)),v,s=4,alpha=.15,color=MC[i]);fr=[np.mean(v<0),np.mean(v==0),np.mean(v>0)];bottom=0
 for f,c in zip(fr,COL):ax[0,1].bar(i,f,bottom=bottom,color=c);bottom+=f
 rr.append({'main_modality':m,'positions':len(v),'negative_fraction':fr[0],'zero_fraction':fr[1],'positive_fraction':fr[2],'min_drop':v.min(),'median_drop':np.median(v),'max_drop':v.max()})
for a in ax.flat:a.set_xticks(range(3),[m.title() for m in MOD])
ax[0,0].axhline(0,color='#777777',ls='--');ax[0,0].set_ylabel('Predicted-class probability drop');title(ax[0,0],'(a) Signed local occlusion effect');title(ax[0,1],'(b) Negative / zero / positive shares');ax[0,1].set_ylim(0,1)
from matplotlib.patches import Patch
ax[0,1].legend(handles=[Patch(color=c,label=t) for c,t in zip(COL,['Negative','Zero','Positive'])],fontsize=8,ncol=3,loc='upper center',bbox_to_anchor=(.5,1.20))
save(fig,46,'signed_occlusion','遮挡效应的正负方向与解释强度','2000个主导模态局部位置；负值表示遮挡后当前预测类别概率反而升高。','遮挡下降并不总为正。正下降值可作为当前预测的支持证据，负值表示该局部信息对该预测存在抑制或与其他信息交互；不能先截成非负再宣称所有局部片段都支持预测。\n\n左图散点保留全部数值，箱线图不另重复画离群符号；右图给出正、零、负比例。三个模态的样本数与记录数不同，不将2000个位置当成2000个独立样本进行显著性检验。\n\n遮挡是对模型输入的局部干预，因此可检验模型响应；它并不等于自然场景中的因果实验，且全零输入可能离开正常数据分布。报告应称为局部忠实度或模型敏感性证据。',rr)

#47 attention vs contribution
fig,ax=grid(1,3,(13,4.4));rr=[]
for a,m,c in zip(ax.flat,MOD,MC):
 x=np.array([e['modality_attention'][m] for e in EC]);y=np.array([e['modality_contribution'][m] for e in EC]);rho=float(spearmanr(x,y).statistic);a.scatter(x,y,s=24,color=c,alpha=.7);a.plot([0,1],[0,1],'--',color='#999999');a.set_xlim(-.03,1.03);a.set_ylim(-.03,1.03);a.set_xlabel('Modality attention');a.set_ylabel('Reported contribution');title(a,m.title()+f': rho={rho:.2f}');rr.append({'modality':m,'n_records':40,'spearman':rho,'mean_attention':x.mean(),'mean_contribution':y.mean(),'mean_abs_difference':abs(x-y).mean()})
save(fig,47,'attention_occlusion_agreement','注意力权重与遮挡贡献是否一致','40张解释卡中的modality_attention与modality_contribution；Spearman为描述性秩相关。','两种量都呈现在[0,1]范围内，但含义不同：注意力描述内部加权，遮挡贡献描述输入扰动后的响应。散点接近对角线只能说明两种摘要接近，不代表注意力已被证明是因果解释。\n\n图中逐模态相关反映样本间排序的一致程度；主导模态组高度不均衡、同编号版本成对，因此不报告把40点视为独立观察的显著性p值。尤其对近零模态，微小数值变化可能使秩相关看起来明显，仍应结合绝对差值。\n\n保留两套量有助于识别“模型关注但遮挡影响不大”或“注意力小但干预响应明显”的样本，后续解释卡将展示具体证据。',rr)

#48-50 representative cards, one per modality from aligned if available
chosen=[]
for num,m in zip([48,49,50],MOD):
 candidates=[e for e in EC if e['main_modality']==m];e=max(candidates,key=lambda e:(e['variant']=='aligned',e['modality_contribution'][m]));chosen.append(e)
 fig,ax=grid(1,3,(13,4.8));rr=[]
 for a,mod,c in zip(ax.flat,MOD,MC):
  ev=e['evidence'][mod];labs=[]
  for r in ev:
   lab=(r.get('token','') if mod=='text' else f'bin {r["bin"]}')
   labs.append(lab);rr.append({'modality':mod,'bin':r['bin'],'token':r.get('token',''),'drop':r['occlusion_drop'],'attention':r['attention'],'reported_start_s':r.get('start_seconds',''),'reported_end_s':r.get('end_seconds',''),'reported_frame_start':r.get('frame_start',''),'reported_frame_end':r.get('frame_end','')})
  a.barh(range(len(ev)),[r['occlusion_drop'] for r in ev],color=[c if r['occlusion_drop']>=0 else '#B9BDC1' for r in ev]);a.set_yticks(range(len(ev)),labs);a.invert_yaxis();a.axvline(0,color='#777777',lw=.8);a.set_xlabel('Local probability drop');title(a,mod.title())
 summary=f'样本{e["variant"]}_{e["id"]}，预测{e["predicted_label"]}，强度{e["regression_prediction"]:.4f}，置信度{e["confidence"]:.4f}；主导模态{m}。三模态贡献分别为'+', '.join(f'{k}={e["modality_contribution"][k]:.4f}' for k in MOD)+'。'
 save(fig,num,'evidence_'+m,f'典型解释卡：{m}主导的{e["variant"]}_{e["id"]}',f'真实保存的解释卡；每模态5个候选位置，按原始输出顺序展示。',summary+'\n\n**原始转写：**'+e['raw_text']+'\n\n图中直接展示有符号遮挡下降，灰色表示负值。选择此例是为了覆盖主导模态类型，不表示它是准确预测或解释最可靠的样本。文本token只提供模型关注的词元，不能仅因词语看似带情感就断言其解释正确；[CLS]、标点或子词也可能进入候选列表。\n\n语音与视觉图使用bin标注，下表原样保留输出中的秒数与帧号以供核对。现有卡片的duration_seconds与真实素材时长需单独检查，未经核实的秒数不能作为已验证的原视频证据位置。',rr)

#51 format consistency of explanations
fig,ax=grid(1,3,(13,4.4));aa=sorted(select(EX,variant='aligned'),key=lambda r:r['id']);bb=sorted(select(EX,variant='unaligned'),key=lambda r:r['id']);rr=[]
trans=np.zeros((3,3));mt=np.zeros((3,3))
for a,b in zip(aa,bb):
 assert a['id']==b['id'];trans[int(a['predicted_class']),int(b['predicted_class'])]+=1;mt[MOD.index(a['main_modality']),MOD.index(b['main_modality'])]+=1;rr.append({'id':a['id'],'aligned_class':a['predicted_label'],'unaligned_class':b['predicted_label'],'same_class':int(a['predicted_class']==b['predicted_class']),'aligned_main':a['main_modality'],'unaligned_main':b['main_modality'],'same_main':int(a['main_modality']==b['main_modality']),'intensity_difference':b['regression_prediction']-a['regression_prediction']})
heat(ax[0,0],trans,CL,CL,dec=0);title(ax[0,0],'(a) Predicted class transitions');heat(ax[0,1],mt,[m.title() for m in MOD],[m.title() for m in MOD],dec=0);title(ax[0,1],'(b) Main modality transitions');ax[0,2].bar(range(20),[r['intensity_difference'] for r in rr],color=MC[0]);ax[0,2].set_xticks(range(0,20,2),[r['id'] for r in rr][::2]);ax[0,2].axhline(0,color='#777777');title(ax[0,2],'(c) Intensity: U minus A')
save(fig,51,'attachment4_formats','附件四预测与解释的跨版本一致性','20个编号配对；矩阵行为aligned、列为unaligned。',f'类别一致{sum(r["same_class"] for r in rr)}/20，主导模态一致{sum(r["same_main"] for r in rr)}/20，强度差绝对值均值{np.mean([abs(r["intensity_difference"]) for r in rr]):.4f}。同一个样本的预测一致不保证解释一致，反之亦然，两个层面应分别报告。\n\n对角线外的主导模态转移揭示解释对输入版本的敏感性。但两种版本不是受控的单一因素扰动，不应据此量化某个模态缺失的因果效应。\n\n题目要求保持训练与专项测试输入版本一致，因此本图用于界定额外输出的稳定性边界，不能用于在无标签专项集上选出所谓更准确版本。',rr)

#52 validate saved evidence time against read-only MP4 container duration
def mp4duration(p):
 data=p.read_bytes()
 def boxes(start,end):
  i=start
  while i+8<=end:
   sz=struct.unpack('>I',data[i:i+4])[0];typ=data[i+4:i+8];head=8
   if sz==1:sz=struct.unpack('>Q',data[i+8:i+16])[0];head=16
   if sz==0:sz=end-i
   if sz<head:break
   yield typ,i+head,i+sz;i+=sz
 for typ,a,b in boxes(0,len(data)):
  if typ==b'moov':
   for t,c,d in boxes(a,b):
    if t==b'mvhd':
     ver=data[c];off=c+(20 if ver==1 else 12);scale=struct.unpack('>I',data[off:off+4])[0];dur=struct.unpack('>Q' if ver==1 else '>I',data[off+4:off+(12 if ver==1 else 8)])[0];return dur/scale
 raise ValueError(str(p))
rr=[]
for e in EC:
 p=R/'DATA/attachment_4_explainability'/e['variant']/'videos'/f'sample_{e["id"]}.mp4'
 duration=mp4duration(reg(p)) if p.exists() else None
 rr.append({'variant':e['variant'],'id':e['id'],'reported_duration_s':e['duration_seconds'],'container_duration_s':duration,'ratio_reported_to_container':e['duration_seconds']/duration if duration else None})
tab('evidence_time_provenance',rr);fig,ax=grid(size=(11,4.5))
for v,c in zip(['aligned','unaligned'],MC):
 z=select(rr,variant=v);ax[0,0].plot(range(1,21),[r['container_duration_s'] for r in z],'o-',color=c,label=v)
ax[0,0].axhline(1,color=COL[0],ls='--',label='All cards: 1.0 s');ax[0,0].set_xlabel('Sample number');ax[0,0].set_ylabel('Duration (s)');title(ax[0,0],'(a) Video duration vs saved card');ax[0,0].legend()
x=[r['container_duration_s'] for r in rr];y=[r['reported_duration_s'] for r in rr];ax[0,1].scatter(x,y,color=MC[2],alpha=.6);ax[0,1].plot([0,max(x)],[0,max(x)],'--',color='#999999');ax[0,1].set_xlabel('MP4 container duration (s)');ax[0,1].set_ylabel('Saved card duration (s)');title(ax[0,1],'(b) Agreement check')
STATS['evidence_time']=rr
save(fig,52,'evidence_time_consistency','解释位置的物理时间可核验性','只读取原始MP4容器mvhd时长，与40张解释卡的duration_seconds比较；未重跑推理。',f'全部40张解释卡保存的duration_seconds均为1.0秒，而本地视频容器时长范围为{min(x):.3f}至{max(x):.3f}秒。当前结果的语音秒数和视觉帧号由此不能直接视为已经验证的原素材定位。这个差异是保存成果之间的实证不一致，不涉及代码审核。\n\n题目问题三要求关键证据可对应原始文本、语音时段或视觉关键帧。现有结果支持bin级局部重要性和token级候选证据，但当前秒数/帧号对应需要额外核验。仅将bin按视频总时长线性拉伸也不能证明正确，因为aligned特征位置未必是等时间采样；本报告不擅自修正。\n\n因此应把“已输出解释”和“原始素材定位已验证”区分开。图48至50用bin作主图标签，同时保留原秒数供复核，避免把有问题的时间映射再包装成确定的科研结论。',rr)

#53 additional saved problem-one validation statistics
pv=js('problem1_outputs/problem1_validation.json');pc=js('problem1_outputs/problem1_complete_analysis.json');manifest=rows('problem1_outputs/manifest.csv');p1features=rows('problem1_outputs/problem1_feature_summary.csv')
fig,ax=grid(size=(11,4.5));v=pc['validation'];counts=[v['shape_counts'][m] for m in MOD];ax[0,0].bar(np.arange(3)-.16,counts,width=.30,color=MC[0],label='Expected shape');ax[0,0].bar(np.arange(3)+.16,[v['finite_counts'][m] for m in MOD],width=.30,color=MC[1],label='Finite entries');ax[0,0].set_xticks(range(3),[m.title() for m in MOD]);ax[0,0].set_ylim(0,115);ax[0,0].set_ylabel('Samples');title(ax[0,0],'(a) Saved validation counts');ax[0,0].legend()
ax[0,1].bar(range(3),[v['valid_bin_counts'][m]/5000 for m in MOD],color=MC);ax[0,1].set_xticks(range(3),[m.title() for m in MOD]);ax[0,1].set_ylim(0,1.12);ax[0,1].set_ylabel('Valid bins / 5000');title(ax[0,1],'(b) Valid sequence positions')
for i,m in enumerate(MOD):ax[0,1].text(i,v['valid_bin_counts'][m]/5000+.02,str(v['valid_bin_counts'][m])+'/5000',ha='center')
rr=[{'modality':m,'samples_with_expected_shape':v['shape_counts'][m],'samples_with_finite_values':v['finite_counts'][m],'feature_dim':pc['dimensions'][m],'valid_bins':v['valid_bin_counts'][m],'source_intervals':v['source_interval_counts'][m]} for m in MOD]
save(fig,53,'p1_completeness','问题一新增汇总中的完整性与可追溯性','problem1_complete_analysis.json及problem1_validation.json；已有核验记录，非本次重跑提取。','三模态各100条样本具有预期形状且数值有限，逐模态汇总有300行，manifest有100行。这支持样本与特征的数量完整性；它不等于特征具备充分情感辨识能力，也不等于时序定位已经经过人工真值验证。\n\n全部模态统一50位置，文本/音频/视觉有效bin为3670/4966/5000。固定存储尺寸与有效信息长度不同，模型应通过掩码区分填充。音视频所用74/35维描述并不因维度与附件二一致，就自动拥有相同物理含义或相同特征分布。\n\n当前保存时长2.260–29.290秒与题面给出的原视频2.648–34.567秒范围不同。本文沿用已有结果的时长字段，不据此推断样本删减或擅自改数；在最终论文中应说明该字段与原视频容器时长的关系。附录提供全部100条样本的逐项统计。',rr)
tab('problem1_all_100_samples',manifest)
tab('problem1_all_300_modality_records',p1features)
tab('attachment3_all_180_predictions',A3)
tab('attachment4_all_40_explanations',EX)
PDF.close()
tab('figure_catalog',CATALOG)
STATS['sources']=SRC;STATS['figures']=CATALOG
(O/'current_statistics.json').write_text(json.dumps(STATS,ensure_ascii=False,indent=2,default=lambda v:float(v)),encoding='utf-8')
(O/'new_figure_analysis.md').write_text('\n'.join(SECTIONS),encoding='utf-8')
print('COMPLETE',len(CATALOG),'figures',len(SRC),'registered sources',flush=True)
