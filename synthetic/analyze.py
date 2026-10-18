#!/usr/bin/env python3
"""Aggregate independent pools, preserve paired contrasts, and draw pilot plots."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np

METRICS=('harm','harm_any','utility','fallback','lower_coverage','audit_fraction','gold_labels_mean','variance_bad_exact','min_audit_probability')

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--results',default='pilot')
    parser.add_argument('--no-plot',action='store_true')
    args=parser.parse_args()
    folder=Path(args.results)
    rows=list(csv.DictReader((folder/'seed_metrics.csv').open()))
    groups={}
    for row in rows:
        key=(row['defender'],row['condition'],row['gate'])
        groups.setdefault(key,[]).append(row)
    rng=np.random.default_rng(7781)
    summary=[]
    for key, records in sorted(groups.items()):
        records=sorted(records,key=lambda x:int(x['seed']))
        out=dict(zip(('defender','condition','gate'),key))
        for metric in METRICS:
            values=np.array([float(r[metric]) for r in records if r.get(metric)])
            if not len(values) or not np.all(np.isfinite(values)): continue
            boot=values[rng.integers(0,len(values),(4000,len(values)))].mean(axis=1)
            out[metric]=float(values.mean())
            out[metric+'_pool_bootstrap95']=np.quantile(boot,[.025,.975]).tolist()
        summary.append(out)
    lookup={(r['defender'],r['condition'],r['gate']):r for r in summary}
    paired=[]
    specs=[('active_joint_minus_scores','active','joint_harm','active','scores_only'),
           ('mix50_minus_active_own_attacks','mix50','joint_harm','active','joint_harm'),
           ('mix50_minus_uniform_own_attacks','mix50','joint_harm','uniform','joint_harm')]
    for label,ma,ca,mb,cb in specs:
        a=sorted(groups[ma,ca,'normal'],key=lambda r:int(r['seed']))
        b=sorted(groups[mb,cb,'normal'],key=lambda r:int(r['seed']))
        assert [r['seed'] for r in a]==[r['seed'] for r in b]
        for metric in ('harm','utility','lower_coverage'):
            values=np.array([float(x[metric])-float(y[metric]) for x,y in zip(a,b)])
            boots=values[rng.integers(0,len(values),(10000,len(values)))].mean(axis=1)
            paired.append({'contrast':label,'metric':metric,'mean':float(values.mean()),'pool_bootstrap95':np.quantile(boots,[.025,.975]).tolist()})
    result={'summary':summary,'paired_contrasts':paired,
            'interval_note':'Percentile bootstrap over independent pools, conditional on this DGP and candidate attack family. All-zero intervals are not evidence of zero population risk. Replays within a pool are not independent datasets.',
            'certification_note':'Normal approximation is not certified; nominal simultaneous coverage is 95%. Bernstein is the separate valid conservative reference.'}
    (folder/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    for m in ('active','mix25','mix50','mix75','uniform'):
        for c in ('clean','scores_only','joint_harm'):
            r=lookup[m,c,'normal']
            print(f"{m:8} {c:12} harm={100*r['harm']:.3f}% improvement={100*r['utility']:+.3f}pp coverage={100*r['lower_coverage']:.2f}% fallback={100*r['fallback']:.2f}%")
    print('PAIRED',json.dumps(paired,indent=2))
    if args.no_plot: return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    names=['active','mix25','mix50','mix75','uniform']
    labels=['Active','25% uniform','50% uniform','75% uniform','Uniform']
    colors=['#bf493e','#cf9d3a','#128780','#4679a5','#192f44']
    fig,axes=plt.subplots(1,3,figsize=(13.5,4.3),layout='constrained')
    data=[lookup[m,'joint_harm','normal'] for m in names]
    for ax,metric,title in zip(axes, ['harm','utility','lower_coverage'],['Harmful selection (%) ↓','True improvement (percentage points) ↑','Simultaneous lower-bound coverage (%) ↑']):
        vals=np.array([100*r[metric] for r in data])
        errors=np.array([[100*(r[metric]-r[metric+'_pool_bootstrap95'][0]) for r in data],[100*(r[metric+'_pool_bootstrap95'][1]-r[metric]) for r in data]])
        ax.bar(np.arange(5),vals,color=colors,yerr=errors,capsize=3)
        ax.set_xticks(np.arange(5),labels,rotation=35,ha='right')
        ax.set_title(title,fontsize=11)
        ax.axhline(0,color='#999999',lw=.7)
        for x,y in enumerate(vals): ax.annotate(f'{y:.2f}',(x,y),xytext=(0,(-15 if metric=='lower_coverage' and y>20 else 6) if y>=0 else -15),textcoords='offset points',ha='center',fontsize=9,color=('white' if metric=='lower_coverage' and y>20 else '#111111'))
    axes[2].axhline(95,color='#b63030',linestyle='--',lw=1,label='Nominal 95%')
    axes[2].set_title('Lower-bound coverage (%, nominal 95%) ↑',fontsize=11)
    axes[0].set_ylim(0,102);axes[1].set_ylim(-1.5,8);axes[2].set_ylim(0,108)
    fig.suptitle('Synthetic pilot: attacks optimized against each allocator\nNormal-approximation gate is NOT certified; 40 pools × 512 audit replays',fontsize=13)
    fig.savefig(folder/'pilot_figure.png',dpi=180)
    plt.close(fig)

if __name__=='__main__': main()
