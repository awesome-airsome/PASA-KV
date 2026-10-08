"""Generate an SVG bar chart for the manuscript's primary comparison Table 1."""
from pathlib import Path
import csv
import html

ROOT=Path(__file__).resolve().parent
INK='#263746'; MUTED='#657782'; GRID='#E0E6E9'
BLUE='#3979B8'; CLEAN='#B9C1C5'; TOP3='#EFA01B'; TOP4='#F4C45C'; TOP5='#E45B59'
COLORS={'Shared clean reference':CLEAN,'PASA-KV Top-3':TOP3,'PASA-KV Top-4':TOP4,'PASA-KV Top-5':TOP5}
MODELS=[('FastWAM','fastwam_policy_sr','fastwam_gpu_h',12),('LingBot-VA','lingbot_policy_sr','lingbot_gpu_h',10),('Cosmos Policy','cosmos_policy_sr','cosmos_gpu_h',25)]

def label(x,y,value,size=13,color=INK,weight=400,anchor='start'):
    return f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" font-weight="{weight}" text-anchor="{anchor}">{html.escape(str(value))}</text>'

with (ROOT/'table1_results.csv').open(newline='',encoding='utf-8') as f:
    data=list(csv.DictReader(f))
methods=[r['method'] for r in data]
def color(method): return COLORS.get(method,BLUE)

s=['''<svg xmlns="http://www.w3.org/2000/svg" width="1500" height="1200" viewBox="0 0 1500 1200" role="img" aria-labelledby="title desc">
<title id="title">Primary comparison results from manuscript Table 1</title><desc id="desc">Horizontal bar charts compare policy success rates and patch construction GPU hours for FastWAM, LingBot-VA, and Cosmos Policy. Lower success rate means a stronger attack.</desc><rect width="1500" height="1200" fill="#F7F9FA"/><style>text{font-family:Inter,-apple-system,BlinkMacSystemFont,"Segoe UI",Arial,sans-serif}</style>''',
label(56,62,'Primary results across world action models',30,INK,700),
label(58,97,'Paper Table 1  ·  LIBERO-10 Eval500 (10 tasks × 50 trials)  ·  lower policy SR indicates a stronger attack',14,MUTED)]

xs=[50,540,1030]; pw=460; rows_top=221; gap=29; methods_y=range(len(methods))
# Legend
legend=[('WAM-adapted baselines',BLUE),('Shared clean reference',CLEAN),('PASA-KV Top-3',TOP3),('PASA-KV Top-4',TOP4),('PASA-KV Top-5',TOP5)]
pos=56
for name,c in legend:
    s += [f'<rect x="{pos}" y="119" width="14" height="14" rx="3" fill="{c}"/>',label(pos+21,131,name,12,MUTED,500)]
    pos += {'WAM-adapted baselines':205,'Shared clean reference':195,'PASA-KV Top-3':145,'PASA-KV Top-4':145,'PASA-KV Top-5':145}[name]

# Policy SR panels.
for x,(model,srcol,costcol,costmax) in zip(xs,MODELS):
    s.append(f'<rect x="{x}" y="158" width="{pw}" height="398" rx="16" fill="white" stroke="{GRID}"/>')
    s += [label(x+20,189,model,18,INK,700),label(x+20,210,'Policy success rate (%) · lower is stronger',11,MUTED)]
    left=x+166; width=220; top=246
    for tick in (0,25,50,75,100):
        xx=left+width*tick/100
        s += [f'<line x1="{xx}" y1="{top-12}" x2="{xx}" y2="{top+gap*9+12}" stroke="{GRID}"/>',label(xx,top-21,tick,10,MUTED,400,'middle')]
    for i,r in enumerate(data):
        y=top+i*gap+5; val=float(r[srcol]); xx=left+width*val/100
        s.append(label(x+15,y+4,r['method'].replace('Shared clean reference','Clean reference'),11,INK,600 if r['method'].startswith('PASA-KV') else 400))
        s.append(f'<rect x="{left}" y="{y-8}" width="{max(1,xx-left)}" height="15" rx="3" fill="{color(r["method"])}"/>')
        s.append(label(min(xx+6,x+pw-28),y+4,f'{val:.1f}',10,INK,700 if r['method'].startswith('PASA-KV Top-5') else 500))

# Construction-cost panels.
s += [label(56,606,'Recorded patch-construction cost',23,INK,700),label(58,630,'GPU-hours · lower is faster; each model has its own horizontal scale',13,MUTED)]
for x,(model,srcol,costcol,costmax) in zip(xs,MODELS):
    s.append(f'<rect x="{x}" y="650" width="{pw}" height="420" rx="16" fill="white" stroke="{GRID}"/>')
    s += [label(x+20,682,model,18,INK,700),label(x+20,703,f'GPU-hours · scale 0–{costmax:g}',11,MUTED)]
    left=x+166; width=220; top=738
    for tick in (0,costmax/2,costmax):
        xx=left+width*tick/costmax
        s += [f'<line x1="{xx}" y1="{top-12}" x2="{xx}" y2="{top+gap*9+12}" stroke="{GRID}"/>',label(xx,top-21,f'{tick:g}',10,MUTED,400,'middle')]
    for i,r in enumerate(data):
        y=top+i*gap+5; raw=r[costcol]
        s.append(label(x+15,y+4,r['method'].replace('Shared clean reference','Clean reference'),11,INK,600 if r['method'].startswith('PASA-KV') else 400))
        if not raw:
            s.append(label(left+8,y+4,'—',11,MUTED,500))
            continue
        val=float(raw); xx=left+width*val/costmax
        s.append(f'<rect x="{left}" y="{y-8}" width="{max(1,xx-left)}" height="15" rx="3" fill="{color(r["method"])}"/>')
        s.append(label(min(xx+6,x+pw-34),y+4,f'{val:.2f}',10,INK,500))

s += [label(57,1114,'FastWAM PASA-KV Top-k cost includes Probe100. Top-k patches are separately trained; Cosmos Policy ties the lowest observed SR at 89.2%.',12,MUTED),
       label(57,1140,'Construction costs follow the paper’s recorded protocols. See the manuscript table for method-specific settings and scope.',12,MUTED),'</svg>']
(ROOT/'table1_results.svg').write_text('\n'.join(s),encoding='utf-8')
print(ROOT/'table1_results.svg')
