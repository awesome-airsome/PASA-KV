"""Generate the Piper SVG chart from transcribed manuscript Tables 3 and 36."""
from pathlib import Path
import csv
import html

ROOT = Path(__file__).resolve().parent
INK, MUTED, GRID = '#253744', '#617582', '#E3E8EB'
CLEAN, BASE, OUR = '#A4AFB5', '#7997AB', '#E45B59'
COLORS = {'Clean FastWAM':CLEAN,'PASA-KV Top-3':OUR,'PASA-KV Top-5':OUR}

def rows(name):
    with (ROOT/name).open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))

def ci(k,n):
    z=1.959963984540054; p=k/n; den=1+z*z/n
    center=(p+z*z/(2*n))/den
    half=z*((p*(1-p)/n+z*z/(4*n*n))**.5)/den
    return 100*(center-half),100*(center+half)

def text(x,y,s,size=14,color=INK,weight=400,anchor='start'):
    return f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" font-weight="{weight}" text-anchor="{anchor}">{html.escape(str(s))}</text>'

s=['''<svg xmlns="http://www.w3.org/2000/svg" width="1440" height="850" viewBox="0 0 1440 850" role="img" aria-labelledby="title desc">
<title id="title">Piper real-robot closed-loop results</title><desc id="desc">PASA-KV Top-3 and Top-5 success rates across three AgileX Piper tasks and baseline success on bowl stacking. Ten trials per condition with 95 percent Wilson intervals.</desc>
<rect width="1440" height="850" fill="#FAFBFC"/><style>text{font-family:Inter,-apple-system,BlinkMacSystemFont,"Segoe UI",Arial,sans-serif}</style>''',
text(58,65,'Piper real-robot closed-loop results',30,INK,700),
text(60,101,'AgileX Piper  ·  10 in-distribution trials per condition  ·  95% Wilson intervals',14,MUTED)]
for x,w in [(50,650),(715,675)]:
    s.append(f'<rect x="{x}" y="140" width="{w}" height="620" rx="18" fill="white" stroke="{GRID}"/>')
s += [text(78,185,'Per-task attack effect',20,INK,700),text(78,210,'Table 3  ·  separate Top-k patches',12,MUTED),
      text(744,185,'Bowl-stacking baselines',20,INK,700),text(744,210,'Table 36  ·  shared task, method-specific geometries',12,MUTED)]

# Table 3: grouped bars for three tasks and three separately trained conditions.
x0,w=255,340
for tick in (0,25,50,75,100):
    x=x0+w*tick/100
    s += [f'<line x1="{x}" y1="245" x2="{x}" y2="684" stroke="{GRID}"/>',text(x,231,tick,11,MUTED,400,'middle')]
s.append(text(x0+w/2,719,'Policy success rate (%)',12,MUTED,500,'middle'))
tasks=['Stack three bowls','Place carrot in bowl','Place pen in cup']
methods=['Clean FastWAM','PASA-KV Top-3','PASA-KV Top-5']
data=rows('table3_task_results.csv')
for i,task in enumerate(tasks):
    mid=290+i*145
    s.append(text(76,mid+5,task,14,INK,600))
    for j,method in enumerate(methods):
        r=next(r for r in data if r['task']==task and r['method']==method)
        k,n=int(r['successes']),int(r['trials']); value=100*k/n; low,high=ci(k,n); y=mid+(j-1)*30
        xx=x0+w*value/100; xl=x0+w*low/100; xh=x0+w*high/100
        s += [f'<rect x="{x0}" y="{y-8}" width="{max(1,xx-x0)}" height="16" rx="4" fill="{COLORS[method]}"/>',
              f'<line x1="{xl}" y1="{y}" x2="{xh}" y2="{y}" stroke="{INK}" stroke-width="1.5"/><line x1="{xl}" y1="{y-5}" x2="{xl}" y2="{y+5}" stroke="{INK}"/><line x1="{xh}" y1="{y-5}" x2="{xh}" y2="{y+5}" stroke="{INK}"/>',
              text(min(xh+8,663),y+5,f'{k}/{n}',12,INK,700)]
for x,method,name in [(94,methods[0],'Clean'),(260,methods[1],'Top-3'),(448,methods[2],'Top-5')]:
    s += [f'<rect x="{x}" y="727" width="13" height="13" rx="3" fill="{COLORS[method]}"/>',text(x+20,739,name,12,MUTED,500)]

# Table 36: shared bowl-stacking task across methods.
x0,w=1000,290
for tick in (0,25,50,75,100):
    x=x0+w*tick/100
    s += [f'<line x1="{x}" y1="245" x2="{x}" y2="684" stroke="{GRID}"/>',text(x,231,tick,11,MUTED,400,'middle')]
s.append(text(x0+w/2,719,'Policy success rate (%)',12,MUTED,500,'middle'))
for i,r in enumerate(rows('table36_bowl_baselines.csv')):
    k,n=int(r['successes']),int(r['trials']); value=100*k/n; low,high=ci(k,n); y=267+i*48
    name=r['method']; color=COLORS.get(name,BASE)
    s.append(text(744,y+5,name,13,INK,700 if name.startswith('PASA-KV') else 500))
    xx=x0+w*value/100; xl=x0+w*low/100; xh=x0+w*high/100
    s += [f'<rect x="{x0}" y="{y-8}" width="{max(1,xx-x0)}" height="16" rx="4" fill="{color}"/>',
          f'<line x1="{xl}" y1="{y}" x2="{xh}" y2="{y}" stroke="{INK}" stroke-width="1.5"/><line x1="{xl}" y1="{y-5}" x2="{xl}" y2="{y+5}" stroke="{INK}"/><line x1="{xh}" y1="{y-5}" x2="{xh}" y2="{y+5}" stroke="{INK}"/>',
          text(min(xh+7,1360),y+5,f'{k}/{n}',11,INK,700 if name.startswith('PASA-KV') else 500)]
s += [text(62,802,'Bars show observed success; labels give successes / trials. Table 36 methods used method-specific patch geometries,',12,MUTED),
      text(62,824,'so its bowl-stacking results are a shared-task comparison rather than a matched-geometry ranking.',12,MUTED),'</svg>']
(ROOT/'piper_results.svg').write_text('\n'.join(s),encoding='utf-8')
print(ROOT/'piper_results.svg')
