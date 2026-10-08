from pathlib import Path
import csv,html
R=Path(__file__).resolve().parents[3]; out=R/'assets/figures/readme'
INK='#253744'; MUTED='#617582'; GRID='#E3E8EB'; CLEAN='#A4AFB5'; BASE='#7997AB'; OUR='#E45B59'
def text(x,y,s,size=14,color=INK,bold=False):
 return f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" font-weight="{700 if bold else 400}">{html.escape(str(s))}</text>'
def begin(w,h,title,sub):
 return [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}"><title>{html.escape(title)}</title><rect width="{w}" height="{h}" fill="#FAFBFC"/><style>text{{font-family:Arial,sans-serif}}</style>',text(36,47,title,25,bold=True),text(36,77,sub,13,MUTED)]
def panel(s,x,y,w,title,rows,scale=100,labelw=170,gap=30,unit='Policy success rate (%)'):
 h=100+gap*len(rows)
 s.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="14" fill="white" stroke="{GRID}"/>')
 s.append(text(x+18,y+30,title,18,bold=True)); start=x+labelw; bw=w-labelw-60
 for tick in [0,scale/2,scale]:
  xx=start+bw*tick/scale
  s.append(f'<line x1="{xx}" y1="{y+61}" x2="{xx}" y2="{y+66+len(rows)*gap}" stroke="{GRID}"/>')
  s.append(text(xx-5,y+54,f'{tick:g}',10,MUTED))
 for i,(name,v,label,c) in enumerate(rows):
  yy=y+81+i*gap
  s.append(text(x+16,yy+4,name,12,bold=c==OUR))
  end=start+bw*v/scale
  s.append(f'<rect x="{start}" y="{yy-8}" width="{max(1,end-start)}" height="16" rx="3" fill="{c}"/>')
  s.append(text(end+6,yy+4,label,11,bold=c==OUR))
 s.append(text(start,y+h-10,unit,11,MUTED))
 return h
def save(name,s): (out/name).write_text('\n'.join(s+['</svg>']))
with (R/'assets/figures/paper-main-results/table1_results.csv').open() as f: data=list(csv.DictReader(f))
s=begin(1500,575,'Main results: architecture-dependent attack effectiveness','Paper Table 1 · LIBERO-10 Eval500 · 500 episodes per condition · lower SR means a stronger attack')
for x,model,col in [(25,'FastWAM','fastwam_policy_sr'),(520,'LingBot-VA','lingbot_policy_sr'),(1015,'Cosmos Policy','cosmos_policy_sr')]:
 rows=[]
 for r in data:
  name=r['method'].replace('Shared clean reference','Clean reference'); v=float(r[col]); c=OUR if name.startswith('PASA-KV') else CLEAN if name.startswith('Clean') else BASE
  rows.append((name,v,f'{v:.1f}',c))
 panel(s,x,108,460,model,rows)
s.append(text(36,545,'PASA-KV is highlighted in coral. Cosmos Top-5 ties DRIFT, UPA-RFAS, and UADA at 89.2%; results cover one checkpoint per system.',12,MUTED));save('main_success.svg',s)
controls=[('Probe-K/V',95,47.5),('Evenly spaced',37,18.5),('Random-1',173,86.5),('Random-2',127,63.5),('Random-3',50,25.0),('Tail-AV',3,1.5)]
with (out/'table2_selection.csv').open('w',newline='') as f:
 wr=csv.writer(f,lineterminator="\n");wr.writerow(['selection','successes','trials','policy_sr']);wr.writerows((a,b,200,c) for a,b,c in controls)
s=begin(1050,480,'Why select action-facing routes?','Paper Table 2 · same victim, joint K/V loss, three targets, patch geometry, and 10k exposures')
panel(s,28,105,994,'Matched FastWAM selection controls',[(a,v,f'{v:.1f}% ({k}/200)',OUR if a=='Tail-AV' else BASE) for a,k,v in controls],labelw=210,gap=34)
s.append(text(36,456,'Tail-AV: 1.5% SR versus 47.5% for probe K/V displacement. Eval200 is separate from the Table 1 Eval500 comparison.',12,MUTED));save('selection_controls.svg',s)
with (R/'assets/figures/piper-real-robot/table3_task_results.csv').open() as f: data=list(csv.DictReader(f))
s=begin(1400,335,'Piper real-robot task outcomes','Paper Table 3 · 10 trials per condition · Top-3 and Top-5 patches are separately trained')
for x,task in [(20,'Stack three bowls'),(485,'Place carrot in bowl'),(950,'Place pen in cup')]:
 rows=[]
 for r in data:
  if r['task']!=task:continue
  k,n=int(r['successes']),int(r['trials']);name=r['method'].replace('Clean FastWAM','Clean')
  rows.append((name,100*k/n,f'{k}/{n}',CLEAN if name=='Clean' else OUR))
 panel(s,x,104,430,task,rows,labelw=155,gap=31)
s.append(text(36,315,'Bars show observed task success, not attack success. These small-sample results are reported per task; uncertainty details are linked below.',12,MUTED));save('robot_success.svg',s)
