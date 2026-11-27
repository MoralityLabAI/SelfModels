"""QSM-04 full replay of aggregate and episode receipts."""
import csv,json,math,sys
from pathlib import Path
run,summary,table=map(Path,sys.argv[1:])
a=json.loads(run.read_text()); b=json.loads(summary.read_text()); rows=a.pop("rows")
def check(x,y,path=""):
    if isinstance(x,dict):
        assert x.keys()==y.keys(),path
        for k in x: check(x[k],y[k],path+"."+k)
    elif isinstance(x,list):
        assert len(x)==len(y),path
        for i,(u,v) in enumerate(zip(x,y)):check(u,v,path+"."+str(i))
    elif isinstance(x,(int,float)):
        assert math.isclose(x,y,abs_tol=1e-9,rel_tol=1e-9),(path,x,y)
    else: assert x==y,(path,x,y)
check(a,b)
with table.open(newline="") as f: saved=list(csv.DictReader(f))
assert len(rows)==len(saved)
for i,(x,y) in enumerate(zip(rows,saved)):
    assert x.keys()==y.keys()
    for k,v in x.items(): check(v,float(y[k]) if isinstance(v,(int,float)) else y[k],str(i)+"."+k)
v=a["validation"]
assert v["violations"]==0 and v["maxIntervalExcess"]<1e-9 and v["maxVoiError"]<1e-9
assert v["world_episodes"]==768 and v["decisions"]==638976
assert v["nomCert"]>0 and v["robustCert"]>0 and v["probeCert"]>0
print("QSM-04 verified",len(rows),"episode-policy rows",v)
