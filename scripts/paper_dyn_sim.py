"""Simulate one fault line at two clearing times straddling its CCT (stable/unstable pair).
Same engine and scenario as e4_sim/scripts/cct_bracket.py; writes npz traces for paper_figures."""
import os, sys, json
from dataclasses import replace
from pathlib import Path
import numpy as np, pandas as pd
for k in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"): os.environ[k]="1"
HOME=Path.home(); case=sys.argv[1]; out=Path(sys.argv[2]); out.mkdir(parents=True,exist_ok=True)
probe = len(sys.argv) > 3 and sys.argv[3] == "probe"
ev2=str(HOME/"data/volume/inputs/v41dyn/work/events_v2"); sys.path.insert(0,ev2)
from v2lib import Engine
from tranpy.simulation.engine import FaultScenario
xlsx=HOME/f"data/volume/outputs/v41dyn/738439695169033/{case}/prep/{case}.xlsx"
c=pd.read_parquet(HOME/f"data/exp/e4_sim/cct/cct_{case}.parquet")
b=c[(c.status=="bracketed")&(c.vn_kv==380)&c.cct_lo.between(0.18,0.30)].sort_values("cct_lo")
row=b.iloc[len(b)//2]
print("line",int(row.fault_line),"pos",row.fault_position,"cct",row.cct_lo,row.cct_hi,row.bus1,row.bus2)
E=Engine(system_name="germany",case=xlsx,simulation_time_s=1.0+1.0+5.0,step_s=0.01,record_buses="all",name_width=4,n_jobs=1)
base=FaultScenario(scenario_id=0,fault_line=int(row.fault_line),fault_start_s=1.0,clearing_duration_s=0.1,fault_resistance_ohm=0.0,
    fault_reactance_ohm=0.5,load_index=0,load_change_pct=0.0,trip_line=None,operating_point=1.0,load_factors=None,fault_position=float(row.fault_position))
meta=dict(case=case,fault_line=int(row.fault_line),fault_position=float(row.fault_position),cct_lo=float(row.cct_lo),cct_hi=float(row.cct_hi),
          bus1=int(row.bus1),bus2=int(row.bus2),line_idx=str(row.line_idx),source=f"e4_sim/cct/cct_{case}.parquet")
for tag,clr in (("stable",round(row.cct_lo-0.02,3)),("unstable",round(row.cct_hi+0.02,3))):
    r=E._simulate_one(replace(base,clearing_duration_s=clr))
    names=list(r.channel_names)
    if probe:
        print(len(names)); import collections; print(collections.Counter(n.split(":")[-1] for n in names)); print(names[:12]); break
    np.savez_compressed(out/f"dyn_{tag}.npz",time=np.asarray(r.time),values=np.asarray(r.values,dtype=np.float32),names=np.array(names),completed=r.completed)
    oos=[j for j,n in enumerate(names) if n.endswith("s:outofstep")]
    meta[tag]=dict(clearing_s=clr,label=int((np.asarray(r.values)[:,oos]>0.5).any()),completed=bool(r.completed))
    print(tag,meta[tag])
if not probe: (out/"dyn_meta.json").write_text(json.dumps(meta,indent=1))
