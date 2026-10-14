"""Audit controls and summarize the matched global/local resource benchmark."""
import argparse
import csv
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('root',type=Path)
    args=parser.parse_args()
    cases=[json.loads((args.root/f'{mode}_{length}_{operation}.json').read_text())
           for operation in ('forward','generation') for length in (5000,7500)
           for mode in ('global','local')]
    controls=('gpu_uuid','checkpoint_sha256','source_hashes','frames','batch_size',
              'precision','tf32','seed','summaries','ffn_policy','ffn_chunk_size',
              'preprocessing','sampling_method','atol','rtol')
    for case in cases:
        assert case['status'] in ('success','oom'),case
        assert case['input_all_finite']
        for key in controls:
            assert case[key]==cases[0][key],(key,case[key],cases[0][key])
        if case['status']=='success':assert case['output_all_finite']
    for operation in ('forward','generation'):
        for length in (5000,7500):
            pair=[c for c in cases if c['operation']==operation and c['residues']==length]
            assert pair[0]['fixture']['input_sha256']==pair[1]['fixture']['input_sha256']
            if operation=='generation':
                for case in pair:
                    assert case['ode_saved_points']==2
                    assert case['actual_solver']==dict(method='dopri5',atol=1e-6,rtol=1e-3)
                    if case['status']=='success':
                        assert case['output_shape']==[1,250,length,14,3]
    rows=[]
    for c in cases:
        rows.append(dict(mode=c['mode'],length=c['residues'],operation=c['operation'],
            status=c['status'],seconds=c.get('median_seconds',c.get('seconds')),
            allocated_gib=c.get('peak_allocated_bytes',0)/2**30,
            reserved_gib=c.get('peak_reserved_bytes',0)/2**30,
            nfe=c.get('model_evaluations'),stage=c['stage']))
    with (args.root/'summary.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)
    audit=dict(status='passed',count=len(cases),success=sum(c['status']=='success' for c in cases),
        oom=sum(c['status']=='oom' for c in cases),controls={key:cases[0][key] for key in controls},
        engineering_fixture=True)
    (args.root/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps(audit),flush=True)


if __name__=='__main__':main()
