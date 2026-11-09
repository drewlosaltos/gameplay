import json, hashlib, shutil
from pathlib import Path
import torch
from safetensors.torch import save_file
root=Path('/mnt/t/output/skyball_gameplay/qwen_v8_publication_20261007_r1')
mid='v8_val48_qwen672x384_single_stage_fp8_20261006_r3'
out=root/mid
out.mkdir(exist_ok=False)
old=Path('/mnt/t/output/skyball_gameplay/qwen_v5_publication_20260926_r1/human_app_v5_qwen672x384_rotate5_e4_20260925_r1')
run=Path('/mnt/t/output/trn/skyball_gameplay/v8_val48_trainclean_qwen672x384_single_stage_4gpu_20261006_r3')
ck=run/'best_qwen672x384_v8_single_stage_4gpu.pth.tar'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
 return h.hexdigest()
def dump(p,d):p.write_text(json.dumps(d,indent=2,sort_keys=True)+'\n')
assert sha(ck)=='67f342a391234c85b25b6b14a1a12d3034d5acd7590677b4b89d0c9d0be0600f'
c=torch.load(ck,map_location='cpu',weights_only=False)
s={k.removeprefix('module.'):v.contiguous() for k,v in c['state_dict'].items()}
save_file(s,out/'model.safetensors')
shutil.copy2(old/'qwen_vision_config.json',out)
metadata={k:v for k,v in c.items() if k not in ('state_dict','optimizer','scheduler','scaler')}
dump(out/'checkpoint_metadata.json',metadata)
shutil.copy2(run/'config.json',out/'training_config.json')
shutil.make_archive(str(out/'training_source'),'gztar',run/'code_snapshot')
policy={'threshold':.9,'padding_sec_per_side':5.,'merge_gap_sec':0.,'sample_fps':1.,'batch_size':4}
dump(out/'gameplay_operating_points.json',{'schema_version':1,'model_id':mid,'score_semantics':'softmax probability of gameplay class index 1','classifier':{'threshold':.5,'validation_split':'clean_val48','f1':.9612237941026078},'selected_full_video':policy,'selection_caveat':'Retrospective full10 policy selection; outside-event footage is a proxy, not human non-gameplay labels.'})
evalroot=Path('/mnt/t/output/skyball_gameplay/v8_single_stage_val48_trainclean_validation_20261007_r3')
(out/'evidence').mkdir()
for name,src in {'comparison.json':evalroot/'single_stage_v8_vs_prod_report.json','event_coverage.json':evalroot/'selected_policy/coverage_candidate/metrics.json','pipeline_fp_impact.json':evalroot/'selected_policy/pipeline_fp_impact.json','validation_gate_result.json':evalroot/'selected_policy/validation_gate_result.json','searched_grid.json':evalroot/'selected_policy/searched_grid.json','val48_comparison.json':Path('/mnt/t/output/skyball_gameplay/v8_workflow_20261006/single_stage_clean_val48_vs_prod.json'),'val48_contract.json':Path('/home/atao/vsdevel/skyball/gameplay/validation/gameplay_indoor_v8_val48_clean_v2.json')}.items():shutil.copy2(src,out/'evidence'/name)
a=json.loads((old/'artifact.json').read_text())
a.update(model_id=mid,production=True,status='production',operating_points=policy)
a['source']={'checkpoint_epoch':int(c['epoch']),'checkpoint_sha256':sha(ck),'training_run':run.name,'dataset':'v8_val48_trainclean_overlay_20261006_r4','initialization':'pretrained Qwen vision; no gameplay initializer'}
a['validation']={'clean_val48_f1':.9612237941026078,'clean_val48_windows':16281,'confirmed_events':8848,'confirmed_events_excluded':0,'rallies_fully_retained':1370,'target_fp_current':114,'target_fp_rejected':94,'outside_event_footage_percent_total_match':39.67477,'full10_gate_status':'pass','label_caveat':'Model-assisted labels pending human acceptance; F1 provisional.'}
a['artifacts']=[{'path':str(p.relative_to(out)),'sha256':sha(p),'size_bytes':p.stat().st_size} for p in sorted(out.rglob('*')) if p.is_file()]
dump(out/'artifact.json',a)
dump(root/'export_receipt.json',{'bundle':str(out),'checkpoint_sha256':sha(ck),'artifact_sha256':sha(out/'artifact.json'),'weights_sha256':sha(out/'model.safetensors'),'checkpoint_epoch':c['epoch']})
print((root/'export_receipt.json').read_text())
