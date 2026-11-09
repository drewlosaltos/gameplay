import csv,json
from pathlib import Path
import numpy as np
from skyball_pipeline.gameplay_trt import validate_gameplay_bundle
from skyball_pipeline.gameplay_qwen import score_qwen_gameplay_video
root=Path('/mnt/t/output/skyball_gameplay/qwen_v8_publication_20261007_r1')
bundle=root/'v8_val48_qwen672x384_single_stage_fp8_20261006_r3'
print(validate_gameplay_bundle(bundle),flush=True)
out=root/'smoke_pac_120s';out.mkdir(exist_ok=False)
src=Path('/mnt/t/output/skyball_gameplay/v8_single_stage_val48_trainclean_validation_20261007_r3/full10_scores/eval_club_005_pac_balboa')
m=json.loads((src/'metadata.json').read_text())
score_qwen_gameplay_video(video=Path(m['video']),engine_path=bundle/'model.safetensors',artifact_json=bundle/'artifact.json',gameplay_csv=out/'gameplay.csv',scores_csv=out/'scores.csv',timing_json=out/'timing.json',max_frames=round(m['video_info']['fps']*120),threshold=.9,merge_gap_sec=0,padding_sec=5,sample_fps=1,batch_size=4,min_start_frame=0)
a=list(csv.DictReader((out/'scores.csv').open()));b=list(csv.DictReader((src/'scores.csv').open()))[:len(a)]
print('columns',a[0].keys(),b[0].keys(),flush=True)
def scores(rows):return np.array([float(r.get('score_gameplay',r.get('gameplay_score',r.get('score')))) for r in rows])
x,y=scores(a),scores(b)
r={'windows':len(a),'max_abs_score_difference':float(abs(x-y).max()),'threshold_decision_differences':int(((x>=.9)!=(y>=.9)).sum())}
(out/'parity.json').write_text(json.dumps(r,indent=2)+'\n');print(r,flush=True)
assert r['max_abs_score_difference']<.005 and r['threshold_decision_differences']==0
