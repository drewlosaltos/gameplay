import json,hashlib,shutil
from pathlib import Path
root=Path('/mnt/t/output/skyball_gameplay/qwen_v8_publication_20261007_r1');out=root/'v8_val48_qwen672x384_single_stage_fp8_20261006_r3'
shutil.copy2(root/'smoke_pac_120s/parity.json',out/'evidence/runtime_parity.json')
shutil.copy2('/home/atao/vsdevel/skyball/worktrees/gameplay_v8_20261007/validation/gameplay_indoor_v8_val48_clean_v2.json',out/'evidence/val48_contract.json')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
p=out/'artifact.json';a=json.loads(p.read_text());a['artifacts']=[{'path':str(x.relative_to(out)),'sha256':sha(x),'size_bytes':x.stat().st_size} for x in sorted(out.rglob('*')) if x.is_file() and x!=p];p.write_text(json.dumps(a,indent=2,sort_keys=True)+'\n')
p=root/'export_receipt.json';d=json.loads(p.read_text());d['artifact_sha256']=sha(out/'artifact.json');p.write_text(json.dumps(d,indent=2)+'\n')
print(d)
