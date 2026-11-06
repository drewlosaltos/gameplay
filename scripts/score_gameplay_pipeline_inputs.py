"""Score the exact production inputs using its RGB preprocessing, in FP32."""
import argparse, hashlib, json, sys, time
from pathlib import Path
import numpy as np
import torch

REPO=Path(__file__).resolve().parents[1]
PIPE=Path('/home/atao/vsdevel/skyball/pipeline')
sys.path[:0]=[str(REPO/'scripts'),str(REPO),str(PIPE)]
from skyball_gameplay.full10 import validate_full10_inputs
from evaluate_gameplay_gate import load_model, score_embeddings, atomic_write_scores
from skyball_pipeline.gameplay_runtime import iter_sampled_rgb_frames, preprocess_rgb_frame, probe_video
from skyball_gameplay.preprocessing import LetterboxResize
from gameplay_wide_video import iter_wide_rgb_frames
from PIL import Image

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--shard',type=int,default=0)
p.add_argument('--num-shards',type=int,default=1)
p.add_argument('--checkpoint',type=Path,required=True)
p.add_argument('--pipeline-root',type=Path,required=True)
p.add_argument('--output-root',type=Path,required=True)
args=p.parse_args()
if not 0 <= args.shard < args.num_shards: p.error('invalid shard')
ROOT=args.output_root
SOURCE=args.pipeline_root
CHECKPOINT=args.checkpoint
paths=validate_full10_inputs(SOURCE)
torch.set_num_threads(4)
device=torch.device('cuda:0');torch.cuda.set_device(device)
model,metadata=load_model(CHECKPOINT,device)
digest=hashlib.sha256()
with CHECKPOINT.open('rb') as checkpoint_file:
    for block in iter(lambda: checkpoint_file.read(8 * 1024 * 1024), b''):
        digest.update(block)
sha=digest.hexdigest()
assert metadata['input_frames']==4
letterbox = metadata.get('resize_mode', 'stretch') == 'letterbox'
if not letterbox:
    assert metadata['image_size']==[512,512]
for path in paths[args.shard::args.num_shards]:
    start=time.monotonic();manifest=json.loads(path.read_text())
    timing=json.loads((path.parent/'gameplay_timing.json').read_text())
    video=Path(manifest['video']);gt=Path(manifest['gt_sync_csv'])
    assert hashlib.sha256(gt.read_bytes()).hexdigest()==manifest['inputs']['gt_sync_csv']['sha256']
    assert video.stat().st_size==manifest['inputs']['video']['size_bytes']
    info=probe_video(video)
    assert info['frame_count']==timing['source_frames']
    assert abs(info['fps']-manifest['fps'])<1e-8
    artifact=json.loads(Path(manifest['gameplay_artifact_json']).read_text())
    contract=artifact['runtime']['input']
    assert contract['normalization_mean']==list(metadata['normalization_mean'])
    assert contract['normalization_std']==list(metadata['normalization_std'])
    out=ROOT/path.parent.name;out.mkdir(parents=True,exist_ok=False)
    print('SCORING',path.parent.name,flush=True)
    buffered=[];embeddings=[];parity=None
    def flush():
        global parity
        if not buffered:return
        batch=torch.from_numpy(np.stack(buffered)).to(device)
        with torch.inference_mode():
            encoded=torch.flatten(model.encoder(batch),1)
            if parity is None and len(batch)>=4:
                direct=torch.softmax(model(batch[:4].unsqueeze(0)),dim=1)[0,1]
                cached=torch.softmax(model.classifier(model.dropout(encoded[:4].mean(0,keepdim=True))),dim=1)[0,1]
                parity=float(abs(direct-cached));assert parity<1e-5,parity
        embeddings.append(encoded.cpu());buffered.clear()
    iterator = (iter_wide_rgb_frames(video, sample_fps=1., duration_limit_sec=timing['source_frames']/info['fps'],
                                    long_edge=max(metadata['image_size'])) if letterbox else
                iter_sampled_rgb_frames(video,sample_fps=1.,duration_limit_sec=timing['source_frames']/info['fps']))
    geometry = LetterboxResize(metadata['image_size']) if letterbox else None
    for frame in iterator:
        if letterbox:
            normalized = np.asarray(geometry(Image.fromarray(frame)), dtype=np.float32).transpose(2,0,1)/255.
            mean = np.asarray(metadata['normalization_mean'], dtype=np.float32)[:,None,None]
            std = np.asarray(metadata['normalization_std'], dtype=np.float32)[:,None,None]
            buffered.append((normalized-mean)/std)
        else:
            buffered.append(preprocess_rgb_frame(frame,contract))
        if len(buffered)==32:flush()
    flush()
    encoded=torch.cat(embeddings)
    assert len(encoded)==timing['sampled_frames'],(len(encoded),timing['sampled_frames'])
    scores=score_embeddings(encoded,model,device,4,1,2048)
    assert len(scores)==timing['windows']
    atomic_write_scores(out/'scores.csv',list(range(len(scores))),scores,4.)
    result={'match':path.parent.name,'checkpoint':str(CHECKPOINT),'checkpoint_sha256':sha,
      'video':str(video),'video_resolved':str(video.resolve()),'video_info':info,'ground_truth':str(gt),
      'gt_sha256':manifest['inputs']['gt_sync_csv']['sha256'],'source_manifest':str(path),
      'preprocessing':'Exact pipeline iter_sampled_rgb_frames and preprocess_rgb_frame; ffmpeg 1Hz/512-long-edge bilinear, OpenCV INTER_LINEAR 512x512, artifact normalization',
      'runtime':'PyTorch FP32; each frame encoded once, four-frame embeddings averaged',
      'model':metadata,'score_count':len(scores),'sampled_frames':len(encoded),
      'first_window_cached_vs_direct_max_score_difference':parity,'elapsed_sec':time.monotonic()-start}
    if letterbox:
        result['preprocessing'] = ('Pipeline-equivalent 1Hz timing, ffmpeg bilinear decode at candidate long edge; '
                                   'PIL bilinear aspect-preserving fit with RGB128 centered padding, checkpoint normalization')
    (out/'metadata.json').write_text(json.dumps(result,indent=2)+'\n')
    print('COMPLETE',path.parent.name,len(scores),round(result['elapsed_sec'],1),flush=True)
