# SkyBall Large Detector TensorRT Experiment

This directory contains a local experiment for benchmarking the SkyBall
`large_dense_v4_addcats_sparse_v3_2g` detector at `1024x1024` with dense
6-frame inputs.

The export wrapper preserves the normal training-time inference path: input is
`[B, 6, 1024, 1024]`, temporal aggregation runs over the two adjacent 3-frame
clips, and `--output-mode all-frames` exports the tensors needed by tracking
visualization/evaluation, including `pred_boxes_all_frames` and
`pred_logits_all_frames`.

Artifacts default to:

```text
/mnt/t/output/rf-detr-mf/benchmarks/large_dense_v4_addcats_sparse_v3_2g_trt
```

Typical commands:

```bash
cd /home/atao/vsdevel/skyball/gameplay

CUDA_VISIBLE_DEVICES=2 conda run --no-capture-output -n skyball-prod-infer \
  python uber/export_onnx.py \
    --resolution 1024 \
    --num-frames 6 \
    --output-mode all-frames

CUDA_VISIBLE_DEVICES=2 conda run --no-capture-output -n skyball-prod-infer \
  python uber/convert_onnx_precision.py \
    --onnx /mnt/t/output/rf-detr-mf/benchmarks/large_dense_v4_addcats_sparse_v3_2g_trt/large_addcats_b1_t6_1024_allframes.onnx \
    --precision fp16

CUDA_VISIBLE_DEVICES=2 conda run --no-capture-output -n skyball-prod-infer \
  python uber/build_trt_engine.py \
    --onnx /mnt/t/output/rf-detr-mf/benchmarks/large_dense_v4_addcats_sparse_v3_2g_trt/large_addcats_b1_t6_1024_allframes_fp16.onnx \
    --engine /mnt/t/output/rf-detr-mf/benchmarks/large_dense_v4_addcats_sparse_v3_2g_trt/large_addcats_b1_t6_1024_allframes_trt11_fp16.engine \
    --precision fp32

CUDA_VISIBLE_DEVICES=2 conda run --no-capture-output -n skyball-prod-infer \
  python uber/benchmark_detector.py \
    --engine /mnt/t/output/rf-detr-mf/benchmarks/large_dense_v4_addcats_sparse_v3_2g_trt/large_addcats_b1_t6_1024_allframes_trt11_fp16.engine \
    --precision fp16
```

Notes:

- `trtexec` is not required; this uses TensorRT's Python API.
- `torch_tensorrt` is not used because it currently fails to import in
  `pt5090new`.
- Supported TensorRT builder precisions are `fp32`, `fp16`, and `bf16` when the
  installed TensorRT exposes the corresponding builder flag.
- ONNX export uses fixed shapes and export-only compatibility patches for two
  exporter limitations: DINOv2 antialiased bicubic interpolation and the
  projector's channel-first `LayerNorm` shape expression.
- TensorRT `10.7.0.post1` cannot build engines for the RTX 5090 because it
  reports `Target GPU SM 120 is not supported by this TensorRT release.` The
  `skyball-prod-infer` env uses TensorRT `11.1.0.106`, which can build and run
  these engines.
- TensorRT 11 is strongly typed here, so FP16 engines are built by first
  converting the ONNX graph to mixed FP16 with ModelOpt while preserving FP32
  I/O, then building with `--precision fp32`.
- Benchmark `fps` means dense 6-frame windows per second. With non-overlapping
  6-frame windows, frame-equivalent detector throughput is `fps * 6`, but that
  is model-only throughput and excludes video decode, preprocessing, stitching,
  drawing, and encoding. `batch_fps` in the JSON payloads means batches per
  second.

Current full-output all-frame FP16 benchmark on GPU 2, 6 frames, 1024x1024, 20
warmup iterations and 100 timed iterations:

| Batch | PyTorch ms/batch | PyTorch windows/s | TRT ms/batch | TRT windows/s | TRT frame-equiv/s | TRT speedup |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 15.672 | 63.81 | 5.095 | 196.27 | 1177.62 | 3.08x |
| 4 | 29.346 | 136.30 | 15.846 | 252.44 | 1514.62 | 1.85x |

Result JSONs:

```text
/mnt/t/output/rf-detr-mf/benchmarks/large_dense_v4_addcats_sparse_v3_2g_trt/benchmark_trt11_fp16_allframes_b1_t6_1024.json
/mnt/t/output/rf-detr-mf/benchmarks/large_dense_v4_addcats_sparse_v3_2g_trt/benchmark_trt11_fp16_allframes_b4_t6_1024.json
```

TRT visualization and true video-path timing, using the bs=1 all-frame engine
on GPU 3:

```text
/mnt/t/output/rf-detr-mf/visualizations/large_addcats_trt_allframes_20260618/videos
```

The rendered videos include detection overlays, the compact track table, and
the event timeline at 1920x1320. The rendered speed includes decode,
preprocessing, TRT inference, stitching, overlay drawing, and MP4 encoding. The
tracking-only speed omits drawing/encoding but still reads the video, preprocesses
windows, runs TRT, and stitches tracks.

| Clip | Processed frames | Windows | Rendered video fps | Tracking-only frames/s | Tracking-only windows/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| `000004_intrepid_iguana_play1` | 360 | 60 | 60.41 | 174.36 | 29.06 |
| `000004_intrepid_iguana_play2` | 324 | 54 | 54.35 | 173.43 | 28.91 |
| `000004_intrepid_iguana_play3` | 534 | 89 | 59.58 | 185.98 | 31.00 |
| Total/weighted | 1218 | 203 | 58.33 | 178.98 | 29.83 |
