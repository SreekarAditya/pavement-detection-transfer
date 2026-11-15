#!/usr/bin/env python3
"""Pod session 1: relative inference speed of the setup A panel (descriptive; protocol Section 2).

Label-free. Uses a fixed sample of validation images only as inputs; no labels or outcomes are read.
Measured on the pod GPU, not on the edge device: report as relative speed on one machine.

Per model (seed 0 checkpoint of each eligible architecture, plus YOLOv10s with suppression):
  model-only  : batch 1, 640 x 640, 16-bit, fused network forward on a fixed tensor
                (50 warm-up passes, 500 timed passes, CUDA synchronized)
  pipeline    : batch 1 prediction on real images with the frozen settings
                (conf 0.001, overlap 0.7, max 300 detections; 30 warm-up, 300 timed),
                Ultralytics' own preprocess / inference / postprocess timings
Writes latency.csv and latency_env.json.
"""
import argparse, json, platform, random, statistics, time
from pathlib import Path

PANEL = [  # (row name, checkpoint stem, extra predict settings)
    ("yolov5s", "yolov5s_seed0", {}),
    ("yolov8s", "yolov8s_seed0", {}),
    ("yolov9s", "yolov9s_seed0", {}),
    ("yolov10s", "yolov10s_seed0", {}),
    ("yolov10s_nms", "yolov10s_seed0", {"end2end": False}),
    ("yolo11s", "yolo11s_seed0", {}),
    ("yolo12n", "yolo12n_seed0", {}),
    ("yolo12s", "yolo12s_seed0", {}),
    ("yolo12m", "yolo12m_seed0", {}),
]
FROZEN = dict(imgsz=640, half=True, conf=0.001, iou=0.7, max_det=300, batch=1, verbose=False)


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def pct(v, q):
    v = sorted(v)
    k = (len(v) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (k - lo)


def summarize(prefix, v):
    return {f"{prefix}_median_ms": statistics.median(v), f"{prefix}_p10_ms": pct(v, 0.10),
            f"{prefix}_p90_ms": pct(v, 0.90)}


def main(a):
    import torch
    import ultralytics
    from ultralytics import YOLO

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    imgs = [l for l in Path(a.val_list).read_text().split("\n") if l.strip()]
    rng = random.Random(2026)
    sample = sorted(rng.sample(imgs, min(a.n_images, len(imgs))))
    gpu = a.device != "cpu" and torch.cuda.is_available()
    dev = torch.device(f"cuda:{a.device}" if gpu else "cpu")
    sync = (lambda: torch.cuda.synchronize(dev)) if gpu else (lambda: None)
    env = dict(gpu=torch.cuda.get_device_name(dev) if gpu else "cpu", torch=torch.__version__, cuda=torch.version.cuda,
               cudnn=torch.backends.cudnn.version(), ultralytics=ultralytics.__version__,
               python=platform.python_version(), n_images=len(sample), image_seed=2026,
               settings=FROZEN, note="Pod GPU, PyTorch 16-bit, batch 1; not edge-device timing")
    (out / "latency_env.json").write_text(json.dumps(env, indent=2))
    (out / "latency_images.txt").write_text("\n".join(sample) + "\n")
    log(f"GPU {env['gpu']}; {len(sample)} images")

    rows = []
    for name, stem, extra in PANEL:
        w = Path(a.weights) / f"{stem}_best.pt"
        if not w.exists():
            log(f"SKIP {name}: missing {w.name}"); rows.append(dict(model=name, status="missing")); continue
        try:
            m = YOLO(str(w))
            from ultralytics.utils.torch_utils import get_flops
            n_params = int(sum(p.numel() for p in m.model.parameters()))
            gflops = float(get_flops(m.model, imgsz=640))  # Ultralytics' convention, this 4-class head
            kw = dict(FROZEN, device=a.device, **extra)
            for p in sample[:a.n_warm_pipeline]:  # warm-up, also builds the fused predictor
                m.predict(p, **kw)
            pre, inf, post = [], [], []
            for i in range(a.n_pipeline):
                s = m.predict(sample[i % len(sample)], **kw)[0].speed
                pre.append(s["preprocess"]); inf.append(s["inference"]); post.append(s["postprocess"])
            tot = [x + y + z for x, y, z in zip(pre, inf, post)]
            net = m.predictor.model  # fused AutoBackend used by predict
            x = torch.zeros(1, 3, 640, 640, device=dev)
            x = x.half() if gpu else x
            with torch.inference_mode():
                for _ in range(a.n_warm):
                    net(x)
                sync()
                fwd = []
                for _ in range(a.n_forward):
                    t0 = time.perf_counter(); net(x); sync()
                    fwd.append((time.perf_counter() - t0) * 1000)
            r = dict(model=name, checkpoint=w.name, status="ok", settings=json.dumps(extra),
                     params=n_params, gflops_640=gflops)
            r.update(summarize("forward", fwd)); r.update(summarize("pre", pre))
            r.update(summarize("infer", inf)); r.update(summarize("post", post)); r.update(summarize("pipeline", tot))
            rows.append(r)
            log(f"{name}: forward {r['forward_median_ms']:.2f} ms, pipeline {r['pipeline_median_ms']:.2f} ms")
            del m, net
            if gpu:
                torch.cuda.empty_cache()
        except Exception as e:  # one model failing must not stop the others
            log(f"FAIL {name}: {e!r}"); rows.append(dict(model=name, checkpoint=w.name, status=f"error: {e!r}"))

    import csv
    keys = sorted({k for r in rows for k in r}, key=lambda k: (k not in ("model", "checkpoint", "status"), k))
    with open(out / "latency.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=keys); wr.writeheader(); wr.writerows(rows)
    ok = sum(r.get("status") == "ok" for r in rows)
    log(f"latency done: {ok}/{len(rows)} models -> {out/'latency.csv'}")
    return 0 if ok == len(rows) else 4


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="/workspace/run/weights")
    ap.add_argument("--val_list", default="/workspace/run/outputs/session1/lists/val_all.txt")
    ap.add_argument("--out", default="/workspace/run/outputs/latency")
    ap.add_argument("--device", default="0")
    ap.add_argument("--n_images", type=int, default=200)
    ap.add_argument("--n_pipeline", type=int, default=300)
    ap.add_argument("--n_warm", type=int, default=50)
    ap.add_argument("--n_forward", type=int, default=500)
    ap.add_argument("--n_warm_pipeline", type=int, default=30)
    raise SystemExit(main(ap.parse_args()))
