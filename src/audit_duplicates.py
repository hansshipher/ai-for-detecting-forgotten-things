#!/usr/bin/env python3
"""Аудит датасета: near-duplicate кадры и пересечения между наборами.

Зачем это нужно:

  1. В CCTV-датасетах соседние кадры почти одинаковы. Если они попадут
     в train и test, метрика начинает мерить запоминание, а не детекцию.
     Поэтому считаем dHash (64-бит перцептивный хеш) и склеиваем кадры
     с Hamming-расстоянием <= --max-dist в один кластер = одну сцену.
  2. Если картинки из нового датасета пересекаются с videos/ (наши демо-
     ролики) или с dataset/all, их нельзя пускать в обучение: оценка на
     демо перестанет быть честной.

Запуск (из папки src/):
    python audit_duplicates.py --root ../dataset/12k-dataset
    python audit_duplicates.py --root ../dataset/12k-dataset --videos ../videos
    python audit_duplicates.py --root ../dataset/12k-dataset --against ../dataset/all/images

Выход: сводка в консоль + JSON/CSV в --out (по умолчанию runs/12k_audit).
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

import config as cfg

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def parse_args():
    p = argparse.ArgumentParser(
        description="dHash-аудит: кластеры near-duplicate кадров и пересечения наборов.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--root", required=True,
                   help="датасет для аудита (ищет картинки рекурсивно)")
    p.add_argument("--videos", nargs="*", default=None,
                   help="видеофайлы или папка с видео (кадры сравниваются с root)")
    p.add_argument("--video-step", type=int, default=30,
                   help="брать каждый N-й кадр видео (по умолчанию 30)")
    p.add_argument("--against", nargs="*", default=None,
                   help="дополнительные папки с картинками для проверки пересечений")
    p.add_argument("--max-dist", type=int, default=5,
                   help="порог Hamming для near-duplicate (по умолчанию 5)")
    p.add_argument("--out", default=str(cfg.BASE_DIR / "runs" / "12k_audit"),
                   help="куда сохранить отчёт (по умолчанию runs/12k_audit)")
    return p.parse_args()


def list_images(root):
    root = Path(root)
    if root.is_file():
        return [root]
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXT)


def dhash_file(path):
    """64-битный difference hash. None, если файл не читается."""
    img = cv2.imread(str(path), cv2.IMREAD_REDUCED_GRAYSCALE_8)
    if img is None:
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None or img.size == 0:
        return None
    img = cv2.resize(img, (9, 8), interpolation=cv2.INTER_AREA)
    bits = np.packbits((img[:, 1:] > img[:, :-1]).flatten())
    return int.from_bytes(bits.tobytes(), "big")


def dhash_frames(video_path, step):
    """Хеши кадров видео: [(frame_idx, hash), ...]."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[WARN] cannot open video: {video_path}")
        return []
    out, idx = [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            small = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
            bits = np.packbits((small[:, 1:] > small[:, :-1]).flatten())
            out.append((idx, int.from_bytes(bits.tobytes(), "big")))
        idx += 1
    cap.release()
    return out


class UnionFind:
    def __init__(self, n):
        self.p = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.p[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


def match_hashes(a, b, max_dist, chunk=512):
    """Все пары (i, j) с Hamming(a[i], b[j]) <= max_dist."""
    a = np.asarray(a, dtype=np.uint64)
    b = np.asarray(b, dtype=np.uint64)
    pairs = []
    for start in range(0, len(a), chunk):
        block = a[start:start + chunk][:, None] ^ b[None, :]
        dist = np.bitwise_count(block)
        ii, jj = np.nonzero(dist <= max_dist)
        pairs.extend((start + int(i), int(j)) for i, j in zip(ii, jj))
    return pairs


def split_of(path, root):
    """Имя сплита (train/valid/test) по пути относительно root."""
    try:
        rel = Path(path).relative_to(root)
    except ValueError:
        return ""
    return rel.parts[0] if len(rel.parts) > 1 else ""


def main():
    args = parse_args()
    t0 = time.perf_counter()

    root = Path(args.root).resolve()
    images = list_images(root)
    if not images:
        print(f"[ERROR] no images found in {root}")
        return 1

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] root: {root}")
    print(f"[INFO] images: {len(images)} | max Hamming: {args.max_dist}")

    hashes, bad = [], []
    for i, p in enumerate(images, 1):
        h = dhash_file(p)
        if h is None:
            bad.append(str(p))
        hashes.append(h)
        if i % 2000 == 0:
            print(f"  hashed {i}/{len(images)} ({time.perf_counter()-t0:.0f}s)")

    valid = [(p, h) for p, h in zip(images, hashes) if h is not None]
    print(f"[INFO] hashed: {len(valid)} | unreadable: {len(bad)}")

    # --- кластеры внутри root ---
    hs = [h for _, h in valid]
    uf = UnionFind(len(valid))
    pairs = match_hashes(hs, hs, args.max_dist)
    n_pairs = 0
    for i, j in pairs:
        if i < j:
            uf.union(i, j)
            n_pairs += 1

    groups = {}
    for i in range(len(valid)):
        groups.setdefault(uf.find(i), []).append(i)

    sizes = sorted((len(v) for v in groups.values()), reverse=True)
    n_clusters = len(groups)
    print(f"\n[RESULT] near-duplicate clusters: {n_clusters} из {len(valid)} картинок")
    print(f"  largest: {sizes[:10]}")
    print(f"  singleton (уникальных сцен): {sum(1 for s in sizes if s == 1)}")
    print(f"  в кластерах >1: {sum(s for s in sizes if s > 1)} картинок "
          f"в {sum(1 for s in sizes if s > 1)} кластерах")

    # утечка между сплитами
    cross_split = []
    for idxs in groups.values():
        if len(idxs) < 2:
            continue
        sp = {split_of(valid[i][0], root) for i in idxs} - {""}
        if len(sp) > 1:
            cross_split.append(idxs)
    n_cross_imgs = sum(len(x) for x in cross_split)
    print(f"  кластеров, пересекающих сплиты: {len(cross_split)} "
          f"({n_cross_imgs} картинок)")

    # --- пересечения с видео ---
    video_hits = []
    if args.videos:
        vpaths = []
        for v in args.videos:
            pv = Path(v)
            if pv.is_dir():
                vpaths.extend(sorted(x for x in pv.iterdir()
                                     if x.suffix.lower() in {".avi", ".mp4", ".mov", ".mkv"}))
            elif pv.exists():
                vpaths.append(pv)
        for vp in vpaths:
            frames = dhash_frames(vp, args.video_step)
            if not frames:
                continue
            fh = [h for _, h in frames]
            hit = match_hashes(fh, hs, args.max_dist)
            uniq_imgs = {valid[j][0] for _, j in hit}
            uniq_frames = {frames[i][0] for i, _ in hit}
            print(f"\n[VIDEO] {vp.name}: frames sampled={len(frames)} | "
                  f"matched frames={len(uniq_frames)} | matched dataset images={len(uniq_imgs)}")
            for fi, j in hit[:5]:
                print(f"    frame {frames[fi][0]} ~ {valid[j][0].relative_to(root)}")
            video_hits.append({
                "video": str(vp),
                "frames_sampled": len(frames),
                "matched_frames": sorted(uniq_frames),
                "matched_images": sorted(str(valid[j][0].relative_to(root)) for _, j in hit),
            })

    # --- пересечения с другими наборами ---
    against_hits = []
    for extra in (args.against or []):
        ep = Path(extra)
        eimgs = list_images(ep)
        if not eimgs:
            print(f"[WARN] no images in {ep}")
            continue
        eh = [(p, dhash_file(p)) for p in eimgs]
        eh = [(p, h) for p, h in eh if h is not None]
        hit = match_hashes([h for _, h in eh], hs, args.max_dist)
        uniq_ours = {valid[j][0] for _, j in hit}
        uniq_theirs = {eh[i][0] for i, _ in hit}
        print(f"\n[AGAINST] {ep}: images={len(eh)} | matched ours={len(uniq_ours)} "
              f"| matched theirs={len(uniq_theirs)}")
        against_hits.append({
            "against": str(ep),
            "matched_ours": sorted(str(valid[j][0].relative_to(root)) for _, j in hit),
            "matched_theirs": sorted(str(eh[i][0]) for i, _ in hit),
        })

    # --- сохранение ---
    report = {
        "root": str(root),
        "images": len(images),
        "unreadable": bad,
        "max_dist": args.max_dist,
        "clusters": n_clusters,
        "cluster_sizes_top": sizes[:50],
        "singletons": sum(1 for s in sizes if s == 1),
        "cross_split_clusters": [[str(valid[i][0].relative_to(root)) for i in x] for x in cross_split],
        "video_hits": video_hits,
        "against_hits": against_hits,
    }
    (out_dir / "audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                        encoding="utf-8")

    with open(out_dir / "clusters.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["cluster_id", "size", "split", "path"])
        for cid, (_, idxs) in enumerate(sorted(groups.items(), key=lambda kv: -len(kv[1]))):
            for i in idxs:
                w.writerow([cid, len(idxs), split_of(valid[i][0], root),
                            str(valid[i][0].relative_to(root))])

    print(f"\n[INFO] report: {out_dir / 'audit.json'}")
    print(f"[INFO] clusters: {out_dir / 'clusters.csv'}")
    print(f"[INFO] elapsed: {time.perf_counter()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
