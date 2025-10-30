import argparse, random, re, shutil
from pathlib import Path
from collections import defaultdict

NAME_RE = re.compile(r'^(-?\d+)_c(\d)s\d+_\d+_\d+\.jpg$', re.IGNORECASE)

def parse_name(p: Path):
    m = NAME_RE.match(p.name)
    if not m: return None
    return int(m.group(1)), int(m.group(2))  # pid, cam

def index_pid_cam(folder: Path):
    """Return {pid: {cam: [Path, ...], ...}, ...}"""
    folder = Path(folder)
    by_pid_cam = defaultdict(lambda: defaultdict(list))
    for img in folder.glob('*.jpg'):
        parsed = parse_name(img)
        if not parsed: continue
        pid, cam = parsed
        by_pid_cam[pid][cam].append(img)
    return by_pid_cam

def choose_cams_topcount(cam2paths: dict[int, list[Path]], k: int, tie_seed: int = 0):
    """Pick up to k cameras with most images; tie-broken deterministically."""
    rng = random.Random(tie_seed)
    items = [(cam, len(paths)) for cam, paths in cam2paths.items()]
    # stable sort by (-count, cam); randomize only exact ties in count
    # group by count
    counts = {}
    for cam, cnt in items:
        counts.setdefault(cnt, []).append(cam)
    ordered = []
    for cnt in sorted(counts.keys(), reverse=True):
        cams = counts[cnt]
        if len(cams) > 1:
            rng.shuffle(cams)  # break ties reproducibly
        ordered.extend((cam, cnt) for cam in cams)
    return [cam for cam, _ in ordered[:k]]

def ensure_query_gallery_balance(k, train_cam2paths, query_cam2paths, gallery_cam2paths, tie_seed=0):
    """For test IDs: try to ensure at least one query cam and one gallery cam are kept (when available)."""
    rng = random.Random(tie_seed)
    all_cam2paths = defaultdict(list)
    for d in (train_cam2paths, query_cam2paths, gallery_cam2paths):
        for cam, lst in (d or {}).items():
            all_cam2paths[cam].extend(lst)
    if not all_cam2paths:
        return []  # shouldn't happen

    q_cams = set(query_cam2paths.keys()) if query_cam2paths else set()
    g_cams = set(gallery_cam2paths.keys()) if gallery_cam2paths else set()

    chosen = []
    # 1) prefer one query cam if possible
    if q_cams:
        q_counts = [(cam, len(query_cam2paths[cam])) for cam in q_cams]
        q_counts.sort(key=lambda x: (-x[1], x[0]))
        chosen.append(q_counts[0][0])
    # 2) prefer one gallery cam (distinct) if possible
    if g_cams:
        g_counts = [(cam, len(gallery_cam2paths[cam])) for cam in g_cams if cam not in set(chosen)]
        if g_counts:
            g_counts.sort(key=lambda x: (-x[1], x[0]))
            chosen.append(g_counts[0][0])

    # 3) fill remaining slots by overall top counts
    if len(chosen) < k:
        top = choose_cams_topcount(all_cam2paths, k, tie_seed)
        for cam in top:
            if cam not in chosen:
                chosen.append(cam)
            if len(chosen) >= k:
                break

    # 4) if there are fewer than k available cameras, keep all
    return chosen[:k]

def copy_filtered(folder_src: Path, folder_dst: Path, cams_to_keep_for_pid: dict[int, set[int]],
                  preserve_junk: bool, sort_output: bool):
    folder_dst.mkdir(parents=True, exist_ok=True)
    kept = []
    for img in folder_src.glob('*.jpg'):
        parsed = parse_name(img)
        if not parsed: continue
        pid, cam = parsed
        if pid <= 0 and preserve_junk:
            keep = True
        else:
            cams = cams_to_keep_for_pid.get(pid)
            keep = (cams is None) or (cam in cams)  # None means pid not in map → keep all (shouldn't happen)
        if keep:
            kept.append(img)
    if sort_output:
        kept.sort(key=lambda p: (parse_name(p)[0], parse_name(p)[1], p.name))
    for p in kept:
        shutil.copy2(p, folder_dst / p.name)
    return kept

def main():
    ap = argparse.ArgumentParser(description="Limit cameras per ID (keep all identities; keep all images from selected cameras).")
    ap.add_argument("--src", required=True, help="Path to Market-1501 root")
    ap.add_argument("--dst", required=True, help="Output root")
    ap.add_argument("--cams_per_id", type=int, default=2, help="Max distinct cameras to keep per identity")
    ap.add_argument("--selection", choices=["topcount","random","first"], default="topcount",
                    help="How to choose cameras when >K exist")
    ap.add_argument("--preserve_junk", action="store_true", help="Keep all gallery images with pid<=0 (distractors)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sort_output", action="store_true", help="Sort output by PID→cam→name")
    args = ap.parse_args()

    src = Path(args.src); dst = Path(args.dst)
    bb_train_src = src / "bounding_box_train"
    bb_test_src  = src / "bounding_box_test"
    query_src    = src / "query"

    bb_train_dst = dst / "bounding_box_train"
    bb_test_dst  = dst / "bounding_box_test"
    query_dst    = dst / "query"

    # Index folders
    train_idx  = index_pid_cam(bb_train_src)
    gallery_idx= index_pid_cam(bb_test_src)
    query_idx  = index_pid_cam(query_src)

    rng = random.Random(args.seed)

    cams_per_id_train = {}
    for pid, cam2paths in train_idx.items():
        cams = list(cam2paths.keys())
        if len(cams) <= args.cams_per_id:
            cams_per_id_train[pid] = set(cams)
            continue
        if args.selection == "topcount":
            chosen = choose_cams_topcount(cam2paths, args.cams_per_id, tie_seed=args.seed+pid)
        elif args.selection == "random":
            chosen = rng.sample(cams, args.cams_per_id)
        else:  # first
            chosen = sorted(cams)[:args.cams_per_id]
        cams_per_id_train[pid] = set(chosen)

    cams_per_id_test = {}
    test_pids = set(query_idx.keys()) | set(gallery_idx.keys())
    for pid in test_pids:
        q_cam2 = query_idx.get(pid, {})
        g_cam2 = gallery_idx.get(pid, {})
        all_cam2 = defaultdict(list)
        for d in (q_cam2, g_cam2):
            for cam, lst in d.items():
                all_cam2[cam].extend(lst)
        cams = list(all_cam2.keys())
        if not cams:
            continue
        if len(cams) <= args.cams_per_id:
            chosen = cams
        else:
            if args.selection == "topcount":
                chosen = ensure_query_gallery_balance(args.cams_per_id, {}, q_cam2, g_cam2, tie_seed=args.seed+pid)
            elif args.selection == "random":
                chosen = rng.sample(cams, args.cams_per_id)
            else:  # first
                chosen = sorted(cams)[:args.cams_per_id]
        cams_per_id_test[pid] = set(chosen)

    # Copy
    kept_train   = copy_filtered(bb_train_src, bb_train_dst, cams_per_id_train, preserve_junk=False,              sort_output=args.sort_output)
    kept_query   = copy_filtered(query_src,    query_dst,    cams_per_id_test,  preserve_junk=False,              sort_output=args.sort_output)
    kept_gallery = copy_filtered(bb_test_src,  bb_test_dst,  cams_per_id_test,  preserve_junk=args.preserve_junk, sort_output=args.sort_output)

    # Summary
    def count_ids(paths): return len({parse_name(p)[0] for p in paths if parse_name(p)})
    readme = dst / "README_LIMIT_CAMERAS.txt"
    readme.write_text(
        "Market-1501: limited cameras per identity (IDs preserved)\n"
        f"cams_per_id={args.cams_per_id}, selection={args.selection}, preserve_junk={args.preserve_junk}\n"
        f"Train:  images={len(kept_train)}  ids={count_ids(kept_train)}\n"
        f"Query:  images={len(kept_query)}  ids={count_ids(kept_query)}\n"
        f"Gallery:images={len(kept_gallery)} ids={count_ids(kept_gallery)}\n"
        f"Seed:   {args.seed}\n"
    )
    print("Done.")
    print(readme.read_text())

if __name__ == "__main__":
    main()
