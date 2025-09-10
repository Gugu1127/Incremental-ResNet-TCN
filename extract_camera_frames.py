#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import cv2
import sys
import shutil

def iter_camera_videos(root: Path):
    """
    在 root 下尋找所有「<學號>/Camera_video/*.mp4」且檔名符合 *_camera.mp4。
    回傳每個影片的 (video_path, student_id)。
    """
    if not root.is_dir():
        raise ValueError(f"根目錄不存在或非資料夾：{root}")

    # 假設 root 之下第一層全是 <學號> 目錄（如 11227218）
    for student_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        cam_dir = student_dir / "Camera_video"
        if not cam_dir.is_dir():
            continue
        # 僅抓 *_camera.mp4
        for mp4 in sorted(cam_dir.glob("*_camera.mp4")):
            yield mp4, student_dir.name

def ensure_dir(path: Path, overwrite: bool = False):
    if path.exists():
        if overwrite:
            shutil.rmtree(path)
            path.mkdir(parents=True, exist_ok=True)
        else:
            path.mkdir(parents=True, exist_ok=True)
    else:
        path.mkdir(parents=True, exist_ok=True)

def extract_frames(
    video_path: Path,
    out_dir: Path,
    frame_step: int = 1,
    start: int = 0,
    max_frames: int = 0,
    jpg_quality: int = 95,
    overwrite: bool = False,
) -> tuple[int, int]:
    """
    從 video_path 逐格讀取並輸出到 out_dir。
    回傳 (實際輸出張數 saved, 影片總格數 total 或 -1 表示未知)。
    """
    # 若已存在影格且不覆寫，直接略過
    if out_dir.exists() and not overwrite:
        existing = list(out_dir.glob("img_*.jpg"))
        if existing:
            return (len(existing), -1)

    # 準備輸出資料夾
    ensure_dir(out_dir, overwrite=overwrite)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"無法開啟影片：{video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        total = -1  # 有些編碼會取不到

    idx = 0
    saved = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx >= start and (idx - start) % frame_step == 0:
            out_file = out_dir / f"img_{saved + 1:06d}.jpg"
            ok = cv2.imwrite(str(out_file), frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpg_quality)])
            if not ok:
                cap.release()
                raise RuntimeError(f"寫入影格失敗：{out_file}")
            saved += 1
            if max_frames > 0 and saved >= max_frames:
                break
        idx += 1

    cap.release()
    return (saved, total)

def main():
    parser = argparse.ArgumentParser(
        description="抽取 <學號>/Camera_video/*_camera.mp4 之影格"
    )
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="輸入根目錄，例如：/mnt/e/ResNet_video/online_exam1"
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        required=True,
        help="輸出影格根目錄，例如：/mnt/e/ResNet_video/online_exam1Frames"
    )
    parser.add_argument(
        "--frame-step",
        type=int,
        default=1,
        help="每隔多少格擷取一張（預設 1：每格都擷取）"
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="從第幾格開始擷取（預設 0）"
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="最多擷取多少張（0 表示不限制）"
    )
    parser.add_argument(
        "--jpg-quality",
        type=int,
        default=95,
        help="JPG 品質（1-100，預設 95）"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="若輸出資料夾已存在且有影格，是否強制覆寫重建"
    )
    args = parser.parse_args()

    videos = list(iter_camera_videos(args.root))
    if not videos:
        print("未找到任何 *_camera.mp4。請確認根目錄與結構。", file=sys.stderr)
        sys.exit(1)

    print(f"共偵測到 {len(videos)} 支 camera 影片，開始處理…\n")

    ok = skip = fail = 0
    for video_path, student_id in videos:
        # 產生對應的輸出路徑：<out-root>/<學號>/Camera_video/
        out_dir = args.out_root / student_id / "Camera_video"

        # 若已存在且未要求覆寫，視為跳過
        if out_dir.exists() and not args.overwrite and list(out_dir.glob("img_*.jpg")):
            print(f"[SKIP] 已有影格：{out_dir}")
            skip += 1
            continue

        try:
            saved, total = extract_frames(
                video_path=video_path,
                out_dir=out_dir,
                frame_step=args.frame_step,
                start=args.start,
                max_frames=args.max_frames,
                jpg_quality=args.jpg_quality,
                overwrite=args.overwrite,
            )
            t = "未知" if total < 0 else total
            print(f"[OK] {video_path} -> {out_dir} | 輸出 {saved} 張（總格數 {t}）")
            ok += 1
        except Exception as e:
            print(f"[FAIL] {video_path} -> {out_dir} | {e}", file=sys.stderr)
            fail += 1

    print(f"\n完成統計：OK={ok}, SKIP={skip}, FAIL={fail}")

if __name__ == "__main__":
    main()
