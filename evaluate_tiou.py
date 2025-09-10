#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import csv
import glob
from typing import List, Dict, Any, Tuple

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager
font_prop = font_manager.FontProperties(fname="font.ttc")

# ============== 可調參數區 ==============
PRED_DIR           = "predictions"      # 推理 CSV 來源目錄：*_pred.csv
ANNOTATION_DIR     = "annotation"       # 標註 JSON 來源目錄：<video_key>.json
OUT_PRED_JSON_DIR  = "predict_json"     # 合併後預測 JSON 輸出目錄
CSV_OUT_DIR        = "csv_out"          # 斜率排名 CSV 輸出目錄
FIGURE_DIR         = "figures"          # 繪圖輸出目錄
SUMMARY_CSV        = os.path.join(FIGURE_DIR, "metrics_summary.csv")

FPS                = 30                 # 轉秒數用的 fps
TIOU_THRS          = [0.1 * i for i in range(1, 11)]  # 0.1 ~ 1.0
YLIM_MAX           = 0.5            # 圖上限（0~1），若要與您範例一致可設 0.5

# 類別映射（僅評估這兩類；其它類別一律忽略）
CAT_MAP = {1: "閱讀題目", 2: "思考解答"}
EVAL_LABELS = set(CAT_MAP.keys())
# ======================================


# ---------- 字型（支援中文） ----------
def _init_font_for_chinese():
    """
    嘗試載入常見中文字型；若找不到就退回預設（英文），圖仍可生成但標題/標籤可能顯示為方塊。
    您也可自行將字型檔加入系統後在此指定 family。
    """
    try:
        # macOS 常見：Heiti TC / PingFang TC；Windows：Microsoft JhengHei；Linux：Noto Sans CJK TC
        candidates = ["Noto Sans CJK TC", "PingFang TC", "Heiti TC", "Microsoft JhengHei", "Arial Unicode MS"]
        for name in candidates:
            try:
                matplotlib.font_manager.findfont(name, fallback_to_default=False)
                return matplotlib.font_manager.FontProperties(family=name)
            except Exception:
                continue
    except Exception:
        pass
    return matplotlib.font_manager.FontProperties()  # fallback


font_prop = font_manager.FontProperties(fname="font.ttc")


# ---------- 基礎工具 ----------
def parse_video_key_from_csv_filename(csv_path: str) -> str:
    """
    假定 CSV 檔名為 <video_key>_pred.csv → 回傳 <video_key>
    """
    base = os.path.basename(csv_path)
    if base.endswith("_pred.csv"):
        return base[:-9]
    return os.path.splitext(base)[0]


def parse_start_end_from_path(path_str: str) -> Tuple[int, int]:
    """
    由 path 欄位解析起訖 frame（整數）。
    規則：最後兩個底線分隔為 start, end，例如 11377034_label4_280_559
    """
    m = re.search(r"_([0-9]+)_([0-9]+)$", path_str)
    if not m:
        raise ValueError(f"path 無法解析起訖 frame：{path_str}")
    return int(m.group(1)), int(m.group(2))


def frames_to_seconds(start_f: int, end_f: int, fps: float = FPS) -> Tuple[float, float]:
    """
    以半開區間 [start_sec, end_sec) 表示：end 取 (end_f+1)/fps，可避免邊界重疊。
    """
    start_sec = start_f / fps
    end_sec   = (end_f + 1) / fps
    return start_sec, end_sec


def merge_consecutive_segments(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    將同一影片的 row（含 path、pred）轉為「合併後的不重疊區間」：
      - 僅保留 pred ∈ {1,2}
      - 以起始 frame 排序
      - 若與上一段 label 相同且相鄰（上段 end + 1 == 本段 start）則合併
    輸出：
      {"start_sec": float, "end_sec": float, "category": str}
    """
    segs = []
    for r in rows:
        label = int(r["pred"])
        if label not in EVAL_LABELS:
            continue
        s, e = parse_start_end_from_path(r["path"])
        segs.append((s, e, label))

    if not segs:
        return []

    segs.sort(key=lambda x: x[0])
    merged = []
    cur_s, cur_e, cur_l = segs[0]

    for s, e, l in segs[1:]:
        if l == cur_l and s == cur_e + 1:
            cur_e = e
        else:
            ss, ee = frames_to_seconds(cur_s, cur_e)
            merged.append({"start_sec": ss, "end_sec": ee, "category": CAT_MAP.get(cur_l, "未知")})
            cur_s, cur_e, cur_l = s, e, l

    ss, ee = frames_to_seconds(cur_s, cur_e)
    merged.append({"start_sec": ss, "end_sec": ee, "category": CAT_MAP.get(cur_l, "未知")})
    return merged


def load_predictions_and_merge(csv_path: str) -> List[Dict[str, Any]]:
    """
    載入單支影片之推理 CSV，合併連續片段。
    容忍欄名 'pred' 或 'Predicted Label'。
    """
    df = pd.read_csv(csv_path)
    if 'path' not in df.columns:
        raise ValueError(f"CSV 缺少欄位 'path'：{csv_path}")
    if 'pred' not in df.columns and 'Predicted Label' not in df.columns:
        raise ValueError(f"CSV 缺少欄位 'pred' 或 'Predicted Label'：{csv_path}")
    if 'pred' not in df.columns:
        df = df.rename(columns={'Predicted Label': 'pred'})

    rows = df[['path', 'pred']].to_dict(orient='records')
    return merge_consecutive_segments(rows)


def save_json(obj: Any, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def parse_gt_annotations(json_path: str) -> List[Dict[str, Any]]:
    """
    讀取標註 JSON：使用 User_annotation[0].tags 的 start/end（秒）與 category（中文）。
    僅保留屬於 CAT_MAP 的目標類別。
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    tags = data.get("User_annotation", [{}])[0].get("tags", [])
    allowed = set(CAT_MAP.values())
    gts = []
    for t in tags:
        if t.get("category") in allowed:
            gts.append({
                "start_sec": float(t["start"]),
                "end_sec": float(t["end"]),
                "category": t["category"]
            })
    return gts


def tiou(seg1: Dict[str, Any], seg2: Dict[str, Any]) -> float:
    a1, a2 = float(seg1["start_sec"]), float(seg1["end_sec"])
    b1, b2 = float(seg2["start_sec"]), float(seg2["end_sec"])
    inter = max(0.0, min(a2, b2) - max(a1, b1))
    union = max(a2, b2) - min(a1, b1)
    return (inter / union) if union > 0 else 0.0


def eval_detection_all(preds: List[Dict[str, Any]],
                       gts: List[Dict[str, Any]],
                       tiou_thr: float = 0.5) -> Tuple[float, float, float, float, int, int, int]:
    """
    單支影片的配對評估：類別一致 + tIoU≥門檻 → TP；每個 GT 僅能匹配一次。
    回傳：(P, R, F1, Acc, TP, FP, FN)
    """
    gt_used = [False] * len(gts)
    tp = fp = 0
    for p in preds:
        matched = False
        for i, g in enumerate(gts):
            if gt_used[i]:
                continue
            if p["category"] != g["category"]:
                continue
            if tiou(p, g) >= tiou_thr:
                tp += 1
                gt_used[i] = True
                matched = True
                break
        if not matched:
            fp += 1
    fn = len(gts) - sum(gt_used)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    acc       = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    return precision, recall, f1, acc, tp, fp, fn


# ---------- 主流程 ----------
def main():
    os.makedirs(OUT_PRED_JSON_DIR, exist_ok=True)
    os.makedirs(CSV_OUT_DIR, exist_ok=True)
    os.makedirs(FIGURE_DIR, exist_ok=True)

    # 取得所有推理 CSV
    csv_files = sorted(glob.glob(os.path.join(PRED_DIR, "*_pred.csv")))
    if not csv_files:
        print(f"找不到推理 CSV：{PRED_DIR}/*_pred.csv")
        return

    # (1) 合併各影片預測，並輸出 JSON
    video_keys: List[str] = []
    for csv_path in csv_files:
        video_key = parse_video_key_from_csv_filename(csv_path)
        video_keys.append(video_key)
        try:
            merged_preds = load_predictions_and_merge(csv_path)
        except Exception as e:
            print(f"[合併失敗] {csv_path} → {e}")
            continue
        out_json = os.path.join(OUT_PRED_JSON_DIR, f"{video_key}_predict.json")
        save_json(merged_preds, out_json)
        print(f"[OK] {video_key}: 合併後 {len(merged_preds)} 段 → {out_json}")

    if not video_keys:
        print("沒有可評估的影片（video_keys 為空）。")
        return

    # (2) 依 tIoU 門檻計算累積指標並繪圖 + 斜率輸出
    summary_rows: List[Dict[str, Any]] = []

    for tiou_thr in TIOU_THRS:
        precisions, recalls, f1s, accuracies, counts = [], [], [], [], []
        total_tp = total_fp = total_fn = 0
        cum_f1_list = []  # (idx, video_key, cum_F1)

        for i, vid in enumerate(video_keys, start=1):
            pfile = os.path.join(OUT_PRED_JSON_DIR, f"{vid}_predict.json")
            gfile = os.path.join(ANNOTATION_DIR, f"{vid}.json")
            if not (os.path.exists(pfile) and os.path.exists(gfile)):
                print(f"[略過] 找不到 p/g：{pfile} 或 {gfile}")
                continue

            preds = json.load(open(pfile, encoding="utf-8"))
            gts   = parse_gt_annotations(gfile)

            _, _, f1, _, tp, fp, fn = eval_detection_all(preds, gts, tiou_thr)
            total_tp += tp
            total_fp += fp
            total_fn += fn

            precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
            recall    = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
            F1        = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
            acc       = total_tp / (total_tp + total_fp + total_fn) if (total_tp + total_fp + total_fn) > 0 else 0.0

            precisions.append(precision)
            recalls.append(recall)
            f1s.append(F1)
            accuracies.append(acc)
            counts.append(i)

            summary_rows.append({
                "tIoU": tiou_thr,
                "影片數": i,
                "Precision": precision,
                "Recall": recall,
                "F1": F1,
                "Accuracy": acc,
            })

            cum_f1_list.append((i, vid, F1))

        # === 繪圖 ===
        # Precision & Recall
        plt.figure(figsize=(8, 5))
        plt.plot(counts, precisions, marker='o', label='Precision')
        plt.plot(counts, recalls,    marker='s', label='Recall')
        plt.xlabel("累積評估影片數", fontproperties=font_prop)
        plt.ylabel("指標值", fontproperties=font_prop)
        plt.title(f"tIoU={tiou_thr:.2f} Precision & Recall", fontproperties=font_prop)
        plt.grid(True)
        plt.legend(prop=font_prop)
        plt.xticks(counts)
        plt.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        plt.ylim(0, YLIM_MAX)
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURE_DIR, f"precision_recall_tIoU_{tiou_thr:.2f}.png"), dpi=300)
        plt.close()

        # F1
        plt.figure(figsize=(8, 5))
        plt.plot(counts, f1s, marker='^', label='F1')
        plt.xlabel("累積評估影片數", fontproperties=font_prop)
        plt.ylabel("F1 值", fontproperties=font_prop)
        plt.title(f"tIoU={tiou_thr:.2f} F1 Score", fontproperties=font_prop)
        plt.grid(True)
        plt.legend(prop=font_prop)
        plt.xticks(counts)
        plt.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        plt.ylim(0, YLIM_MAX)
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURE_DIR, f"f1_tIoU_{tiou_thr:.2f}.png"), dpi=300)
        plt.close()

        # Accuracy
        plt.figure(figsize=(8, 5))
        plt.plot(counts, accuracies, marker='D', label='Accuracy')
        plt.xlabel("累積評估影片數", fontproperties=font_prop)
        plt.ylabel("Accuracy 值", fontproperties=font_prop)
        plt.title(f"tIoU={tiou_thr:.2f} Accuracy", fontproperties=font_prop)
        plt.grid(True)
        plt.legend(prop=font_prop)
        plt.xticks(counts)
        plt.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        plt.ylim(0, YLIM_MAX)
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURE_DIR, f"accuracy_tIoU_{tiou_thr:.2f}.png"), dpi=300)
        plt.close()

        # === 斜率（ΔF1）輸出 ===
        slopes = []
        for j in range(1, len(cum_f1_list)):
            prev_idx, prev_vid, prev_f1 = cum_f1_list[j - 1]
            cur_idx,  cur_vid,  cur_f1  = cum_f1_list[j]
            slope = cur_f1 - prev_f1
            slopes.append((cur_idx, cur_vid, slope))

        slopes_sorted = sorted(slopes, key=lambda x: x[2])
        csv_path = os.path.join(CSV_OUT_DIR, f"slopes_tIoU_{tiou_thr:.2f}.csv")
        with open(csv_path, "w", newline='', encoding="utf-8") as csvf:
            writer = csv.writer(csvf)
            writer.writerow(["累積序號", "video_key", "ΔF1"])
            for idx, vid, slope in slopes_sorted:
                writer.writerow([idx, vid, slope])

        # 簡要列印 Top-3 最負
        print(f"\n=== tIoU = {tiou_thr:.2f}：品質最差影片 Top 3 ===")
        for rank, (idx, vid, slope) in enumerate(slopes_sorted[:3], start=1):
            print(f"{rank}. 第 {idx} 部 ({vid}) → ΔF1 = {slope:.4f}")

    # (3) 匯總 CSV
    os.makedirs(FIGURE_DIR, exist_ok=True)
    with open(SUMMARY_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["tIoU", "影片數", "Precision", "Recall", "F1", "Accuracy"])
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"\n✅ 已輸出圖與彙整：\n  - 圖片：{FIGURE_DIR}\n  - 斜率：{CSV_OUT_DIR}\n  - 彙整：{SUMMARY_CSV}")


if __name__ == "__main__":
    main()
