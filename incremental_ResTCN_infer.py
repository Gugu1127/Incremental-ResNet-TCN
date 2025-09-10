#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import subprocess
import os
import csv
import time
import pathlib
import re
from datetime import datetime, timezone

# ===================== 可調參數 =====================
ENV_DIR         = "ResNet"            # 虛擬環境目錄（ENV_DIR/bin/activate）
ANNOTATION_DIR  = "annotation"        # JSON 所在資料夾
MODELS_DIR      = "models"            # 最佳權重所在資料夾
PRED_DIR        = "predictions"       # 推理輸出資料夾
INFER_IMG_ROOT  = "Infer_images"      # 既有的推理影像根目錄
TIMELOG         = "timelog_infer.csv" # 執行記錄檔
FORCE_CKPT      = ""                  # << 可填 "models/best_model_22.pth"；留空則自動挑最新
# ===================================================

def iso_now():
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")

def run_script(command, env_dir=ENV_DIR):
    if isinstance(command, list):
        command_str = ' '.join(str(c) for c in command)
    else:
        command_str = command
    activate_path = os.path.join(env_dir, "bin", "activate")
    full_command = f"source {activate_path} && {command_str}"
    print(f"👉 執行：{full_command}", flush=True)
    result = subprocess.run(
        full_command, shell=True, capture_output=True, text=True, executable='/bin/bash'
    )
    if result.returncode != 0:
        print(f"❌ 執行失敗：{full_command}", flush=True)
        if result.stderr:
            print(result.stderr, flush=True)
    else:
        if result.stdout:
            print(result.stdout, flush=True)
    return {
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "command_str": full_command
    }

def measure_exec(command, env_dir=ENV_DIR):
    start_ts = iso_now()
    t0 = time.perf_counter()
    res = run_script(command, env_dir=env_dir)
    elapsed = time.perf_counter() - t0
    end_ts = iso_now()
    return elapsed, res, start_ts, end_ts

def ensure_timelog(csv_path=TIMELOG):
    need_header = not os.path.exists(csv_path)
    if need_header:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "index",
                "json_file",
                "infer_start",
                "infer_end",
                "infer_seconds",
                "infer_returncode",
                "infer_status",
                "infer_ckpt",
                "infer_csv_in",
                "infer_csv_out",
            ])

def append_timelog(row, csv_path=TIMELOG):
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(row)

def list_available_ckpts(models_dir=MODELS_DIR):
    if not os.path.isdir(models_dir):
        return []
    pattern = re.compile(r"best_model_(\d+)\.pth$")
    found = []
    for name in os.listdir(models_dir):
        m = pattern.match(name)
        if m:
            found.append((int(m.group(1)), os.path.join(models_dir, name)))
    found.sort(key=lambda x: x[0])  # 依編號排序
    return found

def pick_checkpoint_for_index(idx: int, models_dir: str = MODELS_DIR, strict: bool = True) -> str:
    """
    嚴格對應：第 idx 部影片 → 使用 models/best_model_{idx}.pth
    strict=True 時：若檔案不存在，回傳空字串（外層可選擇報錯或跳過）
    strict=False 時：若不存在則回傳空字串（行為相同，保留參數以利擴充）
    """
    ckpt = os.path.join(models_dir, f"best_model_{idx}.pth")
    if os.path.exists(ckpt):
        return ckpt
    # 不做回退，直接回傳空字串
    return ""

def main():
    ensure_timelog(TIMELOG)
    os.makedirs(PRED_DIR, exist_ok=True)

    # 如需變更清單，直接修改此處
    files = [
        # 'online_exam1_11027261_user_2wOEHNkaoNPdXNbHKmgezQWhkKT.json',
        'online_exam1_11127231_user_2x1eNMA2ywMBAa7iN5XuKH8wXPK.json',
        'online_exam1_11227170_user_2x1eXL6anwZQ3IIZPiiwykDbX1u.json',
        'online_exam1_11227218_user_2wLdBAPqVQJuFroDWby3mrZZpTH.json',
        'online_exam2_11027132_user_2wNqlMx9sSQJvK9yP7IzqFVaHKd.json',
        'online_exam2_11127164_user_2wNteozcl5DKR9DydgB9xQmchhR.json',
        'online_exam2_11227151_user_2wNrt7AJhsR854YlJ52LIxOohc4.json',
        'online_exam2_11227170_user_2x1eXL6anwZQ3IIZPiiwykDbX1u.json',
        'online_exam2_11227240_user_2wOCPJUZLfX1YpEOQMYpXufo0z3.json',
        'online_exam2_11227249_user_2wOCg2IRON9jxgk4cZcTVJ2hQpw.json',
        'online_exam2_11227253_user_2wOGDRY5OdrmN5IMeehgEaavp76.json',
        'online_exam3_11120125_user_2x1eJTEUaPWEQdzKxL3QmNyQl2y.json',
        'online_exam3_11124107_user_2wNqgnqMwUS48l5xie63msBUq6y.json',
        'online_exam3_11227132_user_2wNuJlqfKxrPEfinDtKY1saJzAJ.json',
        'online_exam4_11127164_user_2wNteozcl5DKR9DydgB9xQmchhR.json',
        'online_exam5_11027261_user_2wOEHNkaoNPdXNbHKmgezQWhkKT.json',
        'online_exam5_11227134_user_2x1eldFL11fChivMXucYVnamfhT.json',
        'online_exam5_11227251_user_2x1gJw6X3DAdKk8HvmqZ7wjHrku.json',
        'online_exam5_11227253_user_2wOEHNkaoNPdXNbHKmgezQWhkKT.json',
        'online_test4_11377005_user_2vt9vAdjkpf200pm0RmM8pq2lZt.json',
        'online_test4_11377009_user_2w1MjpWogJP6YKlvXIXjUj2Orpa.json',
        'online_test4_11377034_user_2isVlxzxXqw8ALMYICvsTChJhrO.json'
    ]
    
    for idx, file in enumerate(files, 1):
        print(f"\n===== [{idx}/{len(files)}] 推理檔案：{file} =====", flush=True)
        json_file_path = os.path.join(ANNOTATION_DIR, file)
        run_key = pathlib.Path(json_file_path).stem

        infer_root   = os.path.join(INFER_IMG_ROOT, run_key)
        infer_csv_in = os.path.join(infer_root, 'infer.csv')
        infer_csv_out= os.path.join(PRED_DIR, f'{run_key}_pred.csv')

        # 檢查輸入是否齊備
        if not os.path.isdir(infer_root):
            print(f"⚠️ 找不到推理影像根目錄：{infer_root}，略過。", flush=True)
            append_timelog([idx, file, "", "", "", "", "skipped_missing_infer_root", "", infer_csv_in, ""], TIMELOG)
            continue
        if not os.path.isfile(infer_csv_in):
            print(f"⚠️ 找不到 infer.csv：{infer_csv_in}，略過。", flush=True)
            append_timelog([idx, file, "", "", "", "", "skipped_missing_infer_csv", "", infer_csv_in, ""], TIMELOG)
            continue

        # ★ 關鍵：第 i 部影片用 best_model_i.pth
        ckpt = pick_checkpoint_for_index(idx, models_dir=MODELS_DIR, strict=True)
        if not ckpt:
            print(f"⚠️ 找不到對應權重：{os.path.join(MODELS_DIR, f'best_model_{idx}.pth')}，略過推理。", flush=True)
            append_timelog([idx, file, "", "", "", "", "skipped_missing_checkpoint", "", infer_csv_in, ""], TIMELOG)
            continue

        print(f"🧠 使用權重進行推理：{ckpt}", flush=True)
        infer_cmd = [
            'python', 'infer.py',
            '--ckpt', f'"{ckpt}"',
            '--csv',  f'"{infer_csv_in}"',
            '--root', f'"{infer_root}"',
            '--out',  f'"{infer_csv_out}"'
        ]
        infer_elapsed, infer_res, infer_start, infer_end = measure_exec(infer_cmd, env_dir=ENV_DIR)
        infer_status = "success" if infer_res["returncode"] == 0 else "failed"
        print(f"⏱️ infer 耗時：{infer_elapsed:.3f} 秒（狀態：{infer_status}）", flush=True)

        append_timelog([
            idx, file,
            infer_start,
            infer_end,
            f"{infer_elapsed:.3f}",
            infer_res["returncode"],
            infer_status,
            ckpt,
            infer_csv_in,
            infer_csv_out if infer_status == "success" else "",
        ], TIMELOG)

    print("\n📄 已將逐次耗時與狀態寫入", TIMELOG, flush=True)
    print("🎯 流程完成：第 i 部影片固定使用 best_model_i.pth 進行推理並儲存結果。", flush=True)

if __name__ == "__main__":
    main()
