import subprocess
import os
import shutil
import csv
import time
import json
import pathlib
from datetime import datetime, timezone

def iso_now():
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def run_script(command, env_dir="ResNet"):
    if isinstance(command, list):
        command_str = ' '.join(str(c) for c in command)
    else:
        command_str = command
    activate_path = os.path.join(env_dir, "bin", "activate")
    full_command = f"source {activate_path} && {command_str}"
    print(f"👉 執行：{full_command}", flush=True)
    result = subprocess.run(full_command, shell=True, capture_output=True, text=True, executable='/bin/bash')
    if result.returncode != 0:
        print(f"❌ 執行失敗：{full_command}", flush=True)
        if result.stderr:
            print(result.stderr, flush=True)
    else:
        if result.stdout:
            print(result.stdout, flush=True)
    return {"returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr, "command_str": full_command}

def measure_exec(command, env_dir="ResNet"):
    start_ts = iso_now()
    t0 = time.perf_counter()
    res = run_script(command, env_dir=env_dir)
    elapsed = time.perf_counter() - t0
    end_ts = iso_now()
    return elapsed, res, start_ts, end_ts

def ensure_timelog(csv_path="timelog.csv"):
    need_header = not os.path.exists(csv_path)
    if need_header:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "index",
                "json_file",
                "preprocess_start",
                "preprocess_end",
                "preprocess_seconds",
                "preprocess_returncode",
                "preprocess_status",
                "train_start",
                "train_end",
                "train_seconds",
                "train_returncode",
                "train_status"
            ])

def append_timelog(row, csv_path="timelog.csv"):
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(row)

if __name__ == "__main__":
    ensure_timelog("timelog.csv")

    json_folder_path = 'annotation'
    files = [
        'online_exam1_11027261_user_2wOEHNkaoNPdXNbHKmgezQWhkKT.json',
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
    files.reverse()  # 👈 這行也可以

    prev_best = None  # 用於跨影片的預訓練權重鏈結

    for idx, file in enumerate(files, 1):
        print(f"\n----- [{idx}/{len(files)}] 處理檔案：{file} -----", flush=True)
        json_file_path = os.path.join(json_folder_path, file)
        run_key = pathlib.Path(json_file_path).stem  # 以 JSON 檔名（去 .json）

        # === preprocess ===
        # preprocess_cmd = ['python', 'preprocess.py', '--file-path', f'"{json_file_path}"']
        # pre_elapsed, pre_res, pre_start, pre_end = measure_exec(preprocess_cmd, env_dir="ResNet")
        # pre_status = "success" if pre_res["returncode"] == 0 else "failed"
        # print(f"⏱️ preprocess 耗時：{pre_elapsed:.3f} 秒（狀態：{pre_status}）", flush=True)

        # === train ===
        train_start = train_end = ""
        train_elapsed = ""
        train_returncode = ""
        train_status = "skipped_due_to_preprocess_error"

        data_root = os.path.join('Train_images', run_key)  # 👈 以 run_key 當資料根目錄（單片）
        csv_in_run = os.path.join(data_root, 'train.csv')
        train_cmd = [
            'python', 'train.py',
            '--ID', idx,
            '--data-root', f'"{data_root}"',
            '--csv-file',  f'"{csv_in_run}"'
        ]
        if prev_best:
            train_cmd += ['--pretrained', f'"{prev_best}"']

        train_elapsed_val, train_res, train_start, train_end = measure_exec(train_cmd, env_dir="ResNet")
        train_elapsed = f"{train_elapsed_val:.3f}"
        train_returncode = train_res["returncode"]
        train_status = "success" if train_res["returncode"] == 0 else "failed"
        print(f"⏱️ train 耗時：{float(train_elapsed):.3f} 秒（狀態：{train_status}）", flush=True)

        # 下一輪的預訓練權重
        candidate_best = os.path.join('models', f'best_model_{idx}.pth')
        if train_res["returncode"] == 0 and os.path.exists(candidate_best):
            prev_best = candidate_best
            print(f"➡️ 下一輪將以此最佳權重作為預訓練：{prev_best}", flush=True)

        append_timelog([
            idx, file,
            train_start, train_end, train_elapsed, train_returncode, train_status
        ], csv_path="timelog.csv")

    print("\n📄 已將逐次耗時與狀態寫入 timelog.csv", flush=True)
