import subprocess
import os
import shutil
import csv
import time
from datetime import datetime, timezone

def iso_now():
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")

def remove_data():
    folder_path = 'images_train'
    if os.path.exists(folder_path):
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
    csv_file = "train.csv"
    if os.path.exists(csv_file):
        os.remove(csv_file)
    print("✅ 清空 images_train 與 train.csv 完成", flush=True)

def run_script(command, env_dir="ResNet"):
    """
    在指定的 venv 虛擬環境下執行指令，回傳執行結果。
    :param command: list 或 str, 要執行的指令
    :param env_dir: str, venv 環境目錄名稱，預設 "ResNet"
    :return: dict {'returncode': int, 'stdout': str, 'stderr': str, 'command_str': str}
    """
    # 處理指令字串
    if isinstance(command, list):
        command_str = ' '.join(str(c) for c in command)
    else:
        command_str = command

    # venv 的啟動腳本路徑
    activate_path = os.path.join(env_dir, "bin", "activate")
    full_command = f"source {activate_path} && {command_str}"

    print(f"👉 執行：{full_command}", flush=True)
    # 注意：subprocess 必須在 shell=True 下用 source
    result = subprocess.run(
        full_command,
        shell=True,
        capture_output=True,
        text=True,
        executable='/bin/bash'
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

def measure_exec(command, env_dir="ResNet"):
    """
    量測單次指令執行時間並回傳耗時（秒）與執行結果。
    """
    start_ts = iso_now()
    t0 = time.perf_counter()
    res = run_script(command, env_dir=env_dir)
    elapsed = time.perf_counter() - t0
    end_ts = iso_now()
    return elapsed, res, start_ts, end_ts

def ensure_timelog(csv_path="timelog.csv"):
    """
    若 timelog.csv 不存在則建立含表頭的檔案。
    """
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
    remove_data()

    # 先確保 timelog.csv 存在且有表頭
    ensure_timelog("timelog.csv")

    json_folder_path = 'annotation'
    files = [
        # 'online_exam1_11027261_user_2wOEHNkaoNPdXNbHKmgezQWhkKT.json',
        # 'online_exam1_11127231_user_2x1eNMA2ywMBAa7iN5XuKH8wXPK.json',
        # 'online_exam1_11227170_user_2x1eXL6anwZQ3IIZPiiwykDbX1u.json',
        # 'online_exam1_11227218_user_2wLdBAPqVQJuFroDWby3mrZZpTH.json',
        # 'online_exam2_11027132_user_2wNqlMx9sSQJvK9yP7IzqFVaHKd.json',
        # 'online_exam2_11127164_user_2wNteozcl5DKR9DydgB9xQmchhR.json',
        # 'online_exam2_11227151_user_2wNrt7AJhsR854YlJ52LIxOohc4.json',
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
        print(f"\n----- [{idx}/{len(files)}] 處理檔案：{file} -----", flush=True)
        json_file_path = os.path.join(json_folder_path, file)

        # === preprocess ===
        preprocess_cmd = ['python', 'preprocess.py', '--file-path', f'"{json_file_path}"']
        pre_elapsed, pre_res, pre_start, pre_end = measure_exec(preprocess_cmd, env_dir="ResNet")
        pre_status = "success" if pre_res["returncode"] == 0 else "failed"

        print(f"⏱️ preprocess 耗時：{pre_elapsed:.3f} 秒（狀態：{pre_status}）", flush=True)

        # === train ===
        train_start = train_end = ""
        train_elapsed = ""
        train_returncode = ""
        train_status = "skipped_due_to_preprocess_error"

        if pre_res["returncode"] == 0:
            train_cmd = ['python', 'train.py', '--ID', idx]
            train_elapsed_val, train_res, train_start, train_end = measure_exec(train_cmd, env_dir="ResNet")
            train_elapsed = f"{train_elapsed_val:.3f}"
            train_returncode = train_res["returncode"]
            train_status = "success" if train_res["returncode"] == 0 else "failed"
            print(f"⏱️ train 耗時：{float(train_elapsed):.3f} 秒（狀態：{train_status}）", flush=True)
        else:
            print("⚠️ 因 preprocess 失敗，略過 train。", flush=True)

        # === 寫入 timelog.csv ===
        append_timelog([
            idx,
            file,
            pre_start,
            pre_end,
            f"{pre_elapsed:.3f}",
            pre_res["returncode"],
            pre_status,
            train_start,
            train_end,
            train_elapsed,
            train_returncode,
            train_status
        ], csv_path="timelog.csv")

    print("\n📄 已將逐次耗時與狀態寫入 timelog.csv", flush=True)
