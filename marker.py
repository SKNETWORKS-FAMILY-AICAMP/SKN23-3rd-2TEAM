import subprocess
import gdown
import os

# =========================
# 1. Google Drive 폴더 다운로드
# =========================

folder_id = "1RGMVBBQBfaeiVPGXYxiQu8uOUtTHRsSB" # 여기에_폴더ID_입력
drive_url = f"https://drive.google.com/drive/folders/{folder_id}"

download_dir = "./safety"
os.makedirs(download_dir, exist_ok=True)

print("📥 Google Drive 폴더 다운로드 중...")
gdown.download_folder(drive_url, output=download_dir, quiet=False, use_cookies=False)

print("다운로드 완료!")

# =========================
# 2. Marker 배치 실행 (GPU 사용 가능)
# =========================

output_dir = "./output"
os.makedirs(output_dir, exist_ok=True)

print("Marker 배치 실행 중...")

subprocess.run([
    "marker_batch",
    download_dir,
    "--output_format", "json",
    "--output_dir", output_dir,
    "--device", "cuda"   # GPU 사용 (없으면 삭제)
])

subprocess.run([
    "marker_batch",
    download_dir,
    "--output_format", "markdown",
    "--output_dir", output_dir,
    "--device", "cuda"
])

print("파싱 완료!")