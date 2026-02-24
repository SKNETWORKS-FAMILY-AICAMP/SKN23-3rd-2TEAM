import os
import subprocess
import shutil
import glob
from concurrent.futures import ThreadPoolExecutor

# 1. 경로 및 설정
RCLONE_SOURCE = "gdrive:RAW_DATA/" 
TODO_DIR = "/workspace/todo_pdfs"
OUTPUT_DIR = "/workspace/output"
NUM_GPUS = 6

def cleanup_previous_errors():
    """이전 작업에서 멈춘 마커(marker) 좀비 프로세스와 GPU 메모리를 강제 초기화합니다."""
    print("System check: Cleaning up zombie processes and resetting GPU memory...")
    # 현재 실행 중인 파이썬 스크립트 자신은 죽이지 않고 marker 관련 프로세스만 종료
    subprocess.run("pkill -9 -f marker || true", shell=True, stderr=subprocess.DEVNULL)
    # 모든 GPU의 찌꺼기 메모리 점유 해제
    subprocess.run("for i in {0..5}; do fuser -kv /dev/nvidia$i; done || true", shell=True, stderr=subprocess.DEVNULL)
    print("System check completed. Clean start.")

def download_with_rclone():
    """rclone을 사용하여 구글 드라이브에서 PDF 파일들을 로컬로 다운로드합니다."""
    os.makedirs(TODO_DIR, exist_ok=True)
    print(f"Starting PDF download from {RCLONE_SOURCE} using rclone...")
    
    cmd = [
        "rclone", "copy", RCLONE_SOURCE, TODO_DIR,
        "--progress"
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print("Download completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Critical Error during rclone download: {e}")
        raise

def run_on_gpu(gpu_id, pdf_batch):
    """할당된 PDF 배치를 특정 GPU에서 실행합니다."""
    if not pdf_batch: 
        return
    
    gpu_todo = f"/workspace/todo_gpu_{gpu_id}"
    os.makedirs(gpu_todo, exist_ok=True)
    
    for p in pdf_batch:
        target_path = os.path.join(gpu_todo, os.path.basename(p))
        if p != target_path:
            shutil.move(p, target_path)
    
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    cmd = [
        "marker", gpu_todo,
        "--output_dir", OUTPUT_DIR,
        "--workers", "2" 
    ]
    
    # 에러 방지 처리: 프로세스가 실패하면 무한 대기하지 않고 에러를 출력 후 해당 GPU 작업만 중단
    try:
        subprocess.run(cmd, env=env, check=True)
        print(f"GPU {gpu_id} completed its task successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred on GPU {gpu_id}. Process halted for this batch. Error: {e}")

def main():
    # 1. 시작 전 에러 요인(좀비 프로세스) 싹 정리
    cleanup_previous_errors()
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 2. rclone으로 원본 PDF 다운로드
    download_with_rclone()

    # 3. 작업 대상 PDF 목록 재정비
    all_found_pdfs = glob.glob("/workspace/todo_gpu_*/*.pdf") + glob.glob(os.path.join(TODO_DIR, "*.pdf"))
    print(f"Total {len(all_found_pdfs)} remaining files will be distributed to {NUM_GPUS} GPUs.")

    if not all_found_pdfs:
        print("No PDF files found to process. Exiting.")
        return

    # 4. 파일을 6개의 그룹으로 나누기
    batches = [all_found_pdfs[i::NUM_GPUS] for i in range(NUM_GPUS)]

    # 5. 분산 실행 시작
    print("Starting parallel execution on 6 GPUs (Safe margin mode)...")
    with ThreadPoolExecutor(max_workers=NUM_GPUS) as executor:
        for i in range(NUM_GPUS):
            executor.submit(run_on_gpu, i, batches[i])

    print("All distributed parsing tasks have been completed.")

if __name__ == "__main__":
    main()