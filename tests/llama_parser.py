import os
import asyncio # 비동기 처리를 위해 추가
from dotenv import load_dotenv
from llama_parse import LlamaParse

load_dotenv()

# 비동기 함수로 변경
async def run_llama_parse_async():
    input_dir = "data/1_pdf"
    output_dir = "data/2_llama_md"
    os.makedirs(output_dir, exist_ok=True)

    # 1. 수정된 프롬프트: 영어 기술 용어 보존 및 유연성 확보 (기존 내용 완벽 유지)
    revised_system_prompt = """
    당신은 RAG 시스템을 위한 정밀한 텍스트 추출기입니다.
    제공된 용접 기술 문서에서 텍스트와 표(Table)를 마크다운 형식으로 추출하세요.

    [작업 원칙]
    1. **정확성**: 원문의 내용을 요약하거나 왜곡하지 말고 그대로 추출하세요.
    2. **언어 (매우 중요)**: 한국어 문서는 한국어로 출력하되, 기술 용어, 부품명, 규격(예: TIG, STS304, Sch.40) 등의 영어 표기는 원문 그대로 유지하세요. 억지로 번역하지 마세요.
    3. **표(Table)**: 표는 반드시 마크다운 표 문법(|---|)을 사용하여 구조를 유지하세요.
    4. **여백 제외**: 페이지 번호, 머리말/꼬리말 같은 단순 반복 요소는 제외하세요. (단, 본문 내용은 절대 건드리지 마세요.)
    5. **환각 및 묘사 금지 (안전 장치)**: 도면이나 이미지를 보고 임의로 영어 제목(예: "Technical Content Extraction")이나 설명을 지어내지 마세요. 이미지는 무시하세요.
    6. **에러 메시지 차단 (안전 장치)**: 파싱할 수 없는 영역이 있더라도 "NO_CONTENT_HERE"나 "(내용이 누락되어 있습니다)" 같은 시스템 에러 문구를 결과물에 포함하지 마세요.
    """

    api_key = os.getenv("LLAMA_CLOUD_API_KEY")
    if not api_key:
        print("API Key Missing")
        return

    parser = LlamaParse(
        api_key=api_key, # 명시적 키 전달
        result_type="markdown",
        language="ko",
        system_prompt=revised_system_prompt,
        verbose=True
    )

    # 입력 디렉토리(1_pdf) 내의 모든 PDF 파일을 탐색하여 목록 생성 (os.walk 사용)
    pdf_files_info = []
    for root, dirs, files in os.walk(input_dir): # 하위 폴더까지 순회
        for filename in files:
            if filename.lower().endswith('.pdf'): # 확장자가 pdf인 파일만 필터링
                pdf_files_info.append({
                    "root": root,
                    "filename": filename
                })

    # 비동기 병렬 처리를 위한 작업(Task) 리스트 생성
    tasks = []
    for file_info in pdf_files_info:
        root = file_info["root"]
        filename = file_info["filename"]
        file_path = os.path.join(root, filename) # 원본 PDF 파일의 전체 경로
        
        # 원본 구조를 유지하기 위해 1_pdf의 하위 폴더 경로를 대상 폴더(2_llama_md)에 동일하게 생성
        relative_path = os.path.relpath(root, input_dir)
        target_dir = os.path.join(output_dir, relative_path)
        os.makedirs(target_dir, exist_ok=True) # 대상 디렉토리가 없으면 생성
        
        # 출력 파일명 지정 (.pdf -> .md)
        output_filename = filename.replace(".pdf", ".md")
        output_path = os.path.join(target_dir, output_filename)
        
        # 이미 변환된 마크다운 파일이 존재하면 건너뜀 (중복 방지)
        if os.path.exists(output_path):
            continue
            
        # 변환이 필요한 파일만 비동기 처리 작업 리스트에 추가
        tasks.append(process_single_file(parser, file_path, output_path, filename))

    # 생성된 모든 변환 작업들을 동시에(비동기 병렬로) 실행하여 속도 최적화
    if tasks:
        await asyncio.gather(*tasks)
    else:
        pass

async def process_single_file(parser, file_path, output_path, filename):
    """
    단일 PDF 파일을 마크다운 파일로 변환하여 저장하는 비동기 함수입니다.
    """
    try:
        # 비동기 메서드 aload_data를 사용하여 LlamaParse API 호출 및 문서 텍스트 추출
        documents = await parser.aload_data(file_path)
        
        # 여러 페이지로 나뉜 파싱 결과를 하나의 마크다운 텍스트로 병합
        full_markdown = "\n\n".join([doc.text for doc in documents])
        
        # 추출된 마크다운 텍스트를 대상 경로에 파일로 저장 (UTF-8 인코딩)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(full_markdown)

    except Exception as e:
        # 작업 중 오류 발생 시 에러 메시지 출력
        print(f"실패 ({filename}): {e}")

if __name__ == "__main__":
    # 비동기 실행 루프
    asyncio.run(run_llama_parse_async())