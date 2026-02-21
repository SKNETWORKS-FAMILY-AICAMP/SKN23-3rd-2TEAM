import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

# 한글 폰트 설정 (Mac 기준)
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

load_dotenv()

def visualize_embeddings():
    """
    Chroma DB에 저장된 고차원 임베딩(Embedding) 벡터 데이터를 2차원으로 축소하여
    시각화(그래프)하는 함수입니다.
    이를 통해 문서(청크)들이 벡터 공간에서 어떻게 군집화(Clustering)되어 있는지 확인할 수 있습니다.
    """
    persist_directory = "./db_welding"
    
    if not os.path.exists(persist_directory):
        print(f"{persist_directory} 폴더가 없습니다. 먼저 vector_store.py를 실행하세요.")
        return

    # 1. DB 로드
    # 텍스트를 벡터로 변환할 때 사용했던 동일한 모델을 지정해야 올바르게 불러올 수 있습니다.
    embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_db = Chroma(persist_directory=persist_directory, embedding_function=embeddings_model)
    
    # 2. 모든 데이터와 벡터 가져오기
    # Chroma DB 내부에 저장된 실제 임베딩 벡터 값과 텍스트 내용(documents)을 모두 추출합니다.
    data = vector_db.get(include=['embeddings', 'documents'])
    embeddings = np.array(data['embeddings'])
    documents = data['documents']
    
    if len(embeddings) == 0:
        print("저장된 데이터가 없습니다.")
        return

    # 3. t-SNE를 이용한 차원 축소 (고차원 -> 2차원)
    # 1536 차원에 달하는 고차원 벡터를 사람이 볼 수 있는 X, Y 2차원 평면 공간으로 변환합니다.
    # perplexity는 데이터의 개수에 따라 적절히 조절해야 군집이 잘 보입니다.
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings)-1))
    embeddings_2d = tsne.fit_transform(embeddings)

    # 4. 시각화 (Matplotlib)
    plt.figure(figsize=(12, 10))
    
    # 모든 벡터 데이터를 산점도(Scatter) 점으로 찍기
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.6, c='royalblue', edgecolors='w')

    # 일부 점에 텍스트 라벨 달기 (모두 달면 도면이 겹쳐서 안 보이므로 15개 정도 샘플링)
    for i in range(0, len(documents), len(documents)//15):
        # 텍스트 내용의 첫 20자만 가져와서 말줄임 처리
        label = documents[i][:20].replace('\n', ' ') + "..."
        plt.annotate(label, (embeddings_2d[i, 0], embeddings_2d[i, 1]), 
                     fontsize=9, alpha=0.8, backgroundcolor='white')

    plt.title("용접 에이전트 지식 베이스 (Vector Space Visualization)")
    plt.xlabel("차원 1 (Dimension 1)")
    plt.ylabel("차원 2 (Dimension 2)")
    plt.grid(True, linestyle='--', alpha=0.3)
    
    # 생성된 그래프 결과를 이미지 파일로 저장
    plt.savefig("welding_db_visual.png")

    # 화면에 그래프 출력
    plt.show()

if __name__ == "__main__":
    visualize_embeddings()