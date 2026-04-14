# Retriever Module — RAG Baseline

LangChain + FAISS 기반 RAG Retriever 베이스라인

전체 파이프라인에서 **Retriever** 담당 모듈입니다.

```
Memory State Generator → Router → [Retriever] → Answer Generator
                                       ↑
                                    이 모듈
```

---

## 역할

Router로부터 query를 받아 PDF 문서에서 관련 context를 검색하고,
Answer Generator로 context 문자열을 전달합니다.

```python
# 외부 연결 시 사용 방법
from retriever.rag_baseline import RAGPipeline

context = pipeline.get_context(query)      # Retriever 출력
answer  = answer_generator(query, context) # Answer Generator 입력
```

---

## 파일 구조

```
rag_baseline.py
        │
        ├── PDFProcessor         → PDF 파싱 + RecursiveCharacterTextSplitter 청킹
        ├── EmbeddingManager     → HuggingFace 임베딩 모델 (jhgan/ko-sroberta-multitask)
        ├── FAISSVectorStore     → FAISS 인덱스 빌드/저장/로드/검색 + md5 캐싱
        ├── RAGChain             → LangChain LCEL 체인 조립
        └── RAGPipeline          → 메인 진입점
```

---

## 설치

```bash
pip install -r requirements.txt
```

---

## 사용법

### .py import 방식 (다른 모듈 연결 시)

```python
from retriever.rag_baseline import RAGPipeline

# 1. LLM 설정
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-5-mini", temperature=0)

# from langchain_community.llms import Ollama
# llm = Ollama(model="qwen2.5:7b")

# 2. 파이프라인 초기화
pipeline = RAGPipeline(
    llm=llm,
    embedding_model="jhgan/ko-sroberta-multitask",
    chunk_size=1024,
    chunk_overlap=100,
    top_k=3,
)

# 3. PDF 로드
pipeline.load_pdf("논문.pdf")

# 4. context 반환 → Answer Generator로 전달
context = pipeline.get_context("질문")
```

### .ipynb 방식 (셀별 실행 / 테스트)

`retriever/rag_baseline.ipynb` 열어서 순서대로 실행

---

## 주요 파라미터

| 파라미터 | 기본값 | 설명 |
|---------|--------|------|
| `embedding_model` | `jhgan/ko-sroberta-multitask` | 한국어 경량 모델. 다국어 필요 시 `BAAI/bge-m3`로 교체 |
| `chunk_size` | `1024` | chunk 최대 문자 수. 클수록 문맥 풍부, 작을수록 검색 정밀도 향상 |
| `chunk_overlap` | `100` | chunk 간 겹침. chunk_size의 10% 수준 권장 |
| `top_k` | `3` | 검색할 chunk 수 |
| `cache_dir` | `./faiss_cache` | FAISS 캐시 저장 경로 |

---

## 캐싱

동일한 PDF는 md5 해시로 감지하여 재임베딩하지 않습니다.
캐시 파일은 `faiss_cache/` 폴더에 저장됩니다.

```
faiss_cache/
    {md5해시}.faiss  ← FAISS 인덱스
    {md5해시}.pkl    ← chunk 텍스트 + 메타데이터
```

> chunk_size 등 파라미터를 변경한 경우 `faiss_cache/` 폴더 삭제 후 재실행

```python
import shutil
shutil.rmtree("./faiss_cache")
```

---

## 출력 형식

`get_context()` 반환값 예시:

```
[출처 1: 논문.pdf, p.3]
RRF는 여러 ranked list를 결합하는 방식으로...

---

[출처 2: 논문.pdf, p.7]
RRF 공식: RRF(d) = Σ 1/(k + r(d))...
```
