# CONVERSATIONAL_MODEL_RAG

멀티턴 대화 기반 RAG 파이프라인

---

## 전체 파이프라인 구조

```
사용자 입력
    ↓
Memory State Generator   ← 대화 맥락 정리
    ↓
Router                   ← 처리 방식 결정
    ↓
Retriever                ← PDF 문서 검색 (이 레포)
    ↓
Answer Generator         ← 최종 답변 생성
```

---

## 레포 구조

```
CONVERSATIONAL_MODEL_RAG/
    ├── retriever_baseline/     ← Retriever 모듈
    │   ├── rag_baseline.py     ← 메인 클래스 (import용)
    │   ├── README.md
    │   ├── requirements.txt
    │   └── .gitignore
    └── rag_evaluation/         ← 성능 평가
        ├── results/
        │   ├── ragas_chunk_512.csv
        │   └── ragas_chunk_1024.csv
        ├── ragas_chunk_512.ipynb
        ├── ragas_chunk_1024.ipynb
        └── README.md
```

---

## 담당 모듈

| 모듈 | 설명 | 폴더 |
|------|------|------|
| Retriever | PDF 문서 검색 + context 반환 | `retriever_baseline/` |
| Evaluation | RAGAS 기반 성능 평가 | `rag_evaluation/` |

---

## 빠른 시작

```bash
# 1. 패키지 설치
pip install -r retriever_baseline/requirements.txt

# 2. 사용
from retriever_baseline.rag_baseline import RAGPipeline

pipeline = RAGPipeline(llm=llm)
pipeline.load_pdf("논문.pdf")
context = pipeline.get_context("질문")  # Answer Generator로 전달
```

