# RAG Evaluation

chunk_size 파라미터별 RAG 성능 평가 결과 모음

---

## 평가 지표 (RAGAS)

| 지표 | 설명 | 좋은 점수 |
|------|------|----------|
| `faithfulness` | 답변이 context에 근거하는가 (환각 감지) | 높을수록 좋음 |
| `answer_relevancy` | 답변이 질문과 얼마나 관련 있는가 | 높을수록 좋음 |
| `context_precision` | 검색된 chunk가 얼마나 정확한가 | 높을수록 좋음 |
| `context_recall` | 정답에 필요한 내용이 chunk에 있는가 | 높을수록 좋음 |

---

## 파일 구조

```
rag_evaluation/
    ├── results/
    │   ├── ragas_chunk_512.csv   ← chunk_size=512 실험 결과
    │   └── ragas_chunk_1024.csv  ← chunk_size=1024 실험 결과
    ├── ragas_chunk_512.ipynb     ← chunk_size=512 평가 노트북
    └── ragas_chunk_1024.ipynb    ← chunk_size=1024 평가 노트북
```

---

## 실험 설정

| 설정 | chunk_512 | chunk_1024 |
|------|-----------|------------|
| `chunk_size` | 512 | 1024 |
| `chunk_overlap` | 50 | 100 |
| `embedding_model` | jhgan/ko-sroberta-multitask | jhgan/ko-sroberta-multitask |
| `top_k` | 3 | 3 |
| `llm` | gpt-5-mini | gpt-5-mini |

---

## 실험 결과 요약

결과는 `results/` 폴더의 CSV 파일 참고

---

## 실행 방법

```
1. retriever_baseline/requirements.txt 설치
   pip install -r retriever_baseline/requirements.txt

2. RAGAS 추가 설치
   pip install ragas datasets pandas

3. 노트북 실행
   ragas_chunk_*.ipynb 열어서 순서대로 실행

4. OpenAI API 키 설정 필요
   os.environ["OPENAI_API_KEY"] = "sk-..."
```

---

## 주의사항

- `results/` 폴더의 CSV는 특정 PDF 기준 결과이므로 참고용으로만 사용
- 다른 PDF로 실험 시 결과가 달라질 수 있음
- RAGAS 평가는 내부적으로 OpenAI API를 호출하므로 비용 발생
