"""
RAG Baseline Pipeline
==========================
LangChain + FAISS 기반 RAG Retriever 베이스라인

전체 파이프라인에서의 위치:
    Memory State Generator → Router → [Retriever] → Answer Generator

외부 연결 시 사용:
    pipeline = RAGPipeline(llm=llm)
    pipeline.load_pdf("논문.pdf")
    context = pipeline.get_context(query)  # Answer Generator로 전달

설치:
    pip install -r requirements.txt
"""

# RAG Baseline Pipeline
# LangChain + FAISS 기반 RAG 베이스라인
# 클래스 정의 (PDFProcessor → EmbeddingManager → FAISSVectorStore → RAGChain → RAGPipeline)
# - 임베딩 모델: `BAAI/bge-m3` → `jhgan/ko-sroberta-multitask` (한국어 경량 모델)

# 0. 패키지 설치

# 로컬 환경: pip install -r requirements.txt 로 설치
# Colab 환경: 아래 주석 해제 후 실행
# %pip install langchain langchain-community langchain-huggingface langchain-openai \
#              faiss-cpu pymupdf sentence-transformers tqdm


# 1. 임포트

import os
import json
import hashlib
import pickle
from pathlib import Path
from typing import Optional

import fitz
import faiss
import numpy as np
from tqdm.auto import tqdm  # 임베딩 진행바용

# ver1 대비 변경: langchain_core, langchain_text_splitters 최신 방식으로 통일
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings


# 2. PDFProcessor
# PDF를 페이지 단위로 파싱하고 chunk로 분할

class PDFProcessor:
    """
    [역할] PDF 파싱 및 청킹 담당 클래스
    - PyMuPDF(fitz)로 PDF에서 텍스트 추출
    - RecursiveCharacterTextSplitter로 chunk 단위로 분할
    - 각 chunk에 파일명, 페이지 번호 메타데이터 첨부
    - 전체 파이프라인의 첫 번째 단계
    """

    def __init__(self, chunk_size: int = 1024, chunk_overlap: int = 100):
        """
        Args:
            chunk_size   : chunk 최대 문자 수 (기본 1024)
                           크게 설정하면 문맥 풍부, 작게 설정하면 검색 정밀도 향상
            chunk_overlap: 인접 chunk 간 겹치는 문자 수 (기본 100)
                           chunk 경계에서 문맥이 끊기는 것을 방지
                           chunk_size의 10% 수준 권장 (512 → 50, 1024 → 100)
        """
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            # 분리 우선순위: 문단 → 줄바꿈 → 문장 → 단어 → 글자
            separators=["\n\n", "\n", ".", " ", ""],
        )

    def load_and_split(self, pdf_path: str) -> list[Document]:
        """
        [기능] PDF 파일을 읽어 chunk 리스트로 반환

        Args:
            pdf_path: PDF 파일 경로

        Returns:
            list[Document]: LangChain Document 형식의 chunk 리스트
                            각 chunk에 page_content(텍스트)와
                            metadata(source, page)가 포함됨
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF 파일을 찾을 수 없습니다: {pdf_path}")

        docs = []
        with fitz.open(str(pdf_path)) as pdf:
            for page_num, page in enumerate(pdf, start=1):
                text = page.get_text("text")
                # 텍스트가 너무 짧은 페이지는 헤더/푸터/빈 페이지로 간주하고 스킵
                if len(text.strip()) < 30:
                    continue
                docs.append(Document(
                    page_content=text,
                    metadata={"source": pdf_path.name, "page": page_num}
                ))

        if not docs:
            raise ValueError("PDF에서 텍스트를 추출할 수 없습니다.")

        # 페이지 단위 Document → chunk 단위 Document로 분할
        chunks = self.splitter.split_documents(docs)
        print(f"[PDFProcessor] {pdf_path.name}: {len(docs)}페이지 → {len(chunks)}개 chunk")
        return chunks


# 3. EmbeddingManager
# - 기본 모델: `BAAI/bge-m3` → `jhgan/ko-sroberta-multitask`

class EmbeddingManager:
    """
    [역할] 임베딩 모델 로드 및 벡터 변환 담당 클래스
    - HuggingFace 임베딩 모델을 최초 1회만 로드하여 메모리 낭비 방지
    - 텍스트 → float32 벡터 변환 (FAISS 인덱스 입력값)
    - normalize_embeddings=True로 정규화 → 코사인 유사도 계산 가능
    """

    def __init__(self, model_name: str = "jhgan/ko-sroberta-multitask"):
        """
        Args:
            model_name: HuggingFace 임베딩 모델명
                        기본값:   BAAI/bge-m3 (다국어, 성능 좋지만 2.5GB로 무거움)
                        대안:    jhgan/ko-sroberta-multitask (한국어 특화, 경량)   => 일단 테스트용. RAG용으로는 BGE-M3 추천
        """
        print(f"[EmbeddingManager] 모델 로드 중: {model_name}")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cpu"},  # GPU 사용 시 "cuda"로 변경
            encode_kwargs={"normalize_embeddings": True},  # 코사인 유사도를 위해 정규화
        )
        # 임베딩 차원 수 확인 (FAISS IndexFlatIP 생성 시 필요)
        sample = self.embeddings.embed_query("test")
        self.dim = len(sample)
        print(f"[EmbeddingManager] 임베딩 차원: {self.dim}")

    def embed_documents(self, texts):
        """
        [기능] 텍스트 리스트를 임베딩 행렬로 변환 (배치 처리)

        Args:
            texts: 임베딩할 텍스트 리스트 (chunk 텍스트들)

        Returns:
            np.ndarray: shape (len(texts), dim)의 float32 벡터 행렬
        """
        # tqdm으로 진행바 표시 (chunk 수 많을 때 진행 상황 확인 가능)
        print(f"[EmbeddingManager] {len(texts)}개 텍스트 임베딩 중...")
        return np.array(
            self.embeddings.embed_documents(tqdm(texts, desc="Embedding documents")),
            dtype=np.float32
        )

    def embed_query(self, query):
        """
        [기능] 단일 쿼리를 임베딩 벡터로 변환 (검색 시 사용)

        Args:
            query: 검색 쿼리 문자열

        Returns:
            np.ndarray: shape (dim,)의 float32 벡터
        """
        return np.array(self.embeddings.embed_query(query), dtype=np.float32)


# 4. FAISSVectorStore
# FAISS 인덱스 빌드/저장/로드/검색 + md5 캐싱

class FAISSVectorStore:
    """
    [역할] FAISS 인덱스 생성/저장/로드/검색 담당 클래스
    - 벡터 검색 엔진 역할 (Retriever 핵심 컴포넌트)
    - FAISS는 벡터만 저장 → chunk 텍스트/메타데이터는 pkl로 별도 관리
    - md5 해시 기반 캐싱으로 동일 PDF 재임베딩 방지

    캐시 파일 구조:
        faiss_cache/
            {md5해시}.faiss  ← FAISS 인덱스 (벡터)
            {md5해시}.pkl   ← chunk 텍스트 + 메타데이터 매핑
    """

    def __init__(self, embedding_manager, cache_dir="./faiss_cache"):
        """
        Args:
            embedding_manager: EmbeddingManager 인스턴스
            cache_dir         : 인덱스 캐시 저장 디렉토리 (기본 ./faiss_cache)
        """
        self.em = embedding_manager
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.index = None            # FAISS 인덱스 객체
        self.chunks = []             # FAISS id → Document 매핑 리스트
        self.current_pdf_hash = None # 현재 로드된 PDF의 md5 해시

    @staticmethod
    def _get_file_hash(pdf_path):
        """
        [기능] PDF 파일 내용의 md5 해시 반환
        - 캐시 키로 사용 → 동일 파일이면 재임베딩 없이 캐시 로드
        """
        with open(pdf_path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()

    def _index_path(self, h):
        # FAISS 인덱스 파일 경로
        return self.cache_dir / f"{h}.faiss"

    def _chunks_path(self, h):
        # chunk 텍스트 매핑 파일 경로
        return self.cache_dir / f"{h}.pkl"

    def _is_cached(self, h):
        # 두 캐시 파일이 모두 존재하는지 확인
        return self._index_path(h).exists() and self._chunks_path(h).exists()

    def build_index(self, chunks):
        """
        [기능] chunk 리스트로 FAISS 인덱스 빌드
        - IndexFlatIP: 정규화된 벡터 기준 Inner Product = 코사인 유사도

        Args:
            chunks: PDFProcessor가 반환한 Document 리스트
        """
        texts = [doc.page_content for doc in chunks]
        # EmbeddingManager에서 tqdm 진행바를 표시하므로 여기선 로그 생략
        vectors = self.em.embed_documents(texts)
        # IndexFlatIP: 정규화된 벡터 → Inner Product = 코사인 유사도
        self.index = faiss.IndexFlatIP(self.em.dim)
        self.index.add(vectors)
        self.chunks = chunks
        print(f"[FAISSVectorStore] 인덱스 빌드 완료.")

    def save_index(self, h):
        """
        [기능] 인덱스와 chunk 매핑을 디스크에 저장
        - .faiss: 벡터 인덱스
        - .pkl  : chunk 텍스트 + 메타데이터
        """
        faiss.write_index(self.index, str(self._index_path(h)))
        with open(self._chunks_path(h), "wb") as f:
            pickle.dump(self.chunks, f)

    def load_index(self, h):
        """
        [기능] 디스크 캐시에서 인덱스와 chunk 매핑 로드
        - 재임베딩 없이 이전에 저장한 인덱스를 바로 사용 가능
        """
        self.index = faiss.read_index(str(self._index_path(h)))
        with open(self._chunks_path(h), "rb") as f:
            self.chunks = pickle.load(f)
        self.current_pdf_hash = h
        print(f"[FAISSVectorStore] 캐시 로드 완료.")

    def load_pdf(self, pdf_path, chunks):
        """
        [기능] PDF 로드 통합 메서드
        - 캐시 있으면 로드, 없으면 임베딩 후 저장
        - md5 해시로 동일 PDF 감지 → 재임베딩 방지

        Args:
            pdf_path: PDF 파일 경로 (해시 계산용)
            chunks  : PDFProcessor가 반환한 chunk 리스트
        """
        h = self._get_file_hash(pdf_path)
        if self._is_cached(h):
            print(f"[FAISSVectorStore] 캐시 발견. 재임베딩 없이 로드합니다.")
            self.load_index(h)
        else:
            print(f"[FAISSVectorStore] 캐시 없음. 새로 인덱스 빌드합니다.")
            self.build_index(chunks)
            self.save_index(h)
            self.current_pdf_hash = h

    def search(self, query, top_k=3):
        """
        [기능] 쿼리와 코사인 유사도가 높은 chunk top-k개 반환
        - 쿼리를 벡터로 변환 후 FAISS 인덱스에서 유사도 검색
        - 검색 결과에 유사도 score를 메타데이터로 첨부

        Args:
            query : 검색 쿼리 문자열
            top_k : 반환할 chunk 수 (기본 3)

        Returns:
            list[Document]: 유사도 높은 순으로 정렬된 chunk 리스트
        """
        if self.index is None:
            raise RuntimeError("인덱스가 로드되지 않았습니다.")
        query_vec = self.em.embed_query(query).reshape(1, -1)
        scores, indices = self.index.search(query_vec, top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:  # FAISS가 결과 없을 때 -1 반환
                continue
            doc = self.chunks[idx]
            doc.metadata["score"] = float(score)  # 유사도 점수 첨부 (디버깅용)
            results.append(doc)
        return results


# 5. RAGChain
# LangChain LCEL 방식으로 retrieval → 프롬프트 → LLM → 파싱 체인 조립

class RAGChain:
    """
    [역할] LangChain LCEL 기반 RAG 체인 조립 클래스
    - retrieval → 프롬프트 → LLM → 파싱 순서로 체인 구성
    - 내부 테스트용으로 단독 답변 생성 가능 (invoke)
    - 실제 파이프라인에서는 get_context()를 통해 context만 반환

    LangChain LCEL 체인 구조:
        query
          → _retrieve_and_format()  (FAISS 검색 + context 포맷)
          → PromptTemplate          (context + question 주입)
          → LLM                     (답변 생성)
          → StrOutputParser         (순수 문자열 반환)
    """

    # 기본 프롬프트 템플릿
    # context에 없는 내용은 답하지 않도록 명시 → 환각 방지
    DEFAULT_PROMPT = """당신은 주어진 문서를 바탕으로 질문에 답하는 어시스턴트입니다.

아래 [Context]는 문서에서 검색된 관련 내용입니다.
반드시 Context에 있는 내용만을 근거로 답변하세요.
Context에 관련 내용이 없으면 "문서에서 관련 내용을 찾을 수 없습니다."라고 답하세요.

[Context]
{context}

[Question]
{question}

[Answer]"""

    def __init__(self, vector_store, llm, top_k=3, prompt_template=None):
        """
        Args:
            vector_store   : FAISSVectorStore 인스턴스
            llm            : LangChain 호환 LLM (ChatOpenAI, Ollama 등)
            top_k          : retrieval 시 가져올 chunk 수 (기본 3)
            prompt_template: 커스텀 프롬프트 문자열 (None이면 DEFAULT_PROMPT 사용)
        """
        self.vector_store = vector_store
        self.top_k = top_k
        self.prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=prompt_template or self.DEFAULT_PROMPT,
        )
        # LCEL 체인 조립
        # RunnableLambda   : 커스텀 함수를 체인 단계로 감쌈
        # RunnablePassthrough: 입력을 그대로 다음 단계로 전달
        self.chain = (
            {
                "context" : RunnableLambda(self._retrieve_and_format),
                "question": RunnablePassthrough(),
            }
            | self.prompt
            | llm
            | StrOutputParser()
        )

    def _retrieve_and_format(self, query):
        """
        [기능] 쿼리로 chunk 검색 후 프롬프트에 넣을 context 문자열로 포맷
        - 출처(파일명, 페이지)를 함께 포함해 답변 근거 추적 가능

        Args:
            query: 검색 쿼리 문자열

        Returns:
            str: 출처 포함 context 문자열
        """
        docs = self.vector_store.search(query, top_k=self.top_k)
        if not docs:
            return "관련 문서를 찾을 수 없습니다."
        parts = []
        for i, doc in enumerate(docs, 1):
            parts.append(f"[출처 {i}: {doc.metadata.get('source','?')}, p.{doc.metadata.get('page','?')}]\n{doc.page_content}")
        return "\n\n---\n\n".join(parts)

    def invoke(self, query):
        """
        [기능] 질문을 받아 RAG 기반 답변 반환 (내부 테스트용)
        - retrieval + 답변 생성을 한 번에 처리
        - 실제 파이프라인 연결 시에는 get_context() 사용 권장

        Args:
            query: 사용자 질문

        Returns:
            str: LLM이 생성한 최종 답변
        """
        return self.chain.invoke(query)

    def get_retrieved_chunks(self, query):
        """
        [기능] 검색된 chunk 리스트만 반환 (답변 생성 없음)
        - RAGPipeline.get_context() 내부에서 호출
        - 디버깅 시 직접 호출해서 검색 품질 확인 가능

        Args:
            query: 검색 쿼리

        Returns:
            list[Document]: 유사도 높은 순 chunk 리스트
        """
        return self.vector_store.search(query, top_k=self.top_k)


# 6. RAGPipeline
# 위 4개 클래스를 조합한 메인 진입점

class RAGPipeline:
    """
    [역할] RAG 전체 파이프라인 메인 진입점
    - PDFProcessor / EmbeddingManager / FAISSVectorStore / RAGChain 조합
    - 외부에서는 load_pdf() / query() / get_context() 세 가지만 사용

    전체 파이프라인에서의 위치:
        Memory State Generator → Router → [RAGPipeline] → Answer Generator

    외부 연결 시 사용 메서드:
        get_context(question) → context 문자열 반환 → Answer Generator로 전달
    """

    def __init__(self, llm, embedding_model="jhgan/ko-sroberta-multitask",
                 chunk_size=1024, chunk_overlap=100, top_k=3,
                 cache_dir="./faiss_cache", prompt_template=None):
        """
        Args:
            llm            : LangChain 호환 LLM (ChatOpenAI, Ollama 등)
            embedding_model: HuggingFace 임베딩 모델명
            chunk_size     : PDF 청킹 크기 (기본 1024)
            chunk_overlap  : 청킹 overlap (기본 100, chunk_size의 10% 권장)
            top_k          : retrieval chunk 수 (기본 3)
            cache_dir      : FAISS 캐시 디렉토리 (기본 ./faiss_cache)
            prompt_template: 커스텀 프롬프트 (None이면 DEFAULT_PROMPT 사용)
        """
        # 각 컴포넌트 초기화 (의존성 주입 방식)
        self.pdf_processor    = PDFProcessor(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.embedding_manager = EmbeddingManager(model_name=embedding_model)
        self.vector_store     = FAISSVectorStore(self.embedding_manager, cache_dir=cache_dir)
        self.rag_chain        = RAGChain(self.vector_store, llm, top_k=top_k, prompt_template=prompt_template)
        self._loaded_pdf      = None  # 현재 로드된 PDF 경로 (중복 로드 방지용)

    def load_pdf(self, pdf_path):
        """
        [기능] PDF 파싱 + 벡터 인덱스 준비
        - 동일 PDF는 캐시에서 로드하여 재임베딩 방지
        - 같은 경로로 재호출 시 자동 스킵

        Args:
            pdf_path: PDF 파일 경로
        """
        if self._loaded_pdf == pdf_path:
            print(f"[RAGPipeline] '{pdf_path}' 이미 로드됨. 스킵.")
            return
        chunks = self.pdf_processor.load_and_split(pdf_path)
        self.vector_store.load_pdf(pdf_path, chunks)
        self._loaded_pdf = pdf_path
        print(f"[RAGPipeline] '{pdf_path}' 로드 완료.")

    def query(self, question):
        """
        [기능] 질문에 대한 RAG 기반 답변 반환 (내부 테스트용)
        - retrieval + 답변 생성을 한 번에 처리
        - 실제 파이프라인 연결 시에는 get_context() 사용 권장

        Args:
            question: 사용자 질문

        Returns:
            str: LLM이 생성한 최종 답변
        """
        if self._loaded_pdf is None:
            raise RuntimeError("PDF가 로드되지 않았습니다.")
        return self.rag_chain.invoke(question)

    def get_retrieved_chunks(self, question):
        """
        [기능] 검색된 chunk 리스트 반환 (디버깅용)
        - 답변 생성 없이 검색 결과만 확인할 때 사용
        - 각 chunk에 유사도 score 포함

        Args:
            question: 검색 쿼리

        Returns:
            list[Document]: 유사도 높은 순 chunk 리스트
        """
        return self.rag_chain.get_retrieved_chunks(question)

    def get_context(self, question: str) -> str:
        """
        [기능] Answer Generator에 넘겨줄 context 문자열 반환
        - 전체 파이프라인에서 Retriever의 최종 출력물
        - Router가 RETRIEVE_DOC 결정 후 이 메서드를 호출
        - 반환된 context를 Answer Generator가 받아 최종 답변 생성

        Args:
            question: 사용자 질문

        Returns:
            str: 출처 포함 context 문자열
                 형식: [출처 N: 파일명, p.페이지]
텍스트

---

...

        사용 예시:
            context = pipeline.get_context(query)     # Retriever 출력
            answer  = answer_generator(query, context) # Answer Generator 입력
        """
        chunks = self.rag_chain.get_retrieved_chunks(question)
        if not chunks:
            return "관련 문서를 찾을 수 없습니다."
        parts = []
        for i, doc in enumerate(chunks, 1):
            source = doc.metadata.get("source", "?")
            page   = doc.metadata.get("page", "?")
            parts.append(f"[출처 {i}: {source}, p.{page}]\n{doc.page_content}")
        return "\n\n---\n\n".join(parts)

