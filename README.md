# rag-prototype

RAG 직업 추천 기능 로컬 구현

## Requirements

- Python 3.12에서 구현함

### Libraries

- `xlmtodict`
- [`langchain`, `langchain-community`, `langchain-huggingface`](https://github.com/langchain-ai/langchain)
- [`faiss-cpu`](https://github.com/facebookresearch/faiss)
- [`mlx-lm`](https://github.com/ml-explore/mlx-examples/blob/main/llms/README.md)
- 그 외 하위 의존 라이브러리들 (별도 설치 필요 없음)

## Files

- `save_index.py`: 구직 정보 데이터를 벡터 스토어에 입력 후 인덱스를 파일로 저장
  - 로컬에서 HuggingFace SBERT 임베딩 사용 ([dragonkue/BGE-m3-ko](https://huggingface.co/dragonkue/BGE-m3-ko))
  - FAISS 기반 벡터 스토어 구현 사용 (LangChain Community)
  - `faiss_index` 폴더에 인덱스 저장
  - URL 기반 UUIDv5를 id로 사용, 중복 확인에 사용
- `search_text.py`: 저장된 벡터 스토어 인덱스로 부터 유사도 검색이 잘 되는지 확인
  - 날짜, 지역 기반 메타데이터 필터링 구현
- `job_recommender.py`: 로컬 임베딩, 로컬 LLM으로 직업 추천 기능 구현
  - 로컬에서 MLX LM + LangChain Community 사용
  - Gemma 2 2B instruct 버전 4비트 양자화 모델 사용 ([mlx-community/gemma-2-2b-it-4bit](https://huggingface.co/mlx-community/gemma-2-2b-it-4bit))

### Untracked

- `secrets.yml`: 비밀 정보 파일
  - `API_KEY`: [구직정보 데이터](https://www.data.go.kr/data/15015153/openapi.do) API 인증키
