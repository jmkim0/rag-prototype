from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import List, Annotated

import docx2txt
from fastapi import FastAPI, Request, File, UploadFile, Form
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders.parsers import PyPDFParser
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models.mlx import ChatMLX
from langchain_community.llms.mlx_pipeline import MLXPipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.documents.base import Blob
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.schema import Document


components = {}
chains = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    embeddings = HuggingFaceEmbeddings(
        model_name="dragonkue/bge-m3-ko",
        model_kwargs={"model_kwargs": {"torch_dtype": "float16"}},
    )
    index_path = Path("faiss_index")

    print(f"Loads a saved index: {str(index_path)}.")
    components["vector_store"] = FAISS.load_local(
        str(index_path), embeddings, allow_dangerous_deserialization=True
    )

    pipeline = MLXPipeline.from_model_id(
        "mlx-community/gemma-2-2b-it-4bit", pipeline_kwargs={"max_tokens": 1000}
    )
    components["llm"] = ChatMLX(llm=pipeline)

    rag_template = (
        "You are an assistant for making a personalized job opening list. "
        "Use following pieces of the retrieved context to make the list. "
        "The retrieved context is a set of job openings. "
        "Summarize each job opening into a sublist of up to three items. "
        "If there is no context, apologize about that you cannot find job openings. "
        "Answer in Korean."
        "\n\n"
        "{context}"
        "\n\n"
        "이력: {input}"
    )

    chains["rag"] = create_stuff_documents_chain(
        components["llm"], PromptTemplate.from_template(rag_template)
    )

    summarize_template = (
        "The following context is the user's profile to find a job. "
        "Write a useful summary of the profile in up to three sentences. "
        "Don't add any comments to the summary. "
        "As the context is in Korean, write the summary in Korean."
        "\n\n"
        "{context}"
    )

    chains["summarize"] = create_stuff_documents_chain(
        components["llm"], PromptTemplate.from_template(summarize_template)
    )

    yield

    components.clear()
    chains.clear()


def create_filter_fn(date: str, area: str | None = None):
    def filter_fn(metadata):
        location = metadata.get("location")
        from_date = metadata.get("from")
        to_date = metadata.get("to")

        if (
            bool(area)
            & bool(location)
            & (isinstance(area, str) and area not in location)
        ):
            return False

        if (
            bool(date)
            & bool(from_date)
            & bool(to_date)
            & (
                not datetime.strptime(from_date, "%Y%m%d")
                < datetime.strptime(date, "%Y%m%d")
                < datetime.strptime(to_date, "%Y%m%d")
            )
        ):
            return False

        return True

    return filter_fn


async def summarize_and_generate(documents: List[Document]):
    # 요약 실행
    summary = await chains["summarize"].ainvoke({"context": documents})

    retriever = components["vector_store"].as_retriever(
        search_kwargs={
            "k": 5,
            "filter": create_filter_fn(datetime.now().strftime("%Y%m%d"), None),
            "fetch_k": 50,
        }
    )
    rag_chain = create_retrieval_chain(retriever, chains["rag"])

    result = await rag_chain.ainvoke({"input": summary})

    return summary, result


app = FastAPI(lifespan=lifespan)

templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def root():
    return FileResponse("static/index.html")


@app.get("/upload", response_class=HTMLResponse)
async def upload():
    return FileResponse("static/upload.html")


@app.get("/create", response_class=HTMLResponse)
async def create():
    return FileResponse("static/create.html")


@app.post("/process_upload")
async def process_upload(request: Request, file: UploadFile = File(...)):
    # 파일 형식에 따라 처리
    if file.filename.endswith(".pdf"):
        documents = PyPDFParser().parse(Blob.from_data(await file.read()))
    elif file.filename.endswith(".docx"):
        documents = [Document(docx2txt.process(file.file))]
    else:
        return JSONResponse(
            content={"error": "지원하지 않는 파일 형식입니다."}, status_code=400
        )

    summary, result = await summarize_and_generate(documents)

    # 결과를 프론트로 전달
    return templates.TemplateResponse(
        "result.html",
        {"request": request, "summary": summary, "result": result["answer"]},
    )


@app.post("/process_create")
async def submit_resume(
    request: Request,
    name: Annotated[str | None, Form()] = None,
    age: Annotated[int | str | None, Form()] = None,
    contact: Annotated[str | None, Form()] = None,
    experience: Annotated[str | None, Form()] = None,
    education: Annotated[str | None, Form()] = None,
    certifications: Annotated[str | None, Form()] = None,
    skills: Annotated[str | None, Form()] = None,
    desired_job: Annotated[str | None, Form()] = None,
    desired_location: Annotated[str | None, Form()] = None,
):
    # 이력서 텍스트 생성
    resume = f"이름: {name}\n" if name else ""
    if age:
        resume += f"나이: {age}\n"
    if contact:
        resume += f"연락처: {contact}\n"
    if education:
        resume += f"주요 경력: {experience}\n"
    if education:
        resume += f"학력: {education}\n"
    if certifications:
        resume += f"자격증: {certifications}\n"
    if skills:
        resume += f"기술: {skills}\n"
    if desired_job:
        resume += f"희망 직무: {desired_job}\n"
    if desired_location:
        resume += f"희망 근무지: {desired_location}\n"

    # `Document` 객체로 변환
    document = Document(page_content=resume)

    summary, result = await summarize_and_generate([document])

    return templates.TemplateResponse(
        "result.html",
        {"request": request, "summary": summary, "result": result["answer"]},
    )
