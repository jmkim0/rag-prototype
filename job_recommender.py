import asyncio
from datetime import datetime
from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models.mlx import ChatMLX
from langchain_community.llms.mlx_pipeline import MLXPipeline
from langchain_core.prompts import PromptTemplate
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain


async def main():
    embeddings = HuggingFaceEmbeddings(
        model_name="dragonkue/bge-m3-ko",
        model_kwargs={"model_kwargs": {"torch_dtype": "float16"}},
    )
    index_path = Path("faiss_index")

    print(f"Loads a saved index: {str(index_path)}.")
    vector_store = FAISS.load_local(
        str(index_path), embeddings, allow_dangerous_deserialization=True
    )

    pipeline = MLXPipeline.from_model_id(
        "mlx-community/gemma-2-2b-it-4bit", pipeline_kwargs={"max_tokens": 1000}
    )
    llm = ChatMLX(llm=pipeline)

    template = (
        "You are an assistant for making a personalized job opening list. "
        "Use following pieces of the retrieved context to make the list. "
        "The retrieved context is a set of job openings."
        "Summarize each job opening into a sublist of up to three items. "
        "If there is no context, apologize about that you cannot find job openings."
        "Answer in Korean."
        "\n\n"
        "{context}"
        "\n\n"
        "이력: {input}"
    )

    prompt = PromptTemplate.from_template(template)
    question_answer_chain = create_stuff_documents_chain(llm, prompt)

    def create_filter_fn(date, area):
        def filter_fn(metadata):
            location = metadata.get("location")
            from_date = metadata.get("from")
            to_date = metadata.get("to")

            if bool(area) & bool(location) & (area not in location):
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

    while True:
        area = input("지역: ")

        if area == "exit":
            break

        while True:
            profile = input("이력: ")

            if not profile:
                break

            retriever = vector_store.as_retriever(
                search_kwargs={
                    "k": 5,
                    "filter": create_filter_fn(datetime.now().strftime("%Y%m%d"), area),
                    "fetch_k": 50,
                }
            )
            rag_chain = create_retrieval_chain(retriever, question_answer_chain)
            response = await rag_chain.ainvoke({"input": profile})

            print(response)
            print(response["answer"])


if __name__ == "__main__":
    asyncio.run(main())
