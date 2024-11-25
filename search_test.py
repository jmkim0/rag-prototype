from pathlib import Path
from datetime import datetime

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


def main():
    embeddings = HuggingFaceEmbeddings(
        model_name="dragonkue/bge-m3-ko",
        model_kwargs={"model_kwargs": {"torch_dtype": "float16"}},
    )
    index_path = Path("faiss_index")

    print(f"Loads a saved index: {str(index_path)}.")
    vector_store = FAISS.load_local(
        str(index_path), embeddings, allow_dangerous_deserialization=True
    )

    def create_filter_fn(current_date, area):
        def filter_fn(metadata):
            location = metadata.get("location")
            from_date = metadata.get("from")
            to_date = metadata.get("to")

            if bool(area) & bool(location) & (area not in location):
                return False

            if (
                bool(current_date)
                & bool(from_date)
                & bool(to_date)
                & (
                    not datetime.strptime(from_date, "%Y%m%d")
                    < datetime.strptime(current_date, "%Y%m%d")
                    < datetime.strptime(to_date, "%Y%m%d")
                )
            ):
                return False

            return True

        return filter_fn

    for doc in vector_store.similarity_search(
        "미화원", filter=create_filter_fn("20241021", "서울")
    ):
        print(doc.metadata)
        print(doc.page_content)


if __name__ == "__main__":
    main()
