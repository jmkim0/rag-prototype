import uuid
from pathlib import Path

import faiss
import httpx
import xmltodict
import yaml
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings


def main():
    embeddings = HuggingFaceEmbeddings(model_name="dragonkue/bge-m3-ko")
    index_path = Path("faiss_index")

    if index_path.exists():
        print(f"Loads a saved index: {str(index_path)}.")
        vector_store = FAISS.load_local(
            str(index_path), embeddings, allow_dangerous_deserialization=True
        )
    else:
        print("Creates a new index.")
        vector_store = FAISS(
            embedding_function=embeddings,
            index=faiss.IndexFlatL2(len(embeddings.embed_query("hello world"))),
            docstore=InMemoryDocstore(),
            index_to_docstore_id={},
        )

    secrets_path = Path("secrets.yaml")

    if not secrets_path.exists():
        print(f"No {str(secrets_path)}.")
        exit()

    print(f"Loads {str(secrets_path)}.")
    with open(secrets_path) as file:
        secrets = yaml.load(file, Loader=yaml.CLoader)

    list_url = "http://apis.data.go.kr/B552474/SenuriService/getJobList"
    list_params = {
        "serviceKey": secrets["API_KEY"],
        "pageNo": 1,
        "numOfRows": 1000,
        # 'search': '',
        # 'emplymShp': '',
        # 'workPlcNm': ''
    }
    info_url = "http://apis.data.go.kr/B552474/SenuriService/getJobInfo"
    info_params = {"serviceKey": secrets["API_KEY"]}
    docs = []
    ids = []

    list_response = httpx.get(list_url, params=list_params, timeout=10)

    for item in xmltodict.parse(list_response.text)["response"]["body"]["items"][
        "item"
    ]:
        info_params["id"] = item["jobId"]
        info_url_with_params = httpx.URL(info_url, params=info_params)
        document_id = str(uuid.uuid5(uuid.NAMESPACE_URL, str(info_url_with_params)))

        # 존재하는 문서 id면 넘어감
        # TODO: 존재하는 id의 문서 추가시 해당 문서 최신화하는 방식으로 개선
        if document_id in ids or isinstance(
            vector_store.docstore.search(document_id), Document
        ):
            continue

        info_response = httpx.get(info_url_with_params, timeout=10)
        info_dict = xmltodict.parse(info_response.text)["response"]["body"]["items"][
            "item"
        ]
        metadata = {
            "url": info_response.url,
            "from": info_dict.get("frAcptDd"),
            "to": info_dict.get("toAcptDd"),
            "location": info_dict.get("plDetAddr"),
        }
        content = (
            f"채용제목: {info_dict.get('wantedTitle', '')}\n"
            f"사업장명: {info_dict.get('plbizNm', '')}\n"
            f"상세내용: {info_dict.get('detCnts', '')}\n"
            f"기타사항: {info_dict.get('etcItm', '')}\n"
        )
        docs.append(Document(page_content=content, metadata=metadata))
        ids.append(document_id)

    if docs:
        print(f"Adds {len(docs)} documents.")
        vector_store.add_documents(documents=docs, ids=ids)

        print(f"Saves the index: {str(index_path)}.")
        vector_store.save_local(str(index_path))
    else:
        print("No new documents to add.")


if __name__ == "__main__":
    main()
