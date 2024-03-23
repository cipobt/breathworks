import os
from llama_index.core import ServiceContext, StorageContext, VectorStoreIndex, load_index_from_storage


def get_index(data, index_name,llm):
    index = None
    if not os.path.exists(index_name):
        print("building index", index_name)
        index = VectorStoreIndex.from_documents(data, show_progress=True,llm=llm)
        index.storage_context.persist(persist_dir=index_name)
    else:
        index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir=index_name)
        )

    return index
