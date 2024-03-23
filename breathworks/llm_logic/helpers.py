from llama_index.core import StorageContext, VectorStoreIndex, load_index_from_storage
import os

def get_index(data, index_name,llm,service_context):
    index = None
    if not os.path.exists(index_name):
        print("building index", index_name)
        index = VectorStoreIndex.from_documents(data, show_progress=True,llm=llm,service_context=service_context)
        index.storage_context.persist(persist_dir=index_name)
    else:
        index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir=index_name)
        )

    return index
