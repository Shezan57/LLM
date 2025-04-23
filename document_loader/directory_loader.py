from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader


path = r"C:\Users\arafa\OneDrive\Desktop\Books"

loader = DirectoryLoader(path=path, glob="*.pdf", loader_cls=PyPDFLoader)
# docs = loader.load()

docs_lazy = loader.lazy_load()

for doc in docs_lazy:
    # print(doc.page_content)
    print(doc.metadata)

# print(len(docs))
# print(docs[0].page_content)
# print(docs[25].metadata)