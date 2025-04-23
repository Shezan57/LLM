from langchain_community.document_loaders import CSVLoader

loader = CSVLoader("../data/movies.csv")

docs = loader.lazy_load()

# print(len(docs))
count = 0
for doc in docs:
    count += 1
    print(doc)
    if count > 5:
        break
