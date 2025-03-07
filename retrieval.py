import pandas as pd
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import bm25s


df = pd.read_csv("/root/source/predict_hscode/before_update/code/data_train.csv",dtype= {"hs_code":"object"})
df = df.dropna()

documents = []

for index, row in df.iterrows():
    text = row['description']
    hs_code = row['hs_code']
    documents.append(Document(page_content=text, metadata={'hs_code': hs_code}))

splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,  
    chunk_overlap=0
)

split_documents = []
for doc in documents:
    chunks = splitter.split_text(doc.page_content)
    for chunk in chunks:
        split_documents.append(Document(page_content=chunk, metadata=doc.metadata))

docs = []
for doc in split_documents:
    metadata = doc.metadata
    metadata_str = str(metadata).strip('{}')
    page = doc.page_content
    docs.append([metadata_str + " " + page])

cleaned_list = [item.replace('"', '').replace("'",'')  for items in docs for item in items]

# Create the BM25 model and index the corpus
retriever = bm25s.BM25(corpus=cleaned_list)
retriever.index(bm25s.tokenize(cleaned_list))

list_path = ["/root/source/predict_hscode/full_power/data/data_test/data_94tr_records/part-00749-3c1ae1df-2c80-4abc-9949-c8584678fd4c-c000.csv" ,
             "/root/source/predict_hscode/full_power/data/data_test/data_94tr_records/part-00750-3c1ae1df-2c80-4abc-9949-c8584678fd4c-c000.csv", 
             "/root/source/predict_hscode/full_power/data/data_test/data_94tr_records/part-00751-3c1ae1df-2c80-4abc-9949-c8584678fd4c-c000.csv",
             "/root/source/predict_hscode/full_power/data/data_test/data_94tr_records/part-00752-3c1ae1df-2c80-4abc-9949-c8584678fd4c-c000.csv",
             "/root/source/predict_hscode/full_power/data/data_test/data_94tr_records/part-00753-3c1ae1df-2c80-4abc-9949-c8584678fd4c-c000.csv", 
             "/root/source/predict_hscode/full_power/data/data_test/data_94tr_records/part-00754-3c1ae1df-2c80-4abc-9949-c8584678fd4c-c000.csv", 
             "/root/source/predict_hscode/full_power/data/data_test/data_94tr_records/part-00755-3c1ae1df-2c80-4abc-9949-c8584678fd4c-c000.csv", 
             "/root/source/predict_hscode/full_power/data/data_test/data_94tr_records/part-00756-3c1ae1df-2c80-4abc-9949-c8584678fd4c-c000.csv", 
             "/root/source/predict_hscode/full_power/data/data_test/data_94tr_records/part-00757-3c1ae1df-2c80-4abc-9949-c8584678fd4c-c000.csv", 
             "/root/source/predict_hscode/full_power/data/data_test/data_94tr_records/part-00758-3c1ae1df-2c80-4abc-9949-c8584678fd4c-c000.csv"]
df2 = pd.DataFrame()
for path in list_path:
    df2 = df2._append(pd.read_csv(path,on_bad_lines = "skip"))


list_case = ['WITH','N/M','M/N','0','++++','++++++','+++++++++++']
for case in list_case:
    df2 = df2[df2['CARGO DESCRIPTION']!=case]
len_des = [len(des) for des in df2['CARGO DESCRIPTION']]
df2['len'] = len_des
df2 = df2[df2['len']>=3]

full_description = [des.lower() for des in df2['CARGO DESCRIPTION']]
docs,_ = retriever.retrieve(bm25s.tokenize(full_description),k=5)

#convert to document
documents = []
for doc in docs:
    result = []
    for d in doc:
        result.append(Document(d))
    documents.append(result)

df2['document'] = documents
df2.to_csv("/root/source/predict_hscode/full_power/data/data_test/data_94tr_records/retrieval-749-758.csv")