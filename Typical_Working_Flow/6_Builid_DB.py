# Build the DB with data

import chromadb
import pandas as pd
from chromadb import Documents, EmbeddingFunction, Embeddings
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from tqdm import tqdm
import numpy as np
import time

# 自定义的Embedding函数
class MyEmbeddingFunction(EmbeddingFunction):
    def __call__(self, input: Documents) -> Embeddings:
        embeddings = []
        for text in input:
            result = pipeline_se(input={"source_sentence": [text]})
            embeddings.append(result['text_embedding'][0])
        return embeddings

# 初始化ChromaDB客户端和模型
path = r"DB"
model_id = "damo/nlp_gte_sentence-embedding_chinese-base"
pipeline_se = pipeline(Tasks.sentence_embedding, model=model_id, sequence_length=512)
chroma_client = chromadb.PersistentClient(path)

# 创建本地持久化数据库
collection = chroma_client.create_collection(
    name="Gadio2023", 
    embedding_function=MyEmbeddingFunction(),
    metadata={"hnsw:space": "cosine"} 
)


# # Set up the collection
# collection = chroma_client.get_collection(
#     name="Gadio2023", 
#     embedding_function=MyEmbeddingFunction()
# )

# 灌入数据到数据库，带进度条
def process_csv(file_path, collection):
    df = pd.read_csv(file_path)

    with tqdm(total=df.shape[0], desc="Inserting Documents") as pbar:
        for index, row in df.iterrows():
            document = row['Text']
            updated_id = str(index)

            title = row['Title']
            url_number = row['URL Number']
            start_time = row['Start Time']

            metadatas = {"Title": title, "URL Number": url_number, "Start Time": start_time}

            embedding_str = row['Embedding']
            # 将字符串转换为浮点数列表
            embedding = [float(x) for x in embedding_str[1:-1].split()] if embedding_str else []

            collection.upsert(documents=[document], ids=[updated_id], embeddings=[embedding], metadatas=metadatas)
            pbar.update(1)

# 调用函数处理CSV文件
process_csv(r"G:\Pet_Projects\Gadio\embedding\final_data_with_embeddings.csv", collection)


results = collection.query(
    query_texts=["日本的鬼和中国的鬼含义不同"],
    n_results=1
)
print(results)
