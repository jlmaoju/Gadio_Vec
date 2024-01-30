import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import datetime


# ==================以下写入问题（一系列问题仅供测试，反复定义这个变量的话只有最后一个会起作用）==================
query_texts = ["从小洞看过去，第二天发现对面地上有两个圆形的痕迹，像芭蕾舞鞋"]
query_texts = ["在宿舍里打牌，然后牌消失了"]
query_texts = ["到火车站接我，然后从包里拿出一瓶啤酒"]
query_texts = ["半夜偷偷从父母房间经过偷游戏机"]
query_texts = ["一个小丑骑在独轮车上大笑，在公路上吓了我们一跳"]

# ==================需要拿到几个查询结果（多拿结果不会更慢，但是显示太多了也看不过来）==================
n_results=2


# ==================以下是定义数据库和Embedding方法的部分==================

# 初始化ChromaDB客户端和模型
path = r"DB" # 指定ChromaDB的数据库文件夹
chroma_client = chromadb.PersistentClient(path) # 初始化ChromaDB的client
model_id = "damo/nlp_gte_sentence-embedding_chinese-base" # 用了一个效果比较好的中文专用embedding模型
pipeline_se = pipeline(Tasks.sentence_embedding, model=model_id, sequence_length=512) # Embedding的pipline


# 自定义的Embedding函数（以下不多展开，具体可以看ChromaDB的文档：https://docs.trychroma.com/embeddings）
class MyEmbeddingFunction(EmbeddingFunction):
    def __call__(self, input: Documents) -> Embeddings:
        embeddings = []
        for text in input:
            result = pipeline_se(input={"source_sentence": [text]})
            embeddings.append(result['text_embedding'][0])
        return embeddings

# 从数据库文件夹里get到集合（Collection）
collection = chroma_client.get_collection(
    name="Gadio2023", 
    embedding_function=MyEmbeddingFunction()
)


# ==================以下是执行查询的部分==================
results = collection.query(
    query_texts = query_texts,
    n_results=n_results
)

# ==================以下是负责显示查询结果的部分==================
base_url = "https://www.gcores.com/radios/"

# 打印查询内容和结果
print("==================查询开始==================")
print(f"查询内容：{query_texts}")
print("==================查询结果==================")

for metadata, document in zip(results['metadatas'][0], results['documents'][0]):
    title = metadata['Title']
    
    # 将 StartTime 从毫秒转换为小时:分钟:秒格式
    start_time_milliseconds = metadata['Start Time']
    start_time_seconds = start_time_milliseconds / 1000
    start_time_formatted = str(datetime.timedelta(seconds=int(start_time_seconds)))
    
    # 拼接 URL
    url_number = metadata['URL Number']
    full_url = base_url + str(url_number)
    
    print(f"节目标题: {title}\n开始时间: {start_time_formatted}\nURL: {full_url}\n内容摘要: {document[:200]}...\n")
print("==================查询结束==================")
