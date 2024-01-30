#Embedding with GTE model

import csv
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from tqdm import tqdm

# 设置模型
model_id = "damo/nlp_gte_sentence-embedding_chinese-base"
pipeline_se = pipeline(Tasks.sentence_embedding, model=model_id, sequence_length=512)

# 读取原始CSV文件
with open(r"embedding.csv", 'r', encoding='utf-8') as file:
    csv_reader = list(csv.reader(file))

# 初始化进度条
pbar = tqdm(total=len(csv_reader) - 1, desc="Processing Texts")

# 计算embeddings并将其添加到每行的末尾
new_rows = []
for row in csv_reader:
    if csv_reader.index(row) == 0:  # 如果是标题行，添加新列标题
        row.append('Embedding')
    else:
        text = row[2]  # 获取文本列
        inputs = {"source_sentence": [text]}
        result = pipeline_se(input=inputs)
        embedding = result['text_embedding'][0]
        row.append(embedding)
        pbar.update(1)  # 更新进度条
    new_rows.append(row)

pbar.close()  # 完成后关闭进度条

# 将新数据写入到一个新的CSV文件中
with open(r'final_data_with_embeddings.csv', 'w', newline='', encoding='utf-8') as file:
    csv_writer = csv.writer(file)
    csv_writer.writerows(new_rows)
