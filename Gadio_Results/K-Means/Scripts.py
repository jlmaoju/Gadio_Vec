import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# 读取CSV文件
df = pd.read_csv(r"average_embeddings.csv")

# 提取特征，确保每个嵌入向量都是长度为768的NumPy数组
X = df["Average Embedding"].apply(lambda x: np.array([float(num) for num in x.split(',')]).reshape(1, -1)).values

X = np.vstack(X)  # 将列表中的数组堆叠成一个二维数组
# # 使用肘部法则确定最佳的簇数
# wcss = []
# for i in range(1, 11):  # 预设的聚类数范围从1到10
#     kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
#     kmeans.fit(X)
#     wcss.append(kmeans.inertia_)

# # 绘制肘部法则图
# plt.plot(range(1, 11), wcss)
# plt.title('Elbow Method')
# plt.xlabel('Number of clusters')
# plt.ylabel('WCSS')
# plt.show()

# 根据肘部法则图，选择一个合适的簇数
# 这里我们假设选择3个簇，但你应该根据肘部法则图来决定
n_clusters = 50 
典型的样本数 = 999
# 使用K-Means算法进行聚类
kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42)
y_kmeans = kmeans.fit_predict(X)

# 将聚类结果添加到原始DataFrame
df['Cluster'] = y_kmeans

# 计算轮廓系数
silhouette_avg = silhouette_score(X, y_kmeans)
print(f"轮廓系数: {silhouette_avg}")

# 对于每个聚类，找到最典型的前十个样本

典型的节目标题 = {}
for i in range(n_clusters):
    # 获取当前聚类的索引
    cluster_indices = np.where(y_kmeans == i)[0]
    # 计算每个样本到聚类中心的距离
    distances = np.linalg.norm(X[cluster_indices] - kmeans.cluster_centers_[i], axis=1)
    # 获取距离最小的前十个样本的索引
    top_indices = distances.argsort()[:典型的样本数]
    # 获取这些样本的节目标题
    typical_titles = df.iloc[cluster_indices[top_indices]]['Title'].tolist()
    典型的节目标题[i] = typical_titles

# 输出每个聚类的典型节目标题
for cluster, titles in 典型的节目标题.items():
    print(f"聚类 {cluster} 的典型节目标题:")
    for title in titles:
        print(f" - {title}")


with open('cluster_titles.txt', 'w', encoding='utf-8') as file:
    # 遍历聚类及其典型节目标题
    for cluster, titles in 典型的节目标题.items():
        # 写入聚类信息
        file.write(f"聚类 {cluster} 的典型节目标题:\n")
        # 遍历并写入每个标题
        for title in titles:
            file.write(f" - {title}\n")
        # 在每个聚类之间添加一个空行
        file.write("\n")
