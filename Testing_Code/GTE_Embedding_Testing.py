from modelscope import snapshot_download 
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from modelscope import snapshot_download
from modelscope.models import Model
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import numpy as np

# 设置numpy打印选项，这样可以打印出完整的数组而不是省略的版本
# np.set_printoptions(threshold=np.inf)

model_id = "damo/nlp_gte_sentence-embedding_chinese-base"
pipeline_se = pipeline(Tasks.sentence_embedding,
                       model=model_id,
                       sequence_length=512
                       ) # sequence_length 代表最大文本长度，默认值为128

inputs = {
        "source_sentence": [
            "不可以，早晨喝牛奶不科学",
            "吃了海鲜后是不能再喝牛奶的，因为牛奶中含得有维生素C，如果海鲜喝牛奶一起服用会对人体造成一定的伤害",
            "吃海鲜是不能同时喝牛奶吃水果，这个至少间隔6小时以上才可以。",
            "吃海鲜是不可以吃柠檬的因为其中的维生素C会和海鲜中的矿物质形成砷"
        ]
}
result = pipeline_se(input=inputs)
print (result)

# for embedding in result['text_embedding']:
#     print(embedding)






