#Now STT (Speach to txt)

from funasr import AutoModel
import os
import time
import json
from tqdm import tqdm

# 设置转录的文本长度阈值
text_length_threshold = 128

# 指定音频文件夹路径和输出文件夹
audio_dir = r"wav"
output_dir = r"txt"

# 确保输出文件夹存在
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 初始化ASR模型
model = AutoModel(model="paraformer-zh", model_revision="v2.0.4",
                  vad_model="fsmn-vad", vad_model_revision="v2.0.4",
                  punc_model="ct-punc-c", punc_model_revision="v2.0.4",
                  # spk_model="cam++", spk_model_revision="v2.0.2",
                  )

# 获取所有WAV文件
audio_files = [f for f in os.listdir(audio_dir) if f.endswith('.wav')]

# 初始化进度条
progress_bar = tqdm(audio_files, desc="Processing audio files")

# 初始化错误日志文件路径
error_log_path = os.path.join(output_dir, 'errors.txt')

# 逐个处理每个文件
for file_name in progress_bar:
    try:
        print(f"Processing file: {file_name}")
        start_time = time.time()  # 开始计时
        audio_file_path = os.path.join(audio_dir, file_name)
        output_text_path = os.path.join(output_dir, os.path.splitext(file_name)[0] + '.txt')
        output_json_path = os.path.join(output_dir, os.path.splitext(file_name)[0] + '.json')
        
        # 进行语音识别
        # res = p.generate(audio_file_path, batch_size_token=5000)
        res = model.generate(input=audio_file_path, 
                     batch_size_s=5000)


        # 文本处理和保存
        combined_sentences = []
        current_text = ''
        current_start_time = 0
        
        # Assume each result in 'res' is a dict with a 'sentences' key
        for item in res:
            if 'sentences' in item:  # Ensure there is a 'sentences' key
                for sentence in item['sentences']:
                    if not current_text:  # If current text is empty, it's the beginning of a new segment
                        current_start_time = sentence['start']
                    current_text += sentence['text']
                    if len(current_text) >= text_length_threshold:  # Once the segment reaches the character threshold
                        # Save the segment along with the start time
                        combined_sentences.append({
                            'text': current_text,
                            'start_time': current_start_time
                        })
                        # Reset for the next segment
                        current_text = ''
        
        # Process any remaining text
        if current_text:
            combined_sentences.append({
                'text': current_text,
                'start_time': current_start_time
            })
        
        # Save results to .txt file
        with open(output_text_path, "w", encoding='utf-8') as file:
            for sent in combined_sentences:
                file.write(f"Text: {sent['text']}, Start Time: {sent['start_time']}\n")
        
        # Save results to .json file
        with open(output_json_path, "w", encoding='utf-8') as json_file:
            json.dump(res, json_file, ensure_ascii=False, indent=4)
        
        end_time = time.time()  # 结束计时
        inference_time = end_time - start_time  # 计算推理时间
        progress_bar.set_postfix_str(f"Inference time: {inference_time:.2f}s")
    
    except IndexError as e:
        # 将错误信息写入日志
        with open(error_log_path, "a", encoding='utf-8') as error_log:
            error_message = f"Error processing {file_name}: {e}\n"
            error_log.write(error_message)
        continue  # 继续处理下一个文件

print("All files have been processed.")
