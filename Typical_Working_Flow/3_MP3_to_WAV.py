#This one turn the mp3 to wav format which matches the requirements of STT work.

from pydub import AudioSegment
import os
from tqdm import tqdm

# 设定源文件夹和目标文件夹
source_folder = r"A:\苹花设计\咸鱼头冠\订单统计\Gadio_Index\download\mp3"
target_folder = r"A:\苹花设计\咸鱼头冠\订单统计\Gadio_Index\download\wav"

# 确保目标文件夹存在
if not os.path.exists(target_folder):
    os.makedirs(target_folder)

# 获取所有MP3文件
mp3_files = [f for f in os.listdir(source_folder) if f.endswith(".mp3")]

# 初始化进度条
progress_bar = tqdm(mp3_files, desc="Converting MP3 to WAV")

# 读取所有文件并进行转换
for file_name in progress_bar:
    mp3_path = os.path.join(source_folder, file_name)
    wav_file_name = file_name.replace(".mp3", ".wav")
    wav_path = os.path.join(target_folder, wav_file_name)
    
    # 检查目标文件夹中是否已存在wav文件
    if not os.path.exists(wav_path):
        # 读取mp3文件
        audio = AudioSegment.from_mp3(mp3_path)
        
        # 修改参数，设置采样率为16000Hz，单声道，16bit
        audio = audio.set_frame_rate(16000)  # 设置采样率
        audio = audio.set_channels(1)  # 设置单声道
        audio = audio.set_sample_width(2)  # 设置采样大小为2字节，即16bit
        
        # 转换为wav并保存
        audio.export(wav_path, format="wav", parameters=["-ac", "1", "-ar", "16000", "-sample_fmt", "s16p"])
    else:
        progress_bar.write(f"Skipping conversion for {wav_file_name} as it already exists.")

progress_bar.close()
print("Conversion complete!")
