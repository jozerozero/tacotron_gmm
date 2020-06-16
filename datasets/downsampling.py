import glob
import os

base_path = "wav"
target_path = "Wave"

if not os.path.exists(target_path):
    os.mkdir(target_path)

for wav_path in glob.glob(os.path.join(base_path, "*.wav")):
    target_file_name = os.path.basename(wav_path)
    command = "ffmpeg -i %s -ar 16000 %s" % (wav_path, os.path.join(target_path, target_file_name))
    os.system(command=command)