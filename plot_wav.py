import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import librosa.display
import librosa.core
import sys

wavename=sys.argv[1]
pngname=sys.argv[2]

loaded_wav=librosa.core.load(wavename, sr=22050)[0]
plt.figure(figsize=(14, 5))
librosa.display.waveplot(loaded_wav, sr=22050)
plt.savefig(pngname)
