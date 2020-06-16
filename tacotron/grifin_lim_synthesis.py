import numpy as np
from datasets.audio import *
import os
from hparams import hparams
from glob import glob
import os


def griffin_synthesis(mel_path):
	for mel_file_path in glob(os.path.join(mel_path, "*.npy")):
		mel_spectro = np.load(mel_file_path)
		mel_spectro_shape = mel_spectro.shape
		mel_file_name = os.path.basename(mel_file_path)
		wav = inv_mel_spectrogram(mel_spectro.T, hparams)
		print(os.path.join(mel_path, mel_file_name.replace("npy", "wav")))
		save_wav(wav, os.path.join(mel_path, mel_file_name.replace("npy", "wav")), sr=hparams.sample_rate)
	pass

