import glob
import os
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import string

import numpy as np
from datasets import audio
# from wavenet_vocoder.util import is_mulaw, is_mulaw_quantize, mulaw, mulaw_quantize
import re
from pypinyin import pinyin, Style
from tacotron.utils.cn_convert import cn_convert
from tacotron.utils.cleaners import english_cleaners


def p(input):
    str = ""
    arr = pinyin(input, style=Style.TONE3)
    for i in arr:
        str += i[0] + " "
    return str

# for windows


def replace(str):
    return str.replace('\\', '/')

# punctuate


def segment(zhText, pinyinText):
    arr = pinyinText.split(" ")
    result = " "
    index = 0
    for j in zhText:
        if(j != ' ' and index < len(arr)):
            result += arr[index] + " "
            index += 1
        else:
            result += " "
    return result


def replace_punc(text):
    return text.translate(text.maketrans("，。？：；！“”、（）", ",..,..\"\",()"))


def remove_prosody(text):
    return re.sub(r'#[0-9]', '', text)

def contains_english(text):
    charset=string.ascii_letters
    return any((x in charset for x in text))

def folder_level_build_from_path(hparams, input_dirs, out_base_dir, n_jobs=12, tqdm=lambda x: x, mode='en'):
    """
    Preprocesses the speech dataset from a gven input path to given output directories

    Args:
            - hparams: hyper parameters
            - input_dir: input directory that contains the files to prerocess
            - n_jobs: Optional, number of worker process to parallelize across
            - tqdm: Optional, provides a nice progress bar

    Returns:
            - A list of tuple describing the train examples. this should be written to train.txt
    """

    # We use ProcessPoolExecutor to parallelize across processes, this is just for
    # optimization purposes and it can be omited
    executor = ProcessPoolExecutor(max_workers=n_jobs)
    futures_tuples=[]
    #mode='en' # 'cn' or 'en'
    #speakers = ['p262', 'p272', 'p229', 'p232', 'p292', 'p293', 'p360', 'p361', 'p248', 'p251']
    #input_dirs=[cur_dir for cur_dir in input_dirs if cur_dir[-4:] in speakers]
    for input_dir in input_dirs:
        futures = []
        prosody_labeling = os.path.join(
            input_dir, 'ProsodyLabeling', '000001-010000.txt')
        if not os.path.exists(prosody_labeling):
            print("%s not found, this skip it" % prosody_labeling)
            continue
        out_dir = os.path.join(out_base_dir, os.path.basename(input_dir))
        mel_dir = os.path.join(out_dir, 'mels')
        wav_dir = os.path.join(out_dir, 'audio')
        linear_dir = os.path.join(out_dir, 'linear')
        os.makedirs(mel_dir, exist_ok=True)
        os.makedirs(wav_dir, exist_ok=True)
        os.makedirs(linear_dir, exist_ok=True)
        # index = 0
        with open(prosody_labeling, encoding='utf-8') as fpl:
            for line in fpl:
                # index += 1
                # if index == 2:
                #     break
                line=line.strip()
                if line=='':
                    continue
                fields = line.split('|')
                wav_num = fields[0]
                if mode=='en':
                    text = fields[1].lower()
                    text = english_cleaners(text)
                elif mode=='cn':
                    #if contains_english(fields[1]):
                    #    continue
                    #TODO: remove the #1234
                    #text = p(cn_convert(remove_prosody(fields[1])))
                    text = p(cn_convert(fields[1]))
                basename = wav_num+'.wav'
                wav_path = os.path.join(input_dir, 'Wave', basename)
                futures.append(executor.submit(partial(
                    _process_utterance, mel_dir, linear_dir, wav_dir, basename, wav_path, text, hparams)))
        futures_tuples.append(([future.result() for future in tqdm(futures) if future.result() is not None], out_dir))
    return futures_tuples

def build_from_path(hparams, input_dirs, mel_dir, linear_dir, wav_dir, n_jobs=12, tqdm=lambda x: x):
    """
    Preprocesses the speech dataset from a gven input path to given output directories

    Args:
            - hparams: hyper parameters
            - input_dir: input directory that contains the files to prerocess
            - mel_dir: output directory of the preprocessed speech mel-spectrogram dataset
            - linear_dir: output directory of the preprocessed speech linear-spectrogram dataset
            - wav_dir: output directory of the preprocessed speech audio dataset
            - n_jobs: Optional, number of worker process to parallelize across
            - tqdm: Optional, provides a nice progress bar

    Returns:
            - A list of tuple describing the train examples. this should be written to train.txt
    """

    # We use ProcessPoolExecutor to parallelize across processes, this is just for
    # optimization purposes and it can be omited
    executor = ProcessPoolExecutor(max_workers=n_jobs)
    futures = []
    index = 1
    mode='en' # 'cn' or 'en'
    print("=================CNT====================")
    if mode == 'en':  # TODO: add condition
        for input_dir in input_dirs:
            print("=================cnt====================")
            prosody_labeling = os.path.join(
                input_dir, 'ProsodyLabeling', '000001-010000.txt')
            cnt = 0
            with open(prosody_labeling, encoding='utf-8') as fpl:
                for line in fpl:
                    fields = line.split('\t')
                    wav_num = fields[0]
                    text = fields[1].lower()
                    basename = wav_num+'.wav'
                    wav_path = os.path.join(input_dir, 'Wave', basename)
                    futures.append(executor.submit(partial(
                        _process_utterance, mel_dir, linear_dir, wav_dir, basename, wav_path, text, hparams)))
        return [future.result() for future in tqdm(futures) if future.result() is not None]

    elif mode == 'cn':  # TODO: add condition
        for input_dir in input_dirs:
            prosody_labeling = os.path.join(
                input_dir, 'ProsodyLabeling', '000001-010000.txt')
            cnt = 0
            with open(prosody_labeling, encoding='utf-8') as fpl:
                for line in fpl:
                    cnt += 1
                    if cnt % 2 == 0:
                        continue
                    fields = line.split('\t')
                    wav_num = fields[0]
                    text = p(cn_convert(fields[1]))
                    basename = wav_num+'.wav'
                    wav_path = os.path.join(input_dir, 'Wave', basename)
                    futures.append(executor.submit(partial(
                        _process_utterance, mel_dir, linear_dir, wav_dir, basename, wav_path, text, hparams)))
        return [future.result() for future in tqdm(futures) if future.result() is not None]

    if True:  # TODO: add condition
        for input_dir in input_dirs:
            prosody_labeling = os.path.join(
                input_dir, 'ProsodyLabeling', '000001-010000.txt')
            cnt = 0
            with open(prosody_labeling, encoding='gb18030') as fpl:
                for line in fpl:
                    cnt += 1
                    if cnt % 2 == 0:
                        continue
                    fields = line.split('	')
                    wav_num = fields[0]
                    text = p(remove_prosody(replace_punc(fields[1])))
                    basename = wav_num+'.wav'
                    wav_path = os.path.join(input_dir, 'Wave', basename)
                    futures.append(executor.submit(partial(
                        _process_utterance, mel_dir, linear_dir, wav_dir, basename, wav_path, text, hparams)))

        return [future.result() for future in tqdm(futures) if future.result() is not None]

    for input_dir in input_dirs:
        trn_files = glob.glob(os.path.join(input_dir, "data", 'A*.trn'))
        for trn in trn_files:
            with open(trn, encoding='utf-8') as f:
                basename = trn[:-4]
                text = None
                if basename.endswith('.wav'):
                    # THCHS30
                    zhText = f.readline()
                    pinyinText = f.readline()
                    text = segment(zhText, pinyinText)
                    wav_file = basename
                else:
                    wav_file = basename + '.wav'
                wav_path = wav_file
                basename = basename.split('/')[-1]
                text = text if text != None else f.readline().strip()

                mel_dir = replace(mel_dir)
                linear_dir = replace(linear_dir)
                wav_dir = replace(wav_dir)
                wav_path = replace(wav_path)
                basename = replace(basename)

                futures.append(executor.submit(partial(
                    _process_utterance, mel_dir, linear_dir, wav_dir, basename, wav_path, text, hparams)))
                index += 1
    return [future.result() for future in tqdm(futures) if future.result() is not None]


def _process_utterance(mel_dir, linear_dir, wav_dir, index, wav_path, text, hparams):
    """
    Preprocesses a single utterance wav/text pair

    this writes the mel scale spectogram to disk and return a tuple to write
    to the train.txt file

    Args:
            - mel_dir: the directory to write the mel spectograms into
            - linear_dir: the directory to write the linear spectrograms into
            - wav_dir: the directory to write the preprocessed wav into
            - index: the numeric index to use in the spectogram filename
            - wav_path: path to the audio file containing the speech input
            - text: text spoken in the input audio file
            - hparams: hyper parameters

    Returns:
            - A tuple: (audio_filename, mel_filename, linear_filename, time_steps, mel_frames, linear_frames, text)
    """
    eliminated=0
    try:
        # Load the audio as numpy array
        wav = audio.load_wav(wav_path, sr=hparams.sample_rate)
    except FileNotFoundError:  # catch missing wav exception
        print('file {} present in csv metadata is not present in wav folder. skipping!'.format(
            wav_path))
        return None

    try:
        # rescale wav
        if hparams.rescale:
            wav = wav / np.abs(wav).max() * hparams.rescaling_max

        # M-AILABS extra silence specific
        if hparams.trim_silence:
            new_wav = audio.trim_silence(wav, hparams)
            eliminated+=(len(wav)-len(new_wav))/hparams.sample_rate
    except Exception as e:
        print('%s: %s' % (str(e), wav_path))
        return None


    # Mu-law quantize
#     if is_mulaw_quantize(hparams.input_type):
#         # [0, quantize_channels)
#         out = mulaw_quantize(wav, hparams.quantize_channels)

#         # Trim silences
#         start, end = audio.start_and_end_indices(
#             out, hparams.silence_threshold)
#         wav = wav[start: end]
#         out = out[start: end]

#         constant_values = mulaw_quantize(0, hparams.quantize_channels)
#         out_dtype = np.int16

#     elif is_mulaw(hparams.input_type):
#         #[-1, 1]
#         out = mulaw(wav, hparams.quantize_channels)
#         constant_values = mulaw(0., hparams.quantize_channels)
#         out_dtype = np.float32

#     else:
#         #[-1, 1]
#         out = wav
#         constant_values = 0.
#         out_dtype = np.float32

    out = wav
    constant_values = 0.
    out_dtype = np.float32

    # Compute the mel scale spectrogram from the wav
    mel_spectrogram = audio.melspectrogram(wav, hparams).astype(np.float32)
    mel_frames = mel_spectrogram.shape[1]

    if mel_frames > hparams.max_mel_frames and hparams.clip_mels_length:
        eliminated+=len(wav)/hparams.sample_rate
        print('mel_frame: ' + str(mel_frames) + ', max: '+str(hparams.max_mel_frames) + 'clip mels length: '+str(hparams.clip_mels_length))
        return [None, eliminated]

    if hparams.predict_linear:
        # Compute the linear scale spectrogram from the wav
        linear_spectrogram = audio.linearspectrogram(
            wav, hparams).astype(np.float32)
        linear_frames = linear_spectrogram.shape[1]

        # sanity check
        assert linear_frames == mel_frames

    if hparams.use_lws:
        # Ensure time resolution adjustement between audio and mel-spectrogram
        fft_size = hparams.n_fft if hparams.win_size is None else hparams.win_size
        l, r = audio.pad_lr(wav, fft_size, audio.get_hop_size(hparams))

        # Zero pad audio signal
        out = np.pad(out, (l, r), mode='constant',
                     constant_values=constant_values)
    else:
        # Ensure time resolution adjustement between audio and mel-spectrogram
        pad = audio.librosa_pad_lr(
            wav, hparams.n_fft, audio.get_hop_size(hparams))

        # Reflect pad audio signal (Just like it's done in Librosa to avoid frame inconsistency)
        out = np.pad(out, pad, mode='reflect')

    assert len(out) >= mel_frames * audio.get_hop_size(hparams)

    # time resolution adjustement
    # ensure length of raw audio is multiple of hop size so that we can use
    # transposed convolution to upsample
    out = out[:mel_frames * audio.get_hop_size(hparams)]
    assert len(out) % audio.get_hop_size(hparams) == 0
    time_steps = len(out)

    # Pre filename
    audio_filename = 'audio-{}.npy'.format(index)
    mel_filename = 'mel-{}.npy'.format(index)
    linear_filename = 'linear-{}.npy'.format(index)
    audio_filename_full = replace(os.path.join(wav_dir, audio_filename))
    mel_filename_full = replace(os.path.join(mel_dir, mel_filename))
    linear_filename_full = replace(os.path.join(linear_dir, linear_filename))

    # Make dir
    os.makedirs(os.path.dirname(audio_filename_full), exist_ok=True)
    os.makedirs(os.path.dirname(mel_filename_full), exist_ok=True)
    os.makedirs(os.path.dirname(linear_filename_full), exist_ok=True)

    # Write the spectrogram and audio to disk
    np.save(audio_filename_full, out.astype(out_dtype), allow_pickle=False)
    np.save(mel_filename_full, mel_spectrogram.T, allow_pickle=False) # (L,dim)
    if hparams.predict_linear:
        np.save(linear_filename_full, linear_spectrogram.T, allow_pickle=False)

    # Return a tuple describing this training example
    return [audio_filename, mel_filename, linear_filename, time_steps, mel_frames, text, eliminated]
