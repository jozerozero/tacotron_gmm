import argparse
import os
import re
from multiprocessing import cpu_count
from pypinyin.style._utils import get_initials, get_finals

from datasets.process_text import process_en, get_pinyin2cmu_dict
from datasets import preprocessor
from hparams import hparams
from tqdm import tqdm
from g2p_en import G2p
from glob import glob
g2p=G2p()
outf=open('Preprocess.log', 'w', encoding='utf-8')
# python preprocess.py --single-speaker --dataset ../datasets/tts_data/BZNSYP/ --output training_biaobei_wavernn_24k --language cn
# python preprocess.py --single-speaker --dataset ../datasets/tts_data/ljspeech/ --output training_ljspeech_wavernn_24k
#python preprocess.py --single-speaker --dataset ../datasets/tts_data/ljspeech/ --hparams_json tools/wavernn24khparams.json --output training_ljspeech_wavernn_24k
#python preprocess.py --language cn --dataset ../datasets/cn_dataset/sd235/ --hparams_json logs-cn-sd235-bz60-xvector-mfcc/hparams.json --output training_cn_sd235
#python preprocess.py --single-speaker --language cn --dataset ../datasets/tts_data/cn_60_1/ --hparams_json logs-cn-clearclearsd235-bz60-xvector-mfcc --output training_cn_60_1
#python preprocess.py --dataset ../datasets/tts_data/libritts_clean/ --hparams_json tools/waveglow_hparams.json --output training_libritts_clean_waveglow (for waveglow)
#python preprocess.py --dataset ../datasets/tts_data/libritts_clean/ --hparams_json tools/wavernn_hparams.json --output training_libritts_clean_wavernn (for wavernn)
#python preprocess.py --training_dataset ../datasets/tts_training/training_tts_clean/ --hparams_json tools/waveglow_hparams.json --cmu_only (for producing phoneme only)
def preprocess(args, input_folders, out_dir, hparams):
    mel_dir = os.path.join(out_dir, 'mels')
    wav_dir = os.path.join(out_dir, 'audio')
    linear_dir = os.path.join(out_dir, 'linear')
    os.makedirs(mel_dir, exist_ok=True)
    os.makedirs(wav_dir, exist_ok=True)
    os.makedirs(linear_dir, exist_ok=True)
    metadata = preprocessor.build_from_path(
        hparams, input_folders, mel_dir, linear_dir, wav_dir, args.n_jobs, tqdm=tqdm)
    write_metadata(metadata, out_dir)

# For multiple speakers
def folder_level_preprocess(args, input_folders, out_dir, hparams):
    if args.cmu_only:
        convert_cmu(args.training_dataset)
    else:
        metadata_list = preprocessor.folder_level_build_from_path(
            hparams, input_folders, out_dir, args.n_jobs, tqdm=tqdm, mode=args.language.split('_')[0])
        for metadata, out_dir in metadata_list:
            write_metadata(metadata, out_dir, mode=args.language.split('_')[0])

<<<<<<< HEAD
def preprocess(args, input_folders, out_dir, hparams):
	mel_dir = os.path.join(out_dir, 'mels')
	wav_dir = os.path.join(out_dir, 'audio')
	linear_dir = os.path.join(out_dir, 'linear')
	os.makedirs(mel_dir, exist_ok=True)
	os.makedirs(wav_dir, exist_ok=True)
	os.makedirs(linear_dir, exist_ok=True)
	metadata = preprocessor.build_from_path(hparams, input_folders, mel_dir, linear_dir, wav_dir, args.n_jobs, tqdm=tqdm)
	write_metadata(metadata, out_dir)


def write_metadata(metadata, out_dir):
	with open(os.path.join(out_dir, 'train.txt'), 'w', encoding='utf-8') as f:
		for m in metadata:
			f.write('|'.join([str(x) for x in m]) + '\n')
	mel_frames = sum([int(m[4]) for m in metadata])
	timesteps = sum([int(m[3]) for m in metadata])
	sr = hparams.sample_rate
	hours = timesteps / sr / 3600
	print('Write {} utterances, {} mel frames, {} audio timesteps, ({:.2f} hours)'.format(
		len(metadata), mel_frames, timesteps, hours))
	print('Max input length (text chars): {}'.format(max(len(m[5]) for m in metadata)))
	print('Max mel frames length: {}'.format(max(int(m[4]) for m in metadata)))
	print('Max audio timesteps length: {}'.format(max(m[3] for m in metadata)))
=======
def convert_cmu(training_dataset):
    training_dirs = glob('%s/' % training_dataset)
    for training_dir in training_dirs:
        with open(os.path.join(training_dir, 'train.txt'), 'r', encoding='utf-8') as f:
            with open(os.path.join(training_dir, 'cmu_long.txt'), 'w', encoding='utf-8') as outf:
                for line in f.readlines():
                    line=line.strip()
                    if line=='':
                        continue
                    m=line.split('|')
                    m=m[:-1]
                    if m[0] is not None:
                        spkid=m[0].split('-',1)[1].split('.')[0]
                        m[-1]=' '.join(g2p(m[-1].strip()))
                        m.append("%s.npy" % spkid)
                        outf.write('|'.join([str(x) for x in m]) + '\n')

def part2(text, pinyin2cmu_dict):
    phone_list = list()
    tone_list = list()
    #     print(text)
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    #     exit()
    new_phone_list = list()
    for pinyin in text.split(" "):
        if "#" not in pinyin:
            tone = re.findall(r"\d+\.?\d*", pinyin)
        else:
            tone = []
        if len(tone) == 0:
            tone = 7
        else:
            tone = int(tone[0]) + 2
        #         print(pinyin2cmu_dict.keys())
        #         exit()
        # pinyin = pinyin.replace(str(tone - 2), "")

        head = get_initials(pinyin, False).upper()
        tail = get_finals(pinyin, False).upper()

        if "#" in pinyin:
            new_phone_list.append(pinyin)
            continue

        if head not in pinyin2cmu_dict.keys() and tail not in pinyin2cmu_dict:
            new_phone_list.append(pinyin)
            continue
        if head != "":
            new_phone_list.append(pinyin2cmu_dict[head])
        if tail != "":
            tone = re.findall(r"\d+\.?\d*", tail)
            if len(tone)==0:
                new_phone_list.append(pinyin2cmu_dict[tail])
            else:
                tail = tail.replace(str(tone[0]), "")
                new_phone_list.append(pinyin2cmu_dict[tail]+str(tone[0]))
                pass
        new_phone_list.append(" ")
        # if get_initials(pinyin, False).upper() in pinyin2cmu_dict.keys():
        #     new_phone_list.append(pinyin2cmu_dict[get_initials(pinyin, False).upper()])
        # elif get_finals(pinyin, False).upper() in pinyin2cmu_dict.keys():
        #     new_phone_list.append(pinyin2cmu_dict[get_finals(pinyin, False).upper()])
        # else:
        #     new_phone_list.append(pinyin)

    return new_phone_list
        # print(pinyin, len(pinyin))
        # # print('test', get_initials(pinyin, False).upper())
        # # print('tste', get_finals(pinyin, False).upper())
        #
        # if get_initials(pinyin, False).upper() not in pinyin2cmu_dict.keys() and get_finals(pinyin,
        #                                                                                     False).upper() not in pinyin2cmu_dict.keys():
        #     print("not in:", pinyin)
        #     print(get_initials(pinyin, False).upper())
        #     print(get_finals(pinyin, False).upper())
        #     phone_list.append(pinyin)
        #     phone_list.append(" ")
        #     tone_list.append(str(tone))
        #     tone_list.append(str(tone))
        #
        # for pin_part in (get_initials(pinyin, False), get_finals(pinyin, False)):
        #     #             print(pinyin, pin_part)
        #     if pin_part.upper() in pinyin2cmu_dict.keys():
        #         phone_list.append(pinyin2cmu_dict[pin_part.upper()])
        #         tone_list.append(str(tone))
        # phone_list.append(" ")
        # tone_list.append(str(7))

    # new_phone_list = list()
    # for phone, tone in zip(phone_list, tone_list):
    #     if tone != 7:
    #         new_phone = phone + str(tone)
    #     else:
    #         new_phone = phone
    #     new_phone_list.append(new_phone)
    #
    # return new_phone_list


def part(text, pinyin2cmu_dict):
    phone_list = list()
    tone_list = list()
#     print(text)
    text = re.sub(r'[^\x00-\x7F]+', '', text)
#     exit()
    for pinyin in text.split(" "):
        if len(pinyin) == 0 or pinyin == " " or "#" in pinyin:
            continue
        tone = re.findall(r"\d+\.?\d*", pinyin)
        if len(tone) == 0:
            tone = "5"
            pass
        tone = int(tone[0]) + 3

        pinyin = pinyin.replace(str(tone-3), "")
        print(pinyin, len(pinyin))
        print('test', get_initials(pinyin, False).upper(), get_initials(pinyin, False).upper() not in pinyin2cmu_dict.keys())
        print('tste', get_finals(pinyin, False).upper(), get_finals(pinyin, False).upper() not in pinyin2cmu_dict.keys())
        print((get_initials(pinyin, False).upper() not in pinyin2cmu_dict.keys()) and (get_finals(pinyin, False).upper() not in pinyin2cmu_dict.keys()))
        print("===================")
        # if (get_initials(pinyin, False).upper() not in pinyin2cmu_dict.keys()) and (get_finals(pinyin, False).upper() not in pinyin2cmu_dict.keys()):
        #     print("not in:", pinyin)
        #     print(get_initials(pinyin, False).upper())
        #     print(get_finals(pinyin, False).upper())
        #     phone_list.append(pinyin)
        #     phone_list.append(" ")
        #     tone_list.append(str(tone))
        #     tone_list.append(str(tone))
        #     continue
        
        for pin_part in (get_initials(pinyin, False), get_finals(pinyin, False)):
            print("pin_part", pin_part)
            if pin_part.upper() in pinyin2cmu_dict.keys():
                phone_list.append(pinyin2cmu_dict[pin_part.upper()])
                for _ in pinyin2cmu_dict[pin_part.upper()].split(" "):
                    tone_list.append(str(tone))
                print("cmu", pinyin2cmu_dict[pin_part.upper()])
        phone_list.append("$")
        tone_list.append(str(8))
    # print(len(phone_list[:-5]))
    print(phone_list)
    return phone_list, tone_list


def write_metadata(metadata, out_dir, mode):
    pinyin2cmu_dict = get_pinyin2cmu_dict()
    
    with open(os.path.join(out_dir, 'train.txt'), 'w', encoding='utf-8') as f:
        for m in metadata:
            m=m[:-1]
            if m[0] is not None:
                spkid=m[0].split('-',1)[1].split('.')[0]
                m[-1]=m[-1].strip()
                m.append("%s.npy" % spkid)
                f.write('|'.join([str(x) for x in m]) + '\n')
    with open(os.path.join(out_dir, 'cmu_long.txt'), 'w', encoding='utf-8') as f:
        for m in metadata:
            m=m[:-1]
            if m[0] is not None:
                spkid=m[0].split('-', 1)[1].split('.')[0]
                if mode=='en':
                    m[-1]='$'.join(g2p(m[-1].strip()))
                    # m[-1]=' '.join(g2p(m[-1].strip()))
                    phone_list, stress_list = process_en(m[-1])
                    m[-1] = " ".join(phone_list)
                    m.append(" ".join(stress_list))
                elif mode=='cn':
                    phone_list, tone_list = part(m[-1], pinyin2cmu_dict)
                    # phone_list = part2(m[-1], pinyin2cmu_dict)
                    m[-1] = " ".join(phone_list)
                    m.append(" ".join(tone_list))
                
                m.append("%s.npy" % spkid)
                f.write('|'.join([str(x) for x in m]) + '\n')
    mel_frames = sum([int(m[4]) for m in metadata if m[0] is not None])
    timesteps = sum([int(m[3]) for m in metadata if m[0] is not None])
    total_eliminated_sec = sum([m[-1] for m in metadata])
    sr = hparams.sample_rate
    hours = timesteps / sr / 3600
    print('Write {} utterances, {} mel frames, {} audio timesteps, ({:.2f} hours)'.format(
        len(metadata), mel_frames, timesteps, hours))
    if len(metadata)!=0:
        print('Max input length (text chars): {}'.format(
            max(len(m[5]) for m in metadata if m[0] is not None)))
        print('Max mel frames length: {}'.format(max(int(m[4]) for m in metadata if m[0] is not None)))
        print('Max audio timesteps length: {}'.format(max(m[3] for m in metadata if m[0] is not None)))
    outf.write('%s\t%f\t%f\n' % (out_dir, hours, total_eliminated_sec/3600))
>>>>>>> f33090dba9ba4bc52db8367abdc48841d13c48f8

def norm_data(args):

    merge_books = (args.merge_books == 'True')

<<<<<<< HEAD
	print('Selecting data folders..')
	supported_datasets = ['LJSpeech-1.0', 'LJSpeech-1.1', 'M-AILABS', 'THCHS-30', 'BZNSYP', 'Boya_Female']
	#if args.dataset not in supported_datasets:
	#	raise ValueError('dataset value entered {} does not belong to supported datasets: {}'.format(
	#		args.dataset, supported_datasets))
=======
    print('Selecting data folders..')
    #supported_datasets = ['LJSpeech-1.0',
    #                      'LJSpeech-1.1', 'M-AILABS', 'THCHS-30', 'BZNSYP']
    # if args.dataset not in supported_datasets:
    #	raise ValueError('dataset value entered {} does not belong to supported datasets: {}'.format(
    #		args.dataset, supported_datasets))
>>>>>>> f33090dba9ba4bc52db8367abdc48841d13c48f8

    if args.dataset.startswith('LJSpeech'):
        return [os.path.join(args.base_dir, args.dataset)]

    if args.dataset.startswith('THCHS-30'):
        return [os.path.join(args.base_dir, 'data_thchs30')]

    if args.dataset.startswith('BZNSYP'):
        return [os.path.join(args.base_dir, 'BZNSYP')]

<<<<<<< HEAD
	if args.dataset.startswith('Boya_Female'):
		return [os.path.join(args.base_dir, 'Boya_Female')] 

	if args.dataset == 'M-AILABS':
		supported_languages = ['en_US', 'en_UK', 'fr_FR', 'it_IT', 'de_DE', 'es_ES', 'ru_RU',
			'uk_UK', 'pl_PL', 'nl_NL', 'pt_PT', 'fi_FI', 'se_SE', 'tr_TR', 'ar_SA']
		if args.language not in supported_languages:
			raise ValueError('Please enter a supported language to use from M-AILABS dataset! \n{}'.format(
				supported_languages))
=======
    if args.dataset == 'M-AILABS':
        supported_languages = ['cn', 'en_US', 'en_UK', 'fr_FR', 'it_IT', 'de_DE', 'es_ES', 'ru_RU',
                               'uk_UK', 'pl_PL', 'nl_NL', 'pt_PT', 'fi_FI', 'se_SE', 'tr_TR', 'ar_SA']
        if args.language not in supported_languages:
            raise ValueError('Please enter a supported language to use from M-AILABS dataset! \n{}'.format(
                supported_languages))
>>>>>>> f33090dba9ba4bc52db8367abdc48841d13c48f8

        supported_voices = ['female', 'male', 'mix']
        if args.voice not in supported_voices:
            raise ValueError('Please enter a supported voice option to use from M-AILABS dataset! \n{}'.format(
                supported_voices))

        path = os.path.join(args.base_dir, args.language,
                            'by_book', args.voice)
        supported_readers = [e for e in os.listdir(
            path) if os.path.isdir(os.path.join(path, e))]
        if args.reader not in supported_readers:
            raise ValueError('Please enter a valid reader for your language and voice settings! \n{}'.format(
                supported_readers))

        path = os.path.join(path, args.reader)
        supported_books = [e for e in os.listdir(
            path) if os.path.isdir(os.path.join(path, e))]
        if merge_books:
            return [os.path.join(path, book) for book in supported_books]

        else:
            print('supported_books '+ str(supported_books))
            if args.book not in supported_books:
                raise ValueError('Please enter a valid book for your reader settings! \n{}'.format(
                    supported_books))

            return [os.path.join(path, args.book)]

    return [os.path.join(args.base_dir, args.dataset)]


def run_preprocess(args, hparams):
    input_folders = norm_data(args)
    output_folder = os.path.join(args.base_dir, args.output)

    #preprocess(args, input_folders, output_folder, hparams)
    if not args.single_speaker:
        input_folders = glob('%s/*' % args.dataset)
    else:
        input_folders = [args.dataset]
    folder_level_preprocess(args, input_folders, output_folder, hparams)


def main():
<<<<<<< HEAD
	print('initializing preprocessing..')
	parser = argparse.ArgumentParser()
	parser.add_argument('--base_dir', default='')
	parser.add_argument('--hparams', default='',
		help='Hyperparameter overrides as a comma-separated list of name=value pairs')
	parser.add_argument('--dataset', default='BZNSYP')
	parser.add_argument('--language', default='en_US')
	parser.add_argument('--voice', default='female')
	parser.add_argument('--reader', default='mary_ann')
	parser.add_argument('--merge_books', default='False')
	parser.add_argument('--book', default='northandsouth')
	parser.add_argument('--output', default='training_data')
	parser.add_argument('--n_jobs', type=int, default=cpu_count())
	args = parser.parse_args()
	modified_hp = hparams.parse(args.hparams)
=======
    print('initializing preprocessing..')
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', default='')
    parser.add_argument('--hparams', default='',
                        help='Hyperparameter overrides as a comma-separated list of name=value pairs')
    parser.add_argument('--hparams_json', default='',
                        help='Hyperparameter in json format')
    parser.add_argument('--dataset', default='BZNSYP')
    parser.add_argument('--language', default='en_US')
    parser.add_argument('--voice', default='female')
    parser.add_argument('--reader', default='mary_ann')
    parser.add_argument('--merge_books', default='False')
    parser.add_argument('--book', default='northandsouth')
    parser.add_argument('--output', default='training_data')
    parser.add_argument('--n_jobs', type=int, default=cpu_count())
    parser.add_argument('--max_mel_frames', type=int, default=15000)
    parser.add_argument('--cmu_only', action='store_true', help='whether just convert train.txt to cmu_spkid.txt')
    parser.add_argument('--single-speaker', action='store_true', help='whether the provded dataset is a single speaker or a parent folder of multiple speakers')
    parser.add_argument('--training_dataset', default='training')
    args = parser.parse_args()
>>>>>>> f33090dba9ba4bc52db8367abdc48841d13c48f8

    modified_hp = hparams.parse(args.hparams)
    modified_hp.max_mel_frames = args.max_mel_frames
    assert args.merge_books in ('False', 'True')
    if args.hparams_json:
        import json
        with open(args.hparams_json) as hp_json_file:
            hp_json=json.dumps(json.load(hp_json_file)['hparams'])
            modified_hp=modified_hp.parse_json(hp_json)

    run_preprocess(args, modified_hp)


if __name__ == '__main__':
    main()
    outf.close()
