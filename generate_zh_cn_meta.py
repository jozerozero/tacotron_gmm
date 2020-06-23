import os
import glob

base_path = "wavs"


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass

    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass

    return False


def get_slab(interval_file_path):
    sentence = ""
    count = 0
    for line in open(interval_file_path):
        count += 1
        if count < 12:
            continue
        line = line.strip()
        if not is_number(line):
            sentence += line.replace("\"", "", 2)
            sentence += " "

    print(sentence)
    return sentence


def generate_meta():
    meta_file = open("meta.txt", "w")
    for audio_file_name in glob.glob(os.path.join(base_path, "*.wav")):
        interval_file_path = os.path.join(base_path, os.path.basename(audio_file_name).replace("wav", "interval"))
        assert os.path.exists(interval_file_path)
        sent = get_slab(interval_file_path=interval_file_path)
        meta_file.write(os.path.basename(audio_file_name).replace(".wav", "")+"\t"+sent+"\n")
        pass


if __name__ == '__main__':
    generate_meta()
    pass
