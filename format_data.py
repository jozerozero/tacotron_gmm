word_list = []


_pad = '_'
_eos = '~'
_pinyinchars = 'abcdefghijklmnopqrstuvwxyz1234567890'
#_characters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz1234567890!\'(),-.:;? '
_characters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz1234567890!\'(),-.:;? #$%^'
_english2latin = 'ƖƗƙƚƛƜƝƞƟƠơƢƣƤƥƦƧƨƩƪƫƬƭƮƯưƱƲƳƴƵƶƷƸƹƺƻƼƽƾƿǂǄǅǆǇǈǉǊǋǌǍ'
# Export all symbols:
symbols = [_pad, _eos] + list(_characters) + list(_english2latin)
CMUPhonemes=['J', 'Q', 'X', 'AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'B',  'CH', 'D',  'DH', 'EH', 'ER', 'EY', 'F',  'G',  'HH', 'IH', 'IY', 'JH', 'K',  'L',  'M',  'N',  'NG', 'OW', 'OY', 'P',  'R',  'S',  'SH', 'T',  'TH', 'UH', 'UW', 'V',  'W',  'Y',  'Z',  'ZH', '0', '1', '2', '3']
puncs = '!\'(),-.:;? '
CMUPhonemes += list(puncs)
CMUPhonemes = [_pad, _eos] + CMUPhonemes
CMUPhonemes += '..'
CMUPhonemes += '"'
CMUPhonemes += ''

path = "training_data/train.txt"
en_stress = [0, 1, 2, 3]
cn_tone = [4, 5, 6, 7, 8]
tone = [0, 1, 2, 3, 4, 5, 6, 7, 8]


for line in open(path).readlines():
    line = line.strip().split("|")[5].split(" ")
    for word in line:
        word_list.append(word)

word_set = set(word_list)

print(word_set)
print(len(word_set))

replace_set = []

for word in word_set:
    if word not in CMUPhonemes:
        replace_set.append(word)


path = "training_data/train.txt"
path2 = open("training_data/train_2.txt", mode='w')

for line in open(path).readlines():
    line = line.strip().split("|")
    phone_list = line[5].split(" ")
    formate_phone_list=[]
    for phone in phone_list:
        if "#" not in phone:
            formate_phone_list.append(phone)
        for replace_word in replace_set:
            if replace_word in phone:
                continue
            pass

    formate_phone = " ".join(formate_phone_list)

    line[5] = formate_phone
    new_line = "|".join(line)
    print(new_line, file=path2)
