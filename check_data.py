path = "training_biaobei_wavernn_24k/cmu_long.txt"

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
        print(len(word))
        print("test:"+word+"|")
        pass