import re

from . import cleaners
from .symbols import symbols, _pinyinchars, _english2latin, CMUPhonemes

# Mappings from symbol to numeric ID and vice versa:
_symbol_to_id = {s: i for i, s in enumerate(symbols)}
_id_to_symbol = {i: s for i, s in enumerate(symbols)}
_phoneme_to_id = {s: i for i, s in enumerate(CMUPhonemes)}

# Regular expression matching text enclosed in curly braces:
_curly_re = re.compile(r'(.*?)\{(.+?)\}(.*)')

def split_by_single_space(text):
    res=[]
    cur_str=''
    space_cnt=0
    for ch in text:
        if ch ==' ' and space_cnt != 1:
            space_cnt = (space_cnt+1)
            res.append(cur_str)
            cur_str=''
        else:
            space_cnt=0
            cur_str+=ch
    if cur_str != '':
        res.append(cur_str)
    return res

def phoneme_str_to_seq(phonemes):
    def _should_keep_phoneme(phoneme):
        return phoneme in _phoneme_to_id and phoneme != '_' and phoneme != '~'

    phonemes=split_by_single_space(phonemes)
    new_phonemes=[]
    for ph in phonemes:
        if ph[-1].isdigit():
            new_phonemes.append(ph[:-1])
            new_phonemes.append(ph[-1])
        else:
            new_phonemes.append(ph)
    phonemes=new_phonemes

    sequence = [_phoneme_to_id[p] for p in phonemes if _should_keep_phoneme(p)]
    # Append EOS token
    sequence.append(_phoneme_to_id['~'])
    # print(phonemes)
    # print(sequence)
    # print(len(phonemes))
    # print(len(sequence))
    # exit()
    return sequence

def seq_to_cnen_mask(seq):
    mask = ''
    first_good = 0
    for t in seq:
        if _id_to_symbol[t] in _pinyinchars:
            first_good = 0
            break
        elif _id_to_symbol[t] in _english2latin:
            first_good = 1
            break
    prev = first_good
    for t in seq:
        if _id_to_symbol[t] in _pinyinchars:
            mask += str(0)
            prev = 0
        elif _id_to_symbol[t] in _english2latin:
            mask += str(1)
            prev = 1
        else:
            mask += str(prev)
    assert len(seq) == len(mask)
    return mask


def text_to_sequence(text, cleaner_names):
    '''Converts a string of text to a sequence of IDs corresponding to the symbols in the text.

      The text can optionally have ARPAbet sequences enclosed in curly braces embedded
      in it. For example, "Turn left on {HH AW1 S S T AH0 N} Street."

      Args:
        text: string to convert to a sequence
        cleaner_names: names of the cleaner functions to run the text through

      Returns:
        List of integers corresponding to the symbols in the text
    '''
    sequence = []

    # Check for curly braces and treat their contents as ARPAbet:
    while len(text):
        m = _curly_re.match(text)
        if not m:
            sequence += _symbols_to_sequence(_clean_text(text, cleaner_names))
            break
        sequence += _symbols_to_sequence(
            _clean_text(m.group(1), cleaner_names))
        sequence += _arpabet_to_sequence(m.group(2))
        text = m.group(3)

    # Append EOS token
    sequence.append(_symbol_to_id['~'])
    return sequence


def sequence_to_text(sequence):
    '''Converts a sequence of IDs back to a string'''
    result = ''
    for symbol_id in sequence:
        if symbol_id in _id_to_symbol:
            s = _id_to_symbol[symbol_id]
            # Enclose ARPAbet back in curly braces:
            if len(s) > 1 and s[0] == '@':
                s = '{%s}' % s[1:]
            result += s
    return result.replace('}{', ' ')


def _clean_text(text, cleaner_names):
    for name in cleaner_names:
        cleaner = getattr(cleaners, name)
        if not cleaner:
            raise Exception('Unknown cleaner: %s' % name)
        text = cleaner(text)
    return text


def _symbols_to_sequence(symbols):
    return [_symbol_to_id[s] for s in symbols if _should_keep_symbol(s)]


def _arpabet_to_sequence(text):
    return _symbols_to_sequence(['@' + s for s in text.split()])


def _should_keep_symbol(s):
    return s in _symbol_to_id and s != '_' and s != '~'
