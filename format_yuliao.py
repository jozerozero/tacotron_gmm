import re
from tacotron.utils.cn_convert import cn_convert
import sys

filename=sys.argv[1]
with open(filename, 'r', encoding='utf-8') as f:
    for line in f:
        #if line[0]=='1':
        #    continue
        #line=line[:6]+'\t'+line[7:]
        #fields=re.compile('[ \t]').split(line)
        fields=line.split('\t')
        print('%s\t%s'%(fields[0],cn_convert(fields[1])))
#awk -F'\t' 'length($0)>1{print $1}' ProsodyLabeling/000001-010000.txt  | sort > mem1
#find Wave |  awk -F'[/\.]' '{print $2}' > mem2
