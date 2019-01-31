import re
#awk -F'\t' 'length($0)>1{print $1}' ProsodyLabeling/000001-010000.txt  | sort > mem1
#find Wave |  awk -F'[/\.]' '{print $2}' > mem2
from tacotron.utils.cn_convert import cn_convert
with open('yuliao1.txt', 'r', encoding='utf-8') as f:
    for line in f:
        line=line[:6]+'\t'+line[7:]
        #fields=re.compile('[ \t]').split(line)
        fields=line.split('\t')
        print('%s\t%s'%(fields[0],cn_convert(fields[1])))
