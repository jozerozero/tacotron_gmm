#!/bin/bash
# $1: metafile, $2: lineno

#metafile=$1
#lineno=$2
#lineno=$(( lineno * 2-1 ))
wavefolder='BZNSYP/Wave'
specfolder='training_data/linear'

#echo $lineno

while getopts ":w:s:t:n:" opt; do
  case $opt in
    w)
      echo "-w wave folder is $OPTARG" >&2
      wavefolder=$OPTARG
      ;;
    s)
      echo "-s spec folder is $OPTARG" >&2
      specfolder=$OPTARG
      ;;
    t)
      echo "-t text file is $OPTARG" >&2
      metafile=$OPTARG
      ;;
    n)
      lineno=$OPTARG
      lineno=$(( lineno * 2-1 ))
      echo "-n lineno is $OPTARG" >&2
      ;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      ;;
  esac
done

echo sed -n "${lineno}p" $metafile
theline=$(sed -n "${lineno}p" $metafile)
IFS='\t' read -ra linearr <<< "$theline"
theid=''
thetext=''
cnt=1
for i in ${linearr[@]}
do
	if [ $cnt = 1 ]
	then
		echo $i
		theid=$i
		cnt=2
	else
		echo $i
        thetext=$i
	fi
done

linearspec="linear-${theid}.wav.npy"

#finish linespec of text
./test_demo.sh $thetext
echo tmp-linear-spectrogram.png generated

#finish linespec of text ground truth
python plot_training_linearspec.py $specfolder/$linearspec
echo tmp-training-linear-spectrogram.png generated

python plot_wav.py temp.wav tmp_synch_wav.png
echo  tmp_synch_wav.png generated
python plot_wav.py $wavefolder/${theid}.wav tmp_ref_wav.png
echo " ref wav is $wavefolder/${theid}.wav"
echo  tmp_ref_wav.png generated
python MCD.py temp.wav $wavefolder/${theid}.wav
#paplay -s 127.0.0.1:9999 Wave/${theid}.wav
