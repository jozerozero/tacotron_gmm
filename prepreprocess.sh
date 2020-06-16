data_dir=$(basename $1)                         
wavdir=${data_dir}/Wave                         
textdir=${data_dir}/ProsodyLabeling                         
backup=${data_dir}/.backup                         
text_meta=${textdir}/000001-010000.txt                         
                         
rm $text_meta                         
mkdir $wavdir                         
mkdir $textdir                         
mkdir $backup                         
mv $data_dir/*.txt $backup                         
mv $data_dir/*.wav $backup                         
mv $data_dir/*.mp3 $backup                         
                         
find $backup -wholename "*.wav" -exec bash -c 'newfn=$(basename $0 .wav);sox -D -G $0 -b 16 -r 16000 $1/${newfn}.wav remix 1' {} $wavdir \;                         
find $backup -wholename "*.mp3" -exec bash -c 'newfn=$(basename $0 .mp3);ffmpeg -i $0 -acodec pcm_s16le -ac 1 -ar 16000 $1/${newfn}.wav' {} $wavdir \;                         
                         
for txtfile in $backup/*.txt; do                         
    echo -e "$(basename $txtfile .txt)\t$(cat $txtfile)" >> $text_meta                         
done
