path = "/home/lizijian/data0/tacotron_multi_lingual/tacotron/training_data/train.txt"

count = 0
for line in open(path):
    line = line.strip().split("|")
    phone_list = line[5].split(" ")
    tone_list = line[6].split(" ")
    # print(len(phone_list))
    # print(len(tone_list))
    if len(phone_list) != len(tone_list):
        count += 1
        print(phone_list)
        print(tone_list)
        print(len(phone_list))
        print(len(tone_list))
        # exit()

print(count)
print("end")