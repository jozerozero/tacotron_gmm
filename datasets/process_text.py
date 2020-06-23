import re


def process_en(text):
    word_phone_list = text.split(" ")
    phone_list = []
    stress_list = []
    print(text)
    for word_phone in word_phone_list:
        
        for phone in word_phone.split("$"):
            phone_stress = re.findall(r"\d+\.?\d*", phone)
            if len(phone_stress) == 0:
                stress = "3"
            else:
                stress = str(phone_stress[0])
        
            phone = phone.replace(str(stress), "")
            if len(phone) == 0:
                continue
            if "." in phone:
                continue
        
            phone_list.append(phone)
            stress_list.append(stress)

        phone_list .append("$")
        stress_list.append("3")

    return phone_list, stress_list


def get_pinyin2cmu_dict():
    pinyin2cmu_dict = dict()
    for line in open("datasets/pinyin2cmu.txt"):
        line = line.strip().split(" ", 1)
        pinyin2cmu_dict[line[0]] = line[1]

    return pinyin2cmu_dict


# def process_cn(text):
#     phone_list = []
#     stress_list = []
#
#     for word in text.split(" "):
#         if "#" in word:
#             phone_list.append(word)
#
#
#
#     pass
