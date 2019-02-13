#!/usr/bin/env python
# -*- coding: utf-8 -*-

def num2cn(number, traditional=False):
    '''数字转化为中文
    参数：
        number: 数字
        traditional: 是否使用繁体
    '''
    chinese_num = {
        'Simplified': ['零', '一', '二', '三', '四', '五', '六', '七', '八', '九'],
        'Traditional': ['零', '壹', '贰', '叁', '肆', '伍', '陆', '柒', '捌', '玖']
    }
    chinese_unit = {
        'Simplified': ['个', '十', '百', '千'],
        'Traditional': ['个', '拾', '佰', '仟']
    }
    extra_unit = ['', '万', '亿']

    if traditional:
        chinese_num = chinese_num['Traditional']
        chinese_unit = chinese_unit['Traditional']
    else:
        chinese_num = chinese_num['Simplified']
        chinese_unit = chinese_unit['Simplified']

    num_cn = []

    # 数字转换成字符列表
    num_list = list(str(number))

    # 反转列表，个位在前
    num_list.reverse()

    # 数字替换成汉字
    for num in num_list:
        num_list[num_list.index(num)] = chinese_num[int(num)]

    # 每四位进行拆分，第二个四位加“万”，第三个四位加“亿”
    for loop in range(len(num_list)//4+1):
        sub_num = num_list[(loop * 4):((loop + 1) * 4)]
        if not sub_num:
            continue

        # 是否增加额外单位“万”、“亿”
        if loop > 0 and 4 == len(sub_num) and chinese_num[0] == sub_num[0] == sub_num[1] == sub_num[2] == sub_num[3]:
                use_unit = False
        else:
            use_unit = True

        # 合并数字和单位，单位在每个数字之后
        # from itertools import chain
        # sub_num = list(chain.from_iterable(zip(chinese_unit, sub_num)))
        sub_num = [j for i in zip(chinese_unit, sub_num) for j in i]

        # 删除第一个单位 '个'
        del sub_num[0]

        # “万”、“亿”中如果第一位为0则需加“零”: 101000，十万零一千
        use_zero = True if loop > 0 and chinese_num[0] == sub_num[0] else False

        if len(sub_num) >= 7 and chinese_num[0] == sub_num[6]:
            del sub_num[5]  # 零千 -> 零
        if len(sub_num) >= 5 and chinese_num[0] == sub_num[4]:
            del sub_num[3]  # 零百 -> 零
        if len(sub_num) >= 3 and chinese_num[0] == sub_num[2]:
            del sub_num[1]  # 零十 -> 零
        if len(sub_num) == 3 and chinese_num[1] == sub_num[2]:
            del sub_num[2]  # 一十开头的数 -> 十

        # 删除末位的零
        while len(num_list) > 1 and len(sub_num) and chinese_num[0] == sub_num[0]:
            del sub_num[0]

        # 增加额外的“零”
        if use_zero and len(sub_num) > 0:
            num_cn.append(chinese_num[0])

        # 增加额外单位“万”、“亿”
        if use_unit:
            num_cn.append(extra_unit[loop])

        num_cn += sub_num

    # 删除连续重复数据：零，只有零会重复
    num_cn = [j for i, j in enumerate(num_cn) if i == 0 or j != num_cn[i-1]]
    # 删除末位的零，最后一位为 extra_unit 的 ''
    if len(num_list) > 1 and len(num_cn) > 1 and chinese_num[0] == num_cn[1]:
        del num_cn[1]

    # 反转并连接成字符串
    num_cn.reverse()
    num_cn = ''.join(num_cn)

    return(num_cn)

if '__main__' == __name__:
    for num in [0, 5, 100020034005, 10020000, 123456789, 1000000000, 10, 110000, 10000000000, 100000000000]:
        print('%d: %s, %s' % (num, num2cn(num, False), num2cn(num, True)))

"""
    from itertools import permutations
    import copy

    test = ['1', '1', '1', '1', '1', '1', '1', '1']
    num_list = []
    for vid in range(len(test)):
        for nid in permutations(range(len(test)), vid):
            tmp = copy.copy(test)
            for index in nid:
                tmp[index] = '0'
            num_list.append(int(''.join(tmp)))

    num_list = list(set(num_list))
    for number in num_list:
        print('%d: %s' % (number, num2cn(number, False)))
"""
