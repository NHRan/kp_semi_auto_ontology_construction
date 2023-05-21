import xlrd
import collections
#from words_bert_kmeans import bert_kmeans

#  统计词频  查看

if __name__ == '__main__':
    # 主要元素文档
    text = xlrd.open_workbook("./ere2.xls")
    # 关系表
    sheet = text.sheet_by_name('谓词')
    rows = sheet.nrows  # 获取所有行数
    print(rows)
    # 获取关系词
    word = []
    for i in range(0, rows):
        word.append(sheet.cell_value(i, 0))

    # 词频统计
    counts = dict(collections.Counter(word))
    print("去重词数：", len(counts))

    # 获取词频的最大值与和
    max = 1
    c = 0
    value = []
    for i in counts:
        c = c + counts[i]
        value.append(counts[i])
        if (counts[i] > max):
            max = counts[i]
    print("词频最大值：", max)
    print(value)
    value.sort()
    print(value)
    print(value[566])

    # 平均数
    th = c / len(counts)
    print("平均数：", th)




    # # 词语按字数分开
    # key_list = []
    # for i in range(1, max+1):
    #     list = {}
    #     for key in counts:
    #         if len(key) == i:
    #             list[key] = counts[key]
    #     # print(list)
    #     key_list.append(list)
    #
    # v_after = []
    # # 对比1 2
    # v_after1 = []
    # v_after12 = []
    # v_after2 = []
    # for i, v in enumerate(key_list[0]):
    #     v12 = []
    #     flag = 0
    #     for j, w in enumerate(key_list[1]):
    #         if v in w:
    #             flag = 1
    #             v12.append(w)
    #     if(flag == 0):
    #         v_after1.append(v)
    #     else:
    #         v12.insert(0, v)
    #     if(len(v12) != 0):
    #         v_after12.append(v12)
    # #print(v_after1)
    #
    # for i, v in enumerate(key_list[1]):
    #     flag1 = 0
    #     for j in range(0, len(v_after12)):
    #         if v in v_after12[j]:
    #             flag1 = 1
    #     if(flag1 == 0):
    #         v_after2.append(v)
    #
    # # 对比2 3
    # v_after23 = []
    # for i in range(0, len(v_after12)):
    #     for j in range(0, len(v_after12[i])):
    #         if (len(v_after12[i][j]) == 2):
    #             for k, v in enumerate(key_list[2]):
    #                 if v_after12[i][j] in v:
    #                     v_after12[i].append(v)
    #
    # for i in v_after2:
    #     v23 = []
    #     flag2 = 0
    #     for j, v in enumerate(key_list[2]):
    #         if i in v:
    #             flag2 = 1
    #             v23.append(v)
    #
    #     if(flag2 == 1):
    #         v23.insert(0, i)
    #         v_after2.remove(i)
    #     if(len(v23) != 0):
    #         v_after23.append(v23)
    #
    # # print(v_after2)
    # # print(v_after23)
    #
    # v_after.extend(v_after1)
    # v_after.extend(v_after2)
    # v_after.extend(v_after12)
    # v_after.extend(v_after23)
    # for i in range(3, max):
    #     if(len(key_list[i]) != 0):
    #         for j, v in enumerate(key_list[i]):
    #             v_after.append(v)
    #
    # print(v_after)
    # v_count = []
    # sum = 0
    # for i in range(0, len(v_after)):
    #     if(isinstance(v_after[i], str)):
    #         v_count.append(counts[v_after[i]])
    #         sum = sum + counts[v_after[i]]
    #     else:
    #         c = 0
    #         for j in v_after[i]:
    #             c = c + counts[j]
    #         sum = sum + c
    #         v_count.append(c)
    # print(sum)
    # print(v_count)
    #
    # for i in range(0, len(v_count)):
    #     if(v_count[i]>40):
    #         print("---------------------")
    #         print((v_count[i]/sum)*100)
    #         print(v_after[i])


    # # 词语字数
    # key_list = {}
    # for i in range(1, 4):
    #     for key in counts:
    #         if len(key) == i:
    #             key_list[key] = counts[key]

    # 对字数为1-3进行聚类
    # bert_kmeans(key_list)



    # # 句子文档
    # filepath = './result/2-332/cluster1.txt'
    # with open(filepath, 'r', encoding='utf-8') as f:
    #     words = f.readlines()
    # l = len(words)
    # words_list = []
    # for i in range(0, 332):
    #     print(i)
    #     list = []
    #     count = 0
    #     for j in range(0, l):
    #         w = words[j].strip('\n').split(' ')
    #         if w[0] == str(i):
    #             count = count + 1
    #             list.append(w[1])
    #     print(list)
    #     words_list.append(list)