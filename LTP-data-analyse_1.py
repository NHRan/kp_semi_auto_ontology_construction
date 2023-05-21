import jieba
import jieba.posseg as pos
import jieba.analyse
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict
import collections
from ltp import LTP
import xlwt

# 对句子进行LTP分词，并根据规则区分  前相关实体 - 关系 - 后相关实体，存储到ere.xls

if __name__ == '__main__':
    # 句子文档
    filepath = './data/data3.txt'  # 领域数据路径
    # LTP分词
    print("=== LTP分词 ===")
    ltp = LTP()
    text = []
    text_n = []
    h = 0
    # 创建3个表
    book = xlwt.Workbook(encoding='utf-8', style_compression=0)
    # sheet1存储关系前元素
    sheet1 = book.add_sheet('前概念词', cell_overwrite_ok=True)
    # sheet存储关系
    sheet = book.add_sheet('谓词', cell_overwrite_ok=True)
    # sheet2存储关系前元素
    sheet2 = book.add_sheet('后概念词', cell_overwrite_ok=True)


    # 读取数据
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:  # 按行读取文件
            line_text = []
            line_text.append(line.replace('\n', ''))
            # 分词
            word, hidden = ltp.seg(line_text)
            #print("--------------------------------------------------")
            #print(word)

            # 词性标注 - 获得动词
            text_v = []
            pos = ltp.pos(hidden)
            for i in range(0, len(pos[0])):
                 if(pos[0][i] == 'v' ):
                    text_v.append(word[0][i])
            # print(text_v)


            # 语义角色标注,直接获取谓词主要组合/获不完整的组合
            #print("== 语义角色标注 ==")
            srl = ltp.srl(hidden, keep_empty=False)  # 每句话的语义标注结果
            for i in srl[0]:
                # 如果该谓词在动词里面，写入文档
                # word[0][i[0]]为关系词
                if word[0][i[0]] in text_v:
                    flag = 0
                    tag_str = []
                    # 判断是否含有主要元素
                    for j in range(0, len(i[1])):
                        if('A0' == i[1][j][0] or 'A1' == i[1][j][0] or 'A2' == i[1][j][0] or 'ARGM-TMP'== i[1][j][0] or 'ARGM-LOC'== i[1][j][0]):
                            flag = 1
                        tag_str.append(i[1][j][0])
                    # 当含有主要元素时
                    if(flag == 1):
                        print(i)
                        print(word[0][i[0]])
                        # 关系写入
                        sheet.write(h, 0, word[0][i[0]])
                        # 有A1 - 关系A1前
                        if ('A1' in tag_str):
                            k = tag_str.index('A1')
                            for j in range(0, len(i[1])):
                                if(j < k):
                                    if (i[1][j][1] != i[1][j][2]):
                                        sheet1.write(h, j, word[0][i[1][j][1]] + word[0][i[1][j][2]])
                                        print(word[0][i[1][j][1]] + word[0][i[1][j][2]])
                                    else:
                                        sheet1.write(h, j, word[0][i[1][j][1]])
                                        print(word[0][i[1][j][1]])
                                else:
                                    if (i[1][j][1] != i[1][j][2]):
                                        sheet2.write(h, j-k, word[0][i[1][j][1]] + word[0][i[1][j][2]])
                                        print(word[0][i[1][j][1]] + word[0][i[1][j][2]])
                                    else:
                                         sheet2.write(h, j-k, word[0][i[1][j][1]])
                                         print(word[0][i[1][j][1]])
                        # 有A2无A1 - 关系A2前
                        elif ('A2'in tag_str and 'A1' not in tag_str):
                            k = tag_str.index('A2')
                            for j in range(0, len(i[1])):
                                if (j < k):
                                    if (i[1][j][1] != i[1][j][2]):
                                        sheet1.write(h, j, word[0][i[1][j][1]] + word[0][i[1][j][2]])
                                        print(word[0][i[1][j][1]] + word[0][i[1][j][2]])
                                    else:
                                        sheet1.write(h, j, word[0][i[1][j][1]])
                                        print(word[0][i[1][j][1]])
                                else:
                                    if (i[1][j][1] != i[1][j][2]):
                                        sheet2.write(h, j-k, word[0][i[1][j][1]] + word[0][i[1][j][2]])
                                        print(word[0][i[1][j][1]] + word[0][i[1][j][2]])
                                    else:
                                        sheet2.write(h, j-k, word[0][i[1][j][1]])
                                        print(word[0][i[1][j][1]])
                        # 只有A0 - 关系最后
                        elif ('A0' in tag_str and 'A1' not in tag_str and 'A2' not in tag_str):
                            for j in range(0, len(i[1])):
                                if (i[1][j][1] != i[1][j][2]):
                                    sheet1.write(h, j, word[0][i[1][j][1]] + word[0][i[1][j][2]])
                                    print(word[0][i[1][j][1]] + word[0][i[1][j][2]])
                                else:
                                    sheet1.write(h, j, word[0][i[1][j][1]])
                                    print(word[0][i[1][j][1]])

                        # A0A1A2都无 -关系最后
                        else:
                            for j in range(0, len(i[1])):
                                if (i[1][j][1] != i[1][j][2]):
                                    sheet1.write(h, j, word[0][i[1][j][1]] + word[0][i[1][j][2]])
                                    print(word[0][i[1][j][1]] + word[0][i[1][j][2]])
                                else:
                                    sheet1.write(h, j, word[0][i[1][j][1]])
                                    print(word[0][i[1][j][1]])

                        h = h + 1
        book.save('./ere2.xls')



            # # 依存句法分析
            # print("== 依存句法 ==")
            # dep = ltp.dep(hidden)
            # flag1 = 0
            # flag2 = 0
            # list = []
            # for j in range(0, len(dep[0])):
            #     if(dep[0][j][2] == 'SBV'):   # 主谓
            #         list.append(dep[0][j])
            #         flag1 = 1
            #     if (dep[0][j][2] == 'VOB'):  # 动宾
            #         list.append(dep[0][j])
            #         flag2 = 1
            #
            # if(len(list) != 0):
            #     if(flag1 ==1 and flag2 ==1):
            #         for n in range(0, len(list)):
            #                 if(list[n][2] == 'SBV'):
            #                     for m in range(0, len(list)):
            #                         if(list[m][2] == 'VOB' and (list[m][1] == list[n][1])):
            #                             print(".....")
            #                             print(list[n])
            #                             print(list[m])
            #                             print(word[0][list[n][0] - 1] + ' ' + word[0][list[n][1] - 1] + ' ' + word[0][list[m][0] - 1])
            #                             print(".....")






