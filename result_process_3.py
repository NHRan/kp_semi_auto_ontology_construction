import xlrd
import collections
import pyLDAvis.sklearn
import pyLDAvis
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import pandas as pd
import jieba
import re
import os
import wordcloud as wc


# 根据关系类别词筛选其前后概念词语并存储


if __name__ == '__main__':
    # 主要谓词文档
    filepath = "./result/聚类结果/r.txt"
    text = xlrd.open_workbook("./ere2.xls")
    sheet = text.sheet_by_name('谓词')
    sheet1 = text.sheet_by_name('前概念词')
    sheet2 = text.sheet_by_name('后概念词')
    rows = sheet.nrows  # 获取所有行数
    cols = sheet.ncols  # 获取关系列数
    cols1 = sheet1.ncols  # 获取关系前元素列数
    cols2 = sheet2.ncols  # 获取关系后元素列数
    # print(cols1)
    # print(cols2)
    # 获取所有谓词
    word = []
    for i in range(0, rows):
        word.append(sheet.cell_value(i, 0))
    print("谓词数：", len(word))
    # 读取数据
    with open(filepath, 'r', encoding='utf-8') as f:
        line = f.readlines()
        for i in range(0, len(line)):
            w = line[i].replace("\n", '').split(' ')
            print(w)
            # 根据主要谓词，筛选其前后概念词语
            word_before = []
            word_after = []
            f1 = open("./result/概念结果/前概念/e"+str(i+1)+".txt", 'a', encoding='utf-8')
            f2 = open("./result/概念结果/后概念/e"+str(i+1)+".txt", 'a', encoding='utf-8')
            for s in w:
                for i in range(0, len(word)-1):
                    if(word[i] == s):
                        for j in range(0, cols1):
                            if (len(sheet1.cell_value(i, j)) != 0):
                                f1.write(sheet1.cell_value(i, j) + ' ')
                                word_before.append(sheet1.cell_value(i, j))
                        f1.write("\n")
                        for j in range(0, cols2):
                            if (len(sheet2.cell_value(i, j)) != 0):
                                f2.write(sheet2.cell_value(i, j) + ' ')
                                word_after.append(sheet2.cell_value(i, j))
                        f2.write("\n")
            print("前概念词：", word_before)
            print("后概念词：", word_after)

    # word_before = []
    # word_after = []
    # # 根据主要关系词，筛选其前后词语
    # w = ["历任", "兼任", "调任", "出任", "任命", "出任", "到任", "接任"]
    # for s in w:
    #     for i in range(0, len(word)):
    #         if(word[i] == s):
    #             for j in range(0, cols1):
    #                 if (len(sheet1.cell_value(i, j)) != 0):
    #                     word_before.append(sheet1.cell_value(i, j))
    #             for j in range(0, cols2):
    #                 if (len(sheet2.cell_value(i, j)) != 0):
    #                     word_after.append(sheet2.cell_value(i, j))
    #
    # print("关系前元素：", word_before)
    # print("关系：", w)
    # print("关系后元素：", word_after)

    # # 词频统计
    # count1 = dict(collections.Counter(word_before))
    # count2 = dict(collections.Counter(word_after))
    # print("关系前元素词频：", count1)
    # print("关系后元素词频：", count2)
    #
    # # 生成词云
    # w1 = wc.WordCloud(font_path="C:/Windows/Fonts/simhei.ttf", background_color="white", max_font_size=25, width=400, height=260, max_words=100)
    # w2 = wc.WordCloud(font_path="C:/Windows/Fonts/simhei.ttf", background_color="white", max_font_size=25, width=400, height=260, max_words=100)
    # w1.fit_words(count1)
    # w2.fit_words(count2)
    # w1.to_file("关系前元素.png")
    # w2.to_file("关系后元素.png")

