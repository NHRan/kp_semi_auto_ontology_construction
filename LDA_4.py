# coding:utf-8
import pyLDAvis.sklearn
import pyLDAvis
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import pandas as pd
import jieba
import re
import os
from sklearn.feature_extraction.text import CountVectorizer


def cut_stopwords(stopwords,word):
    word_after = []
    for i in word:
        if i not in stopwords:
            word_after.append(i)
    return word_after


def top_words_data_frame(model: LatentDirichletAllocation,
                         cntVector: CountVectorizer,
                         n_top_words: int) -> pd.DataFrame:
    '''
    求出每个主题的前 n_top_words 个词

    Parameters
    ----------
    model : sklearn 的 LatentDirichletAllocation
    tf_idf_vectorizer : sklearn 的 TfidfVectorizer
    n_top_words :前 n_top_words 个主题词

    Return
    ------
    DataFrame: 包含主题词分布情况
    '''
    rows = []
    feature_names = cntVector.get_feature_names()
    for topic in model.components_:
        top_words = [feature_names[i]
                     for i in topic.argsort()[:-n_top_words - 1:-1]]
        rows.append(top_words)
    columns = [f'topic {i+1}' for i in range(n_top_words)]
    df = pd.DataFrame(rows, columns=columns)

    return df


def predict_to_data_frame(model: LatentDirichletAllocation, X: np.ndarray) -> pd.DataFrame:
    '''
    求出文档主题概率分布情况

    Parameters
    ----------
    model : sklearn 的 LatentDirichletAllocation
    X : 词向量矩阵

    Return
    ------
    DataFrame: 包含主题词分布情况
    '''
    matrix = model.transform(X)
    columns = [f'P(topic {i+1})' for i in range(len(model.components_))]
    df = pd.DataFrame(matrix, columns=columns)
    return df


def LDA(list):
    if (len(list) != 0):
        # 词语文档传入
        cntVector = CountVectorizer()
        # 词频矩阵
        cntTf = cntVector.fit_transform(list)

        # 指定 LDA 主题数
        n_topics = 3
        lda = LatentDirichletAllocation(
            n_components=n_topics, max_iter=50,
            learning_method='online',
            learning_offset=50.,
            random_state=0)
        # 给 LDA 生成的矩阵
        lda.fit(cntTf)
        n_top_word = 9
        top_words_df = top_words_data_frame(lda, cntVector, n_top_word)
        print(top_words_df)

    else:
        print("list is NULL!!")



    # # 可视化 html 文件路径
    # html_path = './lda-visualization.html'
    # # 使用 pyLDAvis 进行可视化
    # data = pyLDAvis.sklearn.prepare(lda, cntTf, cntVector)
    # pyLDAvis.save_html(data, html_path)
    # # 清屏
    # os.system('clear')
    # # 浏览器打开 html 文件以查看可视化结果
    # os.system(f'start {html_path}')


if __name__ == '__main__':
    # 对同一关系的前后概念词语进行LDA主题提取，存储以用来确定主要概念

    for i in range(0, 17):
        print(i+1)
        f1 = open("./result/概念结果/前概念/e" + str(i + 1) + ".txt", 'r', encoding='utf-8')
        f2 = open("./result/概念结果/后概念/e" + str(i + 1) + ".txt", 'r', encoding='utf-8')
        before_list = []
        after_list = []
        for line in f1:
            if(line != ' ' and line != '\n'):
                line = line.replace("\n", '')
                before_list.append("".join(line))
        print(before_list)
        for line in f2:
            if(line != ' ' and line != '\n'):
                line = line.replace("\n", '')
                after_list.append("".join(line))
        print(after_list)
        # 主题词提取
        LDA(before_list)
        LDA(after_list)
        print("=================================")