from transformers import BertTokenizer,BertModel
import sys
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from ltp import LTP
import xlwt
import xlrd
import collections
from sklearn.metrics import calinski_harabasz_score
import pandas as pd
from matplotlib.pyplot import MultipleLocator

def kmeans_cluster(num_clusters,embeds_list):
    # 构造聚类器
    km_cluster = KMeans(n_clusters=num_clusters, max_iter=500, init="k-means++")
    # 聚类
    pred_cluster = km_cluster.fit_predict(embeds_list)
    label_cluster = km_cluster.labels_  # 获取聚类标签
    center_cluster = km_cluster.cluster_centers_  # 获取聚类中心

    return km_cluster, pred_cluster, label_cluster, center_cluster

# 散点图可视化
def vis(title,data,pred_data,fig_name):
    # 设置标题
    plt.title(title)
    plt.scatter(data[:, 0], data[:, 1], c=pred_data, marker='o')
    # 保存图片
    plt.savefig(fig_name)
    plt.clf()

def k_vis(title,X,data,figname):
    # 设置标题
    plt.title(title)
    plt.plot(X, data)
    # 把x轴的刻度间隔设置为1，并存在变量里
    x_major_locator = MultipleLocator(1)
    # ax为两条坐标轴的实例
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    plt.xlim(2, len(data)+2)
    # 把x轴的主刻度设置为1的倍数
    # 保存图片
    plt.savefig(figname)
    plt.clf()

def label_write(cluster_name, label_cluster, sentence_list, num_cluster):
    f = open('./result/去除阈值结果/' + cluster_name + '.txt', 'a', encoding='utf-8')
    for i, item in enumerate(label_cluster):
        for j in range(0, num_cluster):
            if item == j:
                f.write(str(j) + ' ' + sentence_list[i] + '\n')


if __name__ == '__main__':
    # 对所有谓词筛选后使用bert-kmeans聚类，保存结果，聚类后通过手肘+轮廓系数判断最佳K值

    # 主要元素文档
    text = xlrd.open_workbook("./ere2.xls")
    sheet = text.sheet_by_name('谓词')
    rows = sheet.nrows  # 获取所有行数
    # 获取所有谓词
    word = []
    for i in range(0, rows):
        word.append(sheet.cell_value(i, 0))
    print("所有词数：", len(word))
    # 词频统计
    counts = dict(collections.Counter(word))
    print("去重词数：", len(counts))

    # 获取词频的最大值与和
    max = 1
    c = 0
    for i in counts:
        c = c + counts[i]
        if (counts[i] > max):
            max = counts[i]
    print("词频最大值：", max)
    # 设定阈值为平均数
    th = c/len(counts)
    print("阈值：", th)

    # 去除词频小于阈值且长度大于1的词语
    word_after = []
    for i in counts:
        if counts[i] >= th and len(i) > 1:
            word_after.append(i)
    print("去除词频小于阈值且长度大于1的词语后：", len(word_after))
    print(word_after)

    # BERT (将每个句子映射到向量空间，使得语义相似的句子相近,将单个句子输入 BERT 并导出固定大小的句子嵌入)
    tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
    model = BertModel.from_pretrained("bert-base-chinese")
    # 构造BERT输入
    embeds_list = []
    # 进行编码
    for r in range(0, len(word_after)):  #
        print(r)
        input_text = '[CLS]' + word_after[r] + '[SEP]'
        input_tokenized = tokenizer.tokenize(input_text)
        input_id = tokenizer.convert_tokens_to_ids(input_tokenized)
        if len(input_id) < 10:
            input_id.extend([0] * (10 - len(input_id)))
        # 变为tensor
        input_id = torch.tensor([input_id])

        # 对每个句子编码
        embeds = model(input_id).last_hidden_state[0]
        embeds = embeds.detach().numpy()

        # 使用第一个令牌（[CLS] 令牌）的输出
        embeds_list.append(embeds[0])   # 维度[句子总数，768]

    print("编码成功！")
    #p = PCA(n_components=len(word_after)).fit_transform(embeds_list)

    # k-means聚类
    # 确定聚类的K值
    score1 = []  # 存放每次结果的轮廓系数分数
    SSE = []  # 存放每次结果的误差平方和
    score2 = []  # 存放评价指标
    # 创建3个表
    book = xlwt.Workbook(encoding='utf-8', style_compression=0)
    # sheet1存储轮廓系数分数
    sheet1 = book.add_sheet('轮廓系数分数', cell_overwrite_ok=True)
    # sheet2存储误差平方和
    sheet2 = book.add_sheet('误差平方和', cell_overwrite_ok=True)
    # sheet3存储评价指标
    sheet3 = book.add_sheet('评价指标', cell_overwrite_ok=True)
    max_k = 2
    for k in range(2, len(word_after)):
        print(k)
        km_cluster, pred_cluster, label_cluster, center_cluster = kmeans_cluster(k, embeds_list)
        # 轮廓系数法
        ss = silhouette_score(embeds_list, label_cluster)
        score1.append(ss)
        sheet1.write(k, 0, k)
        sheet1.write(k, 1, str(ss))
        # 手肘法
        ki = km_cluster.inertia_   # 获取聚类准则的总和
        SSE.append(ki)
        sheet2.write(k, 0, k)
        sheet2.write(k, 1, str(ki))
        # 评价指标
        chs = calinski_harabasz_score(embeds_list, label_cluster)
        score2.append(chs)
        sheet3.write(k, 0, k)
        sheet3.write(k, 1, str(chs))

        # 将聚类结果写入存储
        if k <= 50:
            name = 'cluster' + str(k)
            # 将每个类对应句子写进文件
            label_write(name, label_cluster, word_after, k)
            print("结果写入成功!")
    book.save('./value/去除阈值结果/k_value.xls')

    # 画图保存
    max = score1[0]
    for s in range(1, len(score1)):
        if(score1[s]>max):
            max = score1[s]
            max_k = 2 + s
    print("确定K值为：", max_k)

    X = range(2, len(word_after))
    X1 = range(2, 100)
    X2 = range(2, 40)
    X3 = range(2, 30)
    X4 = range(2, 15)
    score12 = score1[0:38]
    SSE1 = SSE[0:98]
    SSE2 = SSE[0:38]
    SSE3 = SSE[0:28]
    SSE4 = SSE[0:13]
    score22 = score2[0:38]
    print(SSE)
    k_vis("轮廓系数", X, score1, "./value/去除阈值结果/轮廓系数.png")
    k_vis("轮廓系数2", X2, score12, "./value/去除阈值结果/轮廓系数2.png")
    k_vis("手肘", X, SSE, "./value/去除阈值结果/手肘.png")
    k_vis("手肘1", X1, SSE1, "./value/去除阈值结果/手肘1.png")
    k_vis("手肘2", X2, SSE2, "./value/去除阈值结果/手肘2.png")
    k_vis("手肘3", X3, SSE3, "./value/去除阈值结果/手肘3.png")
    k_vis("手肘4", X4, SSE4, "./value/去除阈值结果/手肘4.png")
    k_vis("评价", X, score2, "./value/去除阈值结果/评价.png")
    k_vis("评价2", X2, score22, "./value/去除阈值结果/评价2.png")


    '''

    km_cluster, pred_cluster, label_cluster, center_cluster = kmeans_cluster(22, p)
    print("聚类成功！")
    print("轮廓系数：", silhouette_score(p, label_cluster))
    print("评价：", calinski_harabasz_score(p, label_cluster))

    # 聚类结果写入文件
    # 聚类中心
    cc = open('./result/去除12结果/cluster_center', 'a', encoding='utf-8')
    cc.write(str(center_cluster))
    for n, embed in enumerate(embeds_list):
        if embed in center_cluster:
            cc.write(word_after[n] + '\n')

    # 将每个类对应句子写进文件
    label_write("cluster", label_cluster, word_after, 22)
    print("结果写入成功！")

    # # PCA降维-线性降维
    # pca = PCA(n_components=2).fit_transform(embeds_list)  # [510,2]
    # pca1 = PCA(n_components=2).fit_transform(embeds_list1)
    # # 可视化
    # vis("avg clustering result(pca)", pca, pred_cluster, "./result/avg_pca.png")
    # vis("cls clustering result(pca)", pca1, pred_cluster1, "./result/cls_pca.png")
'''