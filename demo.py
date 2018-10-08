#!/usr/bin/python3 
# -*- coding:utf-8 -*-
# Author:anning

import jieba
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def read_file(path,stop_path):
    '''
    对句子进行分词并去停止词
    :path  文件路径
    '''
    f = open(path, "r")
    f_stop = open(stop_path, "r")
    stop_temp= []
    for line in f_stop:
        if line.strip()=="":
            continue
        stop_temp.append(line.strip())

    data = []
    for line in f:
        if line.strip()=="":
            continue
        seg = list(jieba.cut(line.strip(),cut_all=False))
        for i in seg:
            if i not in stop_temp:
                data.append(i)
    return data

def get_weight(data):
    '''
    :data  数据列表
    '''
    #计算TF-IDF权重
    vectorizer = CountVectorizer()
    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(vectorizer.fit_transform(data))
    #获取词袋模型中的所有词语特征
    word = vectorizer.get_feature_names()
    # 导出向量, 矩阵的每一行就是文档的向量表示
    tfidf_weight = tfidf.toarray()            
    return tfidf_weight
   
def get_kmeans(tfidf_weight):

    kmeans = KMeans(n_clusters=2)
    kmeans.fit(tfidf_weight)    
    # 打印各个簇的中心点
    print(kmeans.cluster_centers_)
    for index, label in enumerate(kmeans.labels_, 1):
        print("index: {}, label: {}".format(index, label))
    # 样本距其最近的聚类中心的平方距离之和，用来评判分类的准确度，值越小越好
# k-means的超参数n_clusters可以通过该值来评估
    print("inertia: {}".format(kmeans.inertia_))

def main():

    #对数据进行分词和去停止词
    path = "./test.txt"
    stop_path = "./stop.txt"
    data  = read_file(path, stop_path)
    #tf-idf权重
    tfidf_weight = get_weight(data)
    #文本聚类
    get_kmeans(tfidf_weight)
    

if __name__=="__main__":
    main()





