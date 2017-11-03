**原文链接**
> https://machinelearningmastery.com/develop-word-embedding-model-predicting-movie-review-sentiment/

# How to Develop a Word Embedding Model for Predicting Movie Review Sentiment

by Jason Brownlee on October 30, 2017 in Natural Language Processing

***

Word embeddings are a technique for representing text where different words with similar meaning have a similar real-valued vector representation.

They are a key breakthrough that has led to great performance of neural network models on a suite of challenging natural language processing problems.

* 词嵌入是一种用来用实数向量表示文本的技术。
* 特别适合表示包含近义词的文本。
* 这类技术导致了神经网络在一系列自然语言处理问题中极大的性能提升。

In this tutorial, you will discover how to develop word embedding models for neural networks to classify movie reviews.

After completing this tutorial, you will know:

How to prepare movie review text data for classification with deep learning methods.

How to learn a word embedding as part of fitting a deep learning model.

How to learn a standalone word embedding and how to use a pre-trained embedding in a neural network model.

Let’s get started.

* 在此教程中，你会知晓如何开发包含词嵌入模型的神经网络用来分类影评。
* 在完成教程后，你会知晓：
    * 如何为深度神经网络分类器准备影评文本数据。
    * 如何把词嵌入作为深度神经网络模型的一部分。
    * 如何训练一个独立的词嵌入模型，如何把预训练的词嵌入模型应用于神经网络模型。
* 我们开始教程。



## Tutorial Overview

This tutorial is divided into 5 parts; they are:

1. 影评数据集 (Movie Review Dataset)
2. 数据准备 (Data Preparation)
3. 训练嵌入层 (Train Embedding Layer)
4. 训练word2vec嵌入模型 (Train word2vec Embedding)
5. 使用预训练的嵌入模型 (Use Pre-trained Embedding)


## Movie Review Dataset

The Movie Review Data is a collection of movie reviews retrieved from the imdb.com website in the early 2000s by Bo Pang and Lillian Lee. The reviews were collected and made available as part of their research on natural language processing.

* **影评数据集（Movie Review Data）**是20世纪初**Bo Pang**和**Lilian Lee**从imdb.com网站上收集的一系列影评。

The reviews were originally released in 2002, but an updated and cleaned up version were released in 2004, referred to as “v2.0”.

* 这些评论最初在2002年发布。
* 但是在2004年发布了更新和清洗过的版本，称为v2.0。

The dataset is comprised of 1,000 positive and 1,000 negative movie reviews drawn from an archive of the rec.arts.movies.reviews newsgroup hosted at imdb.com. The authors refer to this dataset as the “polarity dataset.”

> Our data contains 1000 positive and 1000 negative reviews all written before 2002, with a cap of 20 reviews per author (312 authors total) per category. We refer to this corpus as the polarity dataset.

— A Sentimental Education: Sentiment Analysis Using Subjectivity Summarization Based on Minimum Cuts, 2004.

* 数据集由从imdb.com的**rec.arts.movies.reviews**的新闻服务器存档中，收集的1000个正面评价1000个负面评价。
* 作者们称这个数据集为**polarity dataset**
* 引用内容同上。


The data has been cleaned up somewhat, for example:

* The dataset is comprised of only English reviews.
* All text has been converted to lowercase.
* There is white space around punctuation like periods, commas, and brackets.
* Text has been split into one sentence per line.

The data has been used for a few related natural language processing tasks. For classification, the performance of machine learning models (such as Support Vector Machines) on the data is in the range of high 70% to low 80% (e.g. 78%-82%).

More sophisticated data preparation may see results as high as 86% with 10-fold cross validation. This gives us a ballpark of low-to-mid 80s if we were looking to use this dataset in experiments of modern methods.

> … depending on choice of downstream polarity classifier, we can achieve highly statistically significant improvement (from 82.8% to 86.4%)

— A Sentimental Education: [Sentiment Analysis Using Subjectivity Summarization Based on Minimum Cuts, 2004.][1]
[1]: http://xxx.lanl.gov/abs/cs/0409058

You can download the dataset from here:

* [Movie Review Polarity Dataset][2] (review_polarity.tar.gz, 3MB)
[2]: http://www.cs.cornell.edu/people/pabo/movie-review-data/review_polarity.tar.gz

After unzipping the file, you will have a directory called “txt_sentoken” with two sub-directories containing the text “neg” and “pos” for negative and positive reviews. Reviews are stored one per file with a naming convention cv000 to cv999 for each neg and pos.

Next, let’s look at loading and preparing the text data.