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


* 数据集已经被使用一些方法清洗，比如：
    * 数据集仅由英文影评组成。
    * 全部文本被转化为小写。
    * 在标点符号前后由空格隔开。
    * 文本被切分为每行一句。

The data has been used for a few related natural language processing tasks. For classification, the performance of machine learning models (such as Support Vector Machines) on the data is in the range of high 70% to low 80% (e.g. 78%-82%).

* 此数据被用于一些相关的自然语言处理任务。
* 在分类任务中，分类器的性能为78%-82%。（准确率？）

More sophisticated data preparation may see results as high as 86% with 10-fold cross validation. This gives us a ballpark of low-to-mid 80s if we were looking to use this dataset in experiments of modern methods.

> … depending on choice of downstream polarity classifier, we can achieve highly statistically significant improvement (from 82.8% to 86.4%)

— A Sentimental Education: [Sentiment Analysis Using Subjectivity Summarization Based on Minimum Cuts, 2004.][1]
[1]: http://xxx.lanl.gov/abs/cs/0409058

You can download the dataset from here:

* [Movie Review Polarity Dataset][2] (review_polarity.tar.gz, 3MB)
[2]: http://www.cs.cornell.edu/people/pabo/movie-review-data/review_polarity.tar.gz

* 更细致的数据准备后，可以在10折交叉验证中得到86%。（准确率？）
* ？

After unzipping the file, you will have a directory called “txt_sentoken” with two sub-directories containing the text “neg” and “pos” for negative and positive reviews. Reviews are stored one per file with a naming convention cv000 to cv999 for each neg and pos.

Next, let’s look at loading and preparing the text data.

* 解压文件后，可以看到一个目录**txt_sentoken**。
* 其中有两个目录**neg**、**pos**，分别包含影评的正例负例。
* 每个正例或负例样本一个文件，文件名从cv000到cv999。
* 接了下来，加载和准备这些文本文件。

## Data Preparation

In this section, we will look at 3 things:

1. Separation of data into training and test sets.
2. Loading and cleaning the data to remove punctuation and numbers.
3. Defining a vocabulary of preferred words.

* 本章做三件事：
    1. 把数据集划分为训练集和测试集。
    2. 加载并清洗数据，去除符号和数字。
    3. 定义**preferred words**的词汇表。

### Split into Train and Test Sets

We are pretending that we are developing a system that can predict the sentiment of a textual movie review as either positive or negative.

* 我们假设正在开发一个可以识别一段影评文本中的情感是正面或是负面。

This means that after the model is developed, we will need to make predictions on new textual reviews. This will require all of the same data preparation to be performed on those new reviews as is performed on the training data for the model.

* 这意味着系统完工后会被用来识别新的影评文本。
* 这所有在训练数据上采用的数据准备工作要同样应用在新数据上。

We will ensure that this constraint is built into the evaluation of our models by splitting the training and test datasets prior to any data preparation. This means that any knowledge in the data in the test set that could help us better prepare the data (e.g. the words used) are unavailable in the preparation of data used for training the model.

* 我们确保这项限制是模型验证的一部分。
* 通过在数据预处理之前把数据集划分为训练集和测试集。
* 在训练模型的数据准备工作中，不会用到测试集中的知识。

That being said, we will use the last 100 positive reviews and the last 100 negative reviews as a test set (100 reviews) and the remaining 1,800 reviews as the training dataset.

This is a 90% train, 10% split of the data.

The split can be imposed easily by using the filenames of the reviews where reviews named 000 to 899 are for training data and reviews named 900 onwards are for test.

* 使用最后100个正面评价，100个负面评价作为测试集；剩下1800个评价作为训练集。
* 数据划分是90%训练10%测试。
* 此划分可以简单的通过文件名实现，000到899的文件是训练数据900之后是测试集。