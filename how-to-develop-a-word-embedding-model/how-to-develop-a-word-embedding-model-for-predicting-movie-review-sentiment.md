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


### Loading and Cleaning Reviews

The text data is already pretty clean; not much preparation is required.

Without getting bogged down too much in the details, we will prepare the data using the following way:

* Split tokens on white space.
* Remove all punctuation from words.
* Remove all words that are not purely comprised of alphabetical characters.
* Remove all words that are known stop words.
* Remove all words that have a length <= 1 character.

We can put all of these steps into a function called clean_doc() that takes as an argument the raw text loaded from a file and returns a list of cleaned tokens. We can also define a function load_doc() that loads a document from file ready for use with the clean_doc() function.

An example of cleaning the first positive review is listed below.

* 文本数据已经相当干净；不需要更多预处理。
* 避免过于深入细节，我们将按照以下方式准备数据：
    * 按空格符划分。
    * 去除标点。
    * 去除不是纯英文字母组成的单词。
    * 去除停用词。
    * 去除长度小于等于1个字母的单词。
* 定义clean_doc()函数，包含以上步骤；输入文件中读取的纯文本，输出清理后的**token**列表。
* 定义load_doc()函数，读取文本文件，作为clean_doc()函数的输入。
* 下面是个清理第一个正例文件的例子。

```python
from nltk.corpus import stopwords
import string

# load doc into memory
def load_doc(filename):
    # open the file as read only
    file = open(filename, 'r')
    # read all text
    text = file.read()
    # close the file
    file.close()
    return text

# turn a doc into clean tokens
def clean_doc(doc):
    # split into tokens by white space
    tokens = doc.split()
    # remove punctuation from each token
    table = str.maketrans('', '', string.punctuation)
    tokens = [w.translate(table) for w in tokens]
    # remove remaining tokens that are not alphabetic
    tokens = [word for word in tokens if word.isalpha()]
    # filter out stop words
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if not w in stop_words]
    # filter out short tokens
    tokens = [word for word in tokens if len(word) > 1]
    return tokens

# load the document
filename = 'txt_sentoken/pos/cv000_29590.txt'
text = load_doc(filename)
tokens = clean_doc(text)
print(tokens)
```

Running the example prints a long list of clean tokens.

There are many more cleaning steps we may want to explore and I leave them as further exercises.

I’d love to see what you can come up with.
Post your approaches and findings in the comments at the end.

* 运行示例脚本，打印一个长长的**clean token**列表。
* 更多的数据清洗步骤留到之后的实践中。
* 尝试并分享你的结果。

```python
...
'creepy', 'place', 'even', 'acting', 'hell', 'solid', 'dreamy', 'depp', 'turning', 'typically', 'strong', 'performance', 'deftly', 'handling', 'british', 'accent', 'ians', 'holm', 'joe', 'goulds', 'secret', 'richardson', 'dalmatians', 'log', 'great', 'supporting', 'roles', 'big', 'surprise', 'graham', 'cringed', 'first', 'time', 'opened', 'mouth', 'imagining', 'attempt', 'irish', 'accent', 'actually', 'wasnt', 'half', 'bad', 'film', 'however', 'good', 'strong', 'violencegore', 'sexuality', 'language', 'drug', 'content']
```

### 定义词汇表 Define a Vocabulary

It is important to define a vocabulary of known words when using a bag-of-words or embedding model.

* 在使用BOW或者词嵌入模型时，必须定义包含全部已知词汇的词汇表。

The more words, the larger the representation of documents, therefore it is important to constrain the words to only those believed to be predictive. This is difficult to know beforehand and often it is important to test different hypotheses about how to construct a useful vocabulary.

* 词汇数量越多，文档的向量表示越大；因此，需要将限制词汇的数量，仅保留有预测性的词汇。
* 很难事先知晓哪些词汇是有预测性的，通常需要尝试不同的假设来构造有用的词汇表。

We have already seen how we can remove punctuation and numbers from the vocabulary in the previous section. We can repeat this for all documents and build a set of all known words.

* 上一节中，我们已经知晓如何去除词汇中的标点和数字。
* 将该方法应用于全部文档，就可以构造一个包含所有已知词汇的词汇表。

We can develop a vocabulary as a Counter, which is a dictionary mapping of words and their counts that allow us to easily update and query.

* 可以用一个**Counter对象**来实现词汇表，便于更新和查询。

Each document can be added to the counter (a new function called add_doc_to_vocab()) and we can step over all of the reviews in the negative directory and then the positive directory (a new function called process_docs()).

* 每个文档都被加入这个计数器（函数**add_doc_to_vocab()**）；
* 可以遍历所有反例路径下和正例路径下的所有影评（函数**process_docs()**）。

The complete example is listed below.

* 如下列例子。

```python
from string import punctuation
from os import listdir
from collections import Counter
from nltk.corpus import stopwords
 
# load doc into memory
def load_doc(filename):
    # open the file as read only
    file = open(filename, 'r')
    # read all text
    text = file.read()
    # close the file
    file.close()
    return text
 
# turn a doc into clean tokens
def clean_doc(doc):
    # split into tokens by white space
    tokens = doc.split()
    # remove punctuation from each token
    table = str.maketrans('', '', punctuation)
    tokens = [w.translate(table) for w in tokens]
    # remove remaining tokens that are not alphabetic
    tokens = [word for word in tokens if word.isalpha()]
    # filter out stop words
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if not w in stop_words]
    # filter out short tokens
    tokens = [word for word in tokens if len(word) > 1]
    return tokens
 
# load doc and add to vocab
def add_doc_to_vocab(filename, vocab):
    # load doc
    doc = load_doc(filename)
    # clean doc
    tokens = clean_doc(doc)
    # update counts
    vocab.update(tokens)
 
# load all docs in a directory
def process_docs(directory, vocab, is_trian):
    # walk through all files in the folder
    for filename in listdir(directory):
        # skip any reviews in the test set
        if is_trian and filename.startswith('cv9'):
            continue
        if not is_trian and not filename.startswith('cv9'):
            continue
        # create the full path of the file to open
        path = directory + '/' + filename
        # add doc to vocab
        add_doc_to_vocab(path, vocab)
 
# define vocab
vocab = Counter()
# add all docs to vocab
process_docs('txt_sentoken/neg', vocab, True)
process_docs('txt_sentoken/pos', vocab, True)
# print the size of the vocab
print(len(vocab))
# print the top words in the vocab
print(vocab.most_common(50))
```

Running the example shows that we have a vocabulary of 44,276 words.
We also can see a sample of the top 50 most used words in the movie reviews.
Note, that this vocabulary was constructed based on only those reviews in the training dataset.

* 运行以上示例，可得一个包含44,276个单词的词汇表。
* 也可查看影评中最高频的50个单词。
* 注意，此词汇表仅仅基于训练集中的文档构造。

```
44276
[('film', 7983), ('one', 4946), ('movie', 4826), ('like', 3201), ('even', 2262), ('good', 2080), ('time', 2041), ('story', 1907), ('films', 1873), ('would', 1844), ('much', 1824), ('also', 1757), ('characters', 1735), ('get', 1724), ('character', 1703), ('two', 1643), ('first', 1588), ('see', 1557), ('way', 1515), ('well', 1511), ('make', 1418), ('really', 1407), ('little', 1351), ('life', 1334), ('plot', 1288), ('people', 1269), ('could', 1248), ('bad', 1248), ('scene', 1241), ('movies', 1238), ('never', 1201), ('best', 1179), ('new', 1140), ('scenes', 1135), ('man', 1131), ('many', 1130), ('doesnt', 1118), ('know', 1092), ('dont', 1086), ('hes', 1024), ('great', 1014), ('another', 992), ('action', 985), ('love', 977), ('us', 967), ('go', 952), ('director', 948), ('end', 946), ('something', 945), ('still', 936)]
```

We can step through the vocabulary and remove all words that have a low occurrence, such as only being used once or twice in all reviews.
For example, the following snippet will retrieve only the tokens that of appears 2 or more times in all reviews.

* 可以删除所有低频词汇，比如只出现过一两次的。
* 下面的代码片段可以查询所有出现狼次或两次以上的词汇。


```python
# keep tokens with a min occurrence
min_occurane = 2
tokens = [k for k,c in vocab.items() if c >= min_occurane]
print(len(tokens))
```

Running the above example with this addition shows that the vocabulary size drops by a little more than half its size from 44,276 to 25,767 words.

* 运行以上示例，可见词汇表的大小缩减为略多于原本的一半，从44,276个词汇到25,767个。

```
25767
```

Finally, the vocabulary can be saved to a new file called vocab.txt that we can later load and use to filter movie reviews prior to encoding them for modeling. We define a new function called save_list() that saves the vocabulary to file, with one word per file.
For example:

* 最后，将词汇表保存到一个新文件**vocab.txt**，以便之后用来过滤文本或编码。
* 定义函数**save_list()**来保存词汇表。

```python
# save list to file
def save_list(lines, filename):
    # convert lines to a single blob of text
    data = '\n'.join(lines)
    # open file
    file = open(filename, 'w')
    # write text
    file.write(data)
    # close file
    file.close()

# save tokens to a vocabulary file
save_list(tokens, 'vocab.txt')
```

Running the min occurrence filter on the vocabulary and saving it to file, you should now have a new file called vocab.txt with only the words we are interested in.
The order of words in your file will differ, but should look something like the following:

* 对词汇表进行低频过滤，并保存结果到文件；
* 过滤后的文档保存于新文本文档**vocab.txt**；
* 也许词汇的顺序不同，但是这个文件看起来是这样：


```
aberdeen
dupe
burt
libido
hamlet
arlene
available
corners
web
columbia
...
```

We are now ready to look at learning features from the reviews.
* 我们现在准备好查看影评文档的特征了。

