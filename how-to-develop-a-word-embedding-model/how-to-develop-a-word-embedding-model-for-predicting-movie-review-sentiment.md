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



## 教程大纲 Tutorial Overview

This tutorial is divided into 5 parts; they are:

1. 影评数据集 (Movie Review Dataset)
2. 数据准备 (Data Preparation)
3. 训练嵌入层 (Train Embedding Layer)
4. 训练word2vec嵌入模型 (Train word2vec Embedding)
5. 使用预训练的嵌入模型 (Use Pre-trained Embedding)


## 影评数据集 Movie Review Dataset

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

## 数据预处理 Data Preparation

In this section, we will look at 3 things:

1. Separation of data into training and test sets.
2. Loading and cleaning the data to remove punctuation and numbers.
3. Defining a vocabulary of preferred words.

* 本章做三件事：
    1. 把数据集划分为训练集和测试集。
    2. 加载并清洗数据，去除符号和数字。
    3. 定义**preferred words**的词汇表。

### 划分训练测试集 Split into Train and Test Sets

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


### 加载并清洗影评数据 Loading and Cleaning Reviews

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

## 训练嵌入层 Train Embedding Layer

In this section, we will learn a word embedding while training a neural network on the classification problem.

* 本节，我们将在训练解决分类问题的神经网络中学习一个词嵌入层。

A word embedding is a way of representing text where each word in the vocabulary is represented by a real valued vector in a high-dimensional space.
The vectors are learned in such a way that words that have similar meanings will have similar representation in the vector space (close in the vector space).
This is a more expressive representation for text than more classical methods like bag-of-words, where relationships between words or tokens are ignored, or forced in bigram and trigram approaches.

* 词嵌入是一种表示文本的方法，每个词汇表中的单词都由一个高维空间中的实向量表示；
* 这些向量的性质，含义相似的词汇在高维空间中的距离接近；
* 这种表示形式对于分类模型来说比BOW表达能力更强；
* 因为在BOW表示形式中单词之间的关系被忽略，或者被强行用二元或三元语法形式表示。

The real valued vector representation for words can be learned while training the neural network. We can do this in the Keras deep learning library using the Embedding layer.
The first step is to load the vocabulary. We will use it to filter out words from movie reviews that we are not interested in.
If you have worked through the previous section, you should have a local file called ‘vocab.txt‘ with one word per line. We can load that file and build a vocabulary as a set for checking the validity of tokens.

* 这些单词的实向量表示可以用于训练神经网络。
* 我们可以用**Keras**库的[Embedding layer][keras-ebd-layer]来实现。
[keras-ebd-layer]: https://keras.io/layers/embeddings/
* 第一步是加载词汇表。词汇表将被用来过滤掉影评中不相关的词汇。
* 如果你执行了上一节的步骤，你应该拥有一个每行一个单词的本地文件**vocab.txt**。
* 我们在过滤词汇时可以把该文件加载为一个**set对象**。

```python
# load doc into memory
def load_doc(filename):
    # open the file as read only
    file = open(filename, 'r')
    # read all text
    text = file.read()
    # close the file
    file.close()
    return text
 
# load the vocabulary
vocab_filename = 'vocab.txt'
vocab = load_doc(vocab_filename)
vocab = vocab.split()
vocab = set(vocab)
```

Next, we need to load all of the training data movie reviews. For that we can adapt the process_docs() from the previous section to load the documents, clean them, and return them as a list of strings, with one document per string. We want each document to be a string for easy encoding as a sequence of integers later.
Cleaning the document involves splitting each review based on white space, removing punctuation, and then filtering out all tokens not in the vocabulary.
The updated clean_doc() function is listed below.

* 接下来，我们需要读取所有训练数据。
* 我们可以应用上节的**process_docs()**函数，读取、清洗、每个文档返回一个字符串。
* 稍后，我们希望每个字符串被编码为一个正整数序列。
* 数据清洗包括，根据空格切分文本，去除标点，过滤掉不在词汇表中的token。
* 更新后的**clean_doc()**函数如下。


```python
# turn a doc into clean tokens
def clean_doc(doc, vocab):
    # split into tokens by white space
    tokens = doc.split()
    # remove punctuation from each token
    table = str.maketrans('', '', punctuation)
    tokens = [w.translate(table) for w in tokens]
    # filter out tokens not in vocab
    tokens = [w for w in tokens if w in vocab]
    tokens = ' '.join(tokens)
    return tokens
```

The updated process_docs() can then call the clean_doc() for each document on the ‘pos‘ and ‘neg‘ directories that are in our training dataset.

* 更新后的**process_docs()**函数对训练集的每个样本调用**clean_doc()**。

```python
# load all docs in a directory
def process_docs(directory, vocab, is_trian):
    documents = list()
    # walk through all files in the folder
    for filename in listdir(directory):
        # skip any reviews in the test set
        if is_trian and filename.startswith('cv9'):
            continue
        if not is_trian and not filename.startswith('cv9'):
            continue
        # create the full path of the file to open
        path = directory + '/' + filename
        # load the doc
        doc = load_doc(path)
        # clean doc
        tokens = clean_doc(doc, vocab)
        # add to list
        documents.append(tokens)
    return documents

# load all training reviews
positive_docs = process_docs('txt_sentoken/pos', vocab, True)
negative_docs = process_docs('txt_sentoken/neg', vocab, True)
train_docs = negative_docs + positive_docs
```

The next step is to encode each document as a sequence of integers.
The Keras Embedding layer requires integer inputs where each integer maps to a single token that has a specific real-valued vector representation within the embedding. These vectors are random at the beginning of training, but during training become meaningful to the network.

* 下一步是把每个文档编码成正整数序列。
* **Keras Embeding layer**需要正整数作为输入，其中每个正整数对应一个token，每个正整数在嵌入空间中对应一个实数向量。
* 向量在训练开始时随机初始化，随着训练变得有意义。

We can encode the training documents as sequences of integers using the [Tokenizer][keras-tokenizer] class in the Keras API.
[keras-tokenizer]: https://keras.io/preprocessing/text/#tokenizer
First, we must construct an instance of the class then train it on all documents in the training dataset. In this case, it develops a vocabulary of all tokens in the training dataset and develops a consistent mapping from words in the vocabulary to unique integers. We could just as easily develop this mapping ourselves using our vocabulary file.

* 我们可以用Keras提供的API-**Tokenizer类**来把文本编码为正整数序列。
* 首先，我们需要构造**Tokenizer类**的一个实例，并且用所有训练数据集进行训练。
* 在这种情况下，它构造了一个包含所有训练集数据的词汇表，且构造了一个从词汇表中的单词到不同的正整数的连续映射。
* 我们也可以通过词汇表文件自行构造此映射。

```python
# create the tokenizer
tokenizer = Tokenizer()
# fit the tokenizer on the documents
tokenizer.fit_on_texts(train_docs)
```

Now that the mapping of words to integers has been prepared, we can use it to encode the reviews in the training dataset. We can do that by calling the texts_to_sequences() function on the Tokenizer.

* 已经准备好从单词到整数的映射，我们可以用来编码训练集。
* 调用**tokenizer.texts_to_sequences()**函数来编码训练集。

```python
# sequence encode
encoded_docs = tokenizer.texts_to_sequences(train_docs)
```

We also need to ensure that all documents have the same length.
This is a requirement of Keras for efficient computation.
We could truncate reviews to the smallest size or zero-pad (pad with the value ‘0’) reviews to the maximum length, or some hybrid.
In this case, we will pad all reviews to the length of the longest review in the training dataset.
First, we can find the longest review using the max() function on the training dataset and take its length.
We can then call the Keras function pad_sequences() to pad the sequences to the maximum length by adding 0 values on the end.

* 我们还需要确保所有的文档长度一致。这是出于Keras计算效率的需要。
* 我们可以，按最短长度把所有文档截断，或按最长长度把所有文档补零，或两者混用。
* 在这种情况下，我们按最长长度把所有文档补零。
* 首先，我们在训练集上用**max()**函数找到长度最长的影评。
* 然后，我们调用keras函数**pad_sequences()**把所有文档按最长长度补零。

```python
# pad sequences
max_length = max([len(s.split()) for s in train_docs])
Xtrain = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
```

Finally, we can define the class labels for the training dataset, needed to fit the supervised neural network model to predict the sentiment of reviews.

* 最后，我们定义训练集的类标；用于训练监督神经网络模型来预测影评的情感倾向。

```python
# define training labels
ytrain = array([0 for _ in range(900)] + [1 for _ in range(900)])
```

We can then encode and pad the test dataset, needed later to evaluate the model after we train it.

* 接着，我们可以编码并补全测试集（验证集）；稍后用来验证训练好的模型。

```python
# load all test reviews
positive_docs = process_docs('txt_sentoken/pos', vocab, False)
negative_docs = process_docs('txt_sentoken/neg', vocab, False)
test_docs = negative_docs + positive_docs
# sequence encode
encoded_docs = tokenizer.texts_to_sequences(test_docs)
# pad sequences
Xtest = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
# define test labels
ytest = array([0 for _ in range(100)] + [1 for _ in range(100)])
```


We are now ready to define our neural network model.

The model will use an Embedding layer as the first hidden layer. The Embedding requires the specification of the vocabulary size, the size of the real-valued vector space, and the maximum length of input documents.

The vocabulary size is the total number of words in our vocabulary, plus one for unknown words. This could be the vocab set length or the size of the vocab within the tokenizer used to integer encode the documents, for example:

* 现在，我们准备好定义神经网络模型了。
* 模型将把嵌入层作为第一个隐藏层。
* 嵌入层需要定义，词汇表长度，实值向量空间长度，以及输入文档的最大长度。
* 词汇表长度是词汇表中所有单词的数量加一；加一用于处理未知词汇。
* 计算方式如下：

```python
# define vocabulary size (largest integer value)
vocab_size = len(tokenizer.word_index) + 1
```

We will use a 100-dimensional vector space, but you could try other values, such as 50 or 150. Finally, the maximum document length was calculated above in the max_length variable used during padding.

* 我们将使用100维的向量空间，但是你可以尝试其他数值，比如50或150。
* 最后，上面用来补零而计算的**max_length**即文档长度。

The complete model definition is listed below including the Embedding layer.

We use a Convolutional Neural Network (CNN) as they have proven to be successful at document classification problems. A conservative CNN configuration is used with 32 filters (parallel fields for processing words) and a kernel size of 8 with a rectified linear (‘relu’) activation function. This is followed by a pooling layer that reduces the output of the convolutional layer by half.

Next, the 2D output from the CNN part of the model is flattened to one long 2D vector to represent the ‘features’ extracted by the CNN. The back-end of the model is a standard Multilayer Perceptron layers to interpret the CNN features. The output layer uses a sigmoid activation function to output a value between 0 and 1 for the negative and positive sentiment in the review.

* 完整的模型定义如下，包括嵌入层。
* 我们使用被证实在文档分类问题中有效的CNN结构。
* 一个CNN层，32个filter，kernel-size=8，激活函数=relu。
* 接着，一个pooling层，把CNN层的输出维度减半。
* 接着，CNN层的2D输出被拉直为**1D**（原文写的2D），作为CNN层提取的特征。
* 模型的最后部分是一个全连接的前馈神经网络。
* 输出层的激活函数是sigmoid函数，用来输出01值。

```python
# define model
model = Sequential()
model.add(Embedding(vocab_size, 100, input_length=max_length))
model.add(Conv1D(filters=32, kernel_size=8, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
print(model.summary())
```

Running just this piece provides a summary of the defined network.
We can see that the Embedding layer expects documents with a length of 442 words as input and encodes each word in the document as a 100 element vector.

* 运行以上脚本片段，输出神经网络的结构。
* 我们可以看到嵌入层输入442个单词组成的文档，每个单词背被编码为一个100维的向量。

```
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
embedding_1 (Embedding)      (None, 442, 100)          2576800
_________________________________________________________________
conv1d_1 (Conv1D)            (None, 435, 32)           25632
_________________________________________________________________
max_pooling1d_1 (MaxPooling1 (None, 217, 32)           0
_________________________________________________________________
flatten_1 (Flatten)          (None, 6944)              0
_________________________________________________________________
dense_1 (Dense)              (None, 10)                69450
_________________________________________________________________
dense_2 (Dense)              (None, 1)                 11
=================================================================
Total params: 2,671,893
Trainable params: 2,671,893
Non-trainable params: 0
_________________________________________________________________
```


Next, we fit the network on the training data.

* 接下来，用训练集训练神经网络。

We use a binary cross entropy loss function because the problem we are learning is a binary classification problem. The efficient Adam implementation of stochastic gradient descent is used and we keep track of accuracy in addition to loss during training. The model is trained for 10 epochs, or 10 passes through the training data.

The network configuration and training schedule were found with a little trial and error, but are by no means optimal for this problem. If you can get better results with a different configuration, let me know.

* 因为我们在处理二分类问题，所以用二值cross-entropy作为损失函数。
* 我们使用随机梯度下降的高效Adam实现，我们在训练过程中除了记录loss外，同时跟踪accuracy。
* 模型训练10个epoch，或遍历训练集10次。
* 网络结构和训练参数是通过一些尝试后确定的，但是没有办法做到最优化。
* 请分享你的更好的结果。

```python
# compile network
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit network
model.fit(Xtrain, ytrain, epochs=10, verbose=2)
```

After the model is fit, it is evaluated on the test dataset. This dataset contains words that we have not seen before and reviews not seen during training.

* 模型训练之后，用验证集验证。
* 验证集包含训练集中未见的单词。

```python
# evaluate
loss, acc = model.evaluate(Xtest, ytest, verbose=0)
print('Test Accuracy: %f' % (acc*100))
```

We can tie all of this together.
The complete code listing is provided below.

* 我们组合以上的代码片段。完整的脚本如下。

```python
from string import punctuation
from os import listdir
from numpy import array
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Embedding
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
 
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
def clean_doc(doc, vocab):
    # split into tokens by white space
    tokens = doc.split()
    # remove punctuation from each token
    table = str.maketrans('', '', punctuation)
    tokens = [w.translate(table) for w in tokens]
    # filter out tokens not in vocab
    tokens = [w for w in tokens if w in vocab]
    tokens = ' '.join(tokens)
    return tokens
 
# load all docs in a directory
def process_docs(directory, vocab, is_trian):
    documents = list()
    # walk through all files in the folder
    for filename in listdir(directory):
        # skip any reviews in the test set
        if is_trian and filename.startswith('cv9'):
            continue
        if not is_trian and not filename.startswith('cv9'):
            continue
        # create the full path of the file to open
        path = directory + '/' + filename
        # load the doc
        doc = load_doc(path)
        # clean doc
        tokens = clean_doc(doc, vocab)
        # add to list
        documents.append(tokens)
    return documents
 
# load the vocabulary
vocab_filename = 'vocab.txt'
vocab = load_doc(vocab_filename)
vocab = vocab.split()
vocab = set(vocab)
 
# load all training reviews
positive_docs = process_docs('txt_sentoken/pos', vocab, True)
negative_docs = process_docs('txt_sentoken/neg', vocab, True)
train_docs = negative_docs + positive_docs
 
# create the tokenizer
tokenizer = Tokenizer()
# fit the tokenizer on the documents
tokenizer.fit_on_texts(train_docs)
 
# sequence encode
encoded_docs = tokenizer.texts_to_sequences(train_docs)
# pad sequences
max_length = max([len(s.split()) for s in train_docs])
Xtrain = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
# define training labels
ytrain = array([0 for _ in range(900)] + [1 for _ in range(900)])
 
# load all test reviews
positive_docs = process_docs('txt_sentoken/pos', vocab, False)
negative_docs = process_docs('txt_sentoken/neg', vocab, False)
test_docs = negative_docs + positive_docs
# sequence encode
encoded_docs = tokenizer.texts_to_sequences(test_docs)
# pad sequences
Xtest = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
# define test labels
ytest = array([0 for _ in range(100)] + [1 for _ in range(100)])
 
# define vocabulary size (largest integer value)
vocab_size = len(tokenizer.word_index) + 1
 
# define model
model = Sequential()
model.add(Embedding(vocab_size, 100, input_length=max_length))
model.add(Conv1D(filters=32, kernel_size=8, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
print(model.summary())

# compile network
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit network
model.fit(Xtrain, ytrain, epochs=10, verbose=2)
# evaluate
loss, acc = model.evaluate(Xtest, ytest, verbose=0)
print('Test Accuracy: %f' % (acc*100))
```

Running the example prints the loss and accuracy at the end of each training epoch. We can see that the model very quickly achieves 100% accuracy on the training dataset.

At the end of the run, the model achieves an accuracy of 84.5% on the test dataset, which is a great score.

Given the stochastic nature of neural networks, your specific results will vary. Consider running the example a few times and taking the average score as the skill of the model.

* 运行样例脚本，打印每轮迭代后的训练loss和accuracy。
* 可以看到，模型很快在训练集上达到100%的accuracy。
* 在脚本运行结束后，测试集的acc达到84.5%，这个评分很高。
* 考虑到神经网络的随机性，你得到的运行结果可能略有不同。
* 考虑多次运行脚本，取评分的平均来作为模型的性能评价。

```
Epoch 6/10
2s - loss: 0.0013 - acc: 1.0000
Epoch 7/10
2s - loss: 8.4573e-04 - acc: 1.0000
Epoch 8/10
2s - loss: 5.8323e-04 - acc: 1.0000
Epoch 9/10
2s - loss: 4.3155e-04 - acc: 1.0000
Epoch 10/10
2s - loss: 3.3083e-04 - acc: 1.0000
Test Accuracy: 84.500000
```

We have just seen an example of how we can learn a word embedding as part of fitting a neural network model.

Next, let’s look at how we can efficiently learn a standalone embedding that we could later use in our neural network.

* 我们刚刚见到了如何把词嵌入作为神经网络的一部分。
* 接下来，我们看如何有效的独立训练一个可以稍后在神经网络中使用的嵌入层。

## 训练word2vec嵌入 Train word2vec Embedding

In this section, we will discover how to learn a standalone word embedding using an efficient algorithm called word2vec.
A downside of learning a word embedding as part of the network is that it can be very slow, especially for very large text datasets.
The word2vec algorithm is an approach to learning a word embedding from a text corpus in a standalone way. The benefit of the method is that it can produce high-quality word embeddings very efficiently, in terms of space and time complexity.

* 本节，我们将知晓如何用一种叫做**word2vec**的有效算法训练一个独立的词嵌入模型。
* 把词嵌入作为更大的神经网络的一部分进行训练的一个缺陷是训练速度很慢；特别是文本数据集很大的时候。
* **word2vec**是一种从文本数据集独立训练词嵌入的算法。
* 此算法的好处是从时间复杂度和空间复杂度的角度考虑，都能提供一个高效且高质量的词嵌入模型。

The first step is to prepare the documents ready for learning the embedding.
This involves the same data cleaning steps from the previous section, namely splitting documents by their white space, removing punctuation, and filtering out tokens not in the vocabulary.
The word2vec algorithm processes documents sentence by sentence. This means we will preserve the sentence-based structure during cleaning.
We start by loading the vocabulary, as before.

* 第一步是为训练嵌入模型准备文档数据。
* 这包括和上一节一样的数据清洗步骤，即按空格分词，去除标点，过滤掉不在词汇表中的token。
* **word2vec**算法逐句处理文档。这意味着在清洗过程中需要保留基于句子的结构。
* 同之前一样，我们从加载词汇表开始。

```python
# load doc into memory
def load_doc(filename):
    # open the file as read only
    file = open(filename, 'r')
    # read all text
    text = file.read()
    # close the file
    file.close()
    return text

# load the vocabulary
vocab_filename = 'vocab.txt'
vocab = load_doc(vocab_filename)
vocab = vocab.split()
vocab = set(vocab)
```

Next, we define a function named doc_to_clean_lines() to clean a loaded document line by line and return a list of the cleaned lines.

* 接下来，我们定义**doc_to_clean_lines()**函数来逐行清洗文档，并返回由清洗后的行组成的列表。

```python
# turn a doc into clean tokens
def doc_to_clean_lines(doc, vocab):
    clean_lines = list()
    lines = doc.splitlines()
    for line in lines:
        # split into tokens by white space
        tokens = line.split()
        # remove punctuation from each token
        table = str.maketrans('', '', punctuation)
        tokens = [w.translate(table) for w in tokens]
        # filter out tokens not in vocab
        tokens = [w for w in tokens if w in vocab]
        clean_lines.append(tokens)
    return clean_lines
```

Next, we adapt the process_docs() function to load and clean all of the documents in a folder and return a list of all document lines.
The results from this function will be the training data for the word2vec model.

* 接下来，我们用**process_docs()**函数读取一个目录下所有文档，并返回全部文档中逐行组成的列表。
* 这个函数返回的结果将用于训练word2vec模型。

```python
# load all docs in a directory
def process_docs(directory, vocab, is_trian):
    lines = list()
    # walk through all files in the folder
    for filename in listdir(directory):
        # skip any reviews in the test set
        if is_trian and filename.startswith('cv9'):
            continue
        if not is_trian and not filename.startswith('cv9'):
            continue
        # create the full path of the file to open
        path = directory + '/' + filename
        # load and clean the doc
        doc = load_doc(path)
        doc_lines = doc_to_clean_lines(doc, vocab)
        # add lines to list
        lines += doc_lines
    return lines
```

We can then load all of the training data and convert it into a long list of ‘sentences’ (lists of tokens) ready for fitting the word2vec model.

* 我们现在可以读取所有训练集数据，并且把它变成一个由句子（list of tokens）组成的长列表，用来训练word2vec模型。

```python
# load training data
positive_lines = process_docs('txt_sentoken/pos', vocab, True)
negative_lines = process_docs('txt_sentoken/neg', vocab, True)
sentences = negative_docs + positive_docs
print('Total training sentences: %d' % len(sentences))
```

We will use the word2vec implementation provided in the Gensim Python library. Specifically the Word2Vec class.

The model is fit when constructing the class. We pass in the list of clean sentences from the training data, then specify the size of the embedding vector space (we use 100 again), the number of neighboring words to look at when learning how to embed each word in the training sentences (we use 5 neighbors), the number of threads to use when fitting the model (we use 8, but change this if you have more or less CPU cores), and the minimum occurrence count for words to consider in the vocabulary (we set this to 1 as we have already prepared the vocabulary).

After the model is fit, we print the size of the learned vocabulary, which should match the size of our vocabulary in vocab.txt of 25,767 tokens.

* 我们使用**Gensim**库提供的word2vec算法，即Word2Vec类。
* 模型在构造这个类（的实例）时训练。
* 我们把从训练数据读取的清洗后的句子列表传入；
    * 接着指定嵌入向量空间的大小（还用100维）；
    * 指定训练时查询的邻近词数量（5-neighbours）；
    * 训练时使用的线程数（8线程，根据CPU数量调整）；
    * 考虑在词汇表中的最小出现次数（设为1次，因为我们已经预处理了词汇表）。
* 在模型训练后，打印学到的词汇表的长度，这个长度应该和词汇表vocab.txt文件长度一致。


```python
# train word2vec model
model = Word2Vec(sentences, size=100, window=5, workers=8, min_count=1)
# summarize vocabulary size in model
words = list(model.wv.vocab)
print('Vocabulary size: %d' % len(words))
```

Finally, we save the learned embedding vectors to file using the save_word2vec_format() on the model’s ‘wv‘ (word vector) attribute. The embedding is saved in ASCII format with one word and vector per line.

The complete example is listed below.

* 最后，我们用**save_word2vec_format()**函数把学到的嵌入向量（model.wv属性，词向量）保存到文件。
* 嵌入使用ASCII编码，每行一个单词，一个向量。
* 完整的样例脚本如下。

```python
from string import punctuation
from os import listdir
from gensim.models import Word2Vec

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
def doc_to_clean_lines(doc, vocab):
    clean_lines = list()
    lines = doc.splitlines()
    for line in lines:
        # split into tokens by white space
        tokens = line.split()
        # remove punctuation from each token
        table = str.maketrans('', '', punctuation)
        tokens = [w.translate(table) for w in tokens]
        # filter out tokens not in vocab
        tokens = [w for w in tokens if w in vocab]
        clean_lines.append(tokens)
    return clean_lines

# load all docs in a directory
def process_docs(directory, vocab, is_trian):
    lines = list()
    # walk through all files in the folder
    for filename in listdir(directory):
        # skip any reviews in the test set
        if is_trian and filename.startswith('cv9'):
            continue
        if not is_trian and not filename.startswith('cv9'):
            continue
        # create the full path of the file to open
        path = directory + '/' + filename
        # load and clean the doc
        doc = load_doc(path)
        doc_lines = doc_to_clean_lines(doc, vocab)
        # add lines to list
        lines += doc_lines
    return lines

# load the vocabulary
vocab_filename = 'vocab.txt'
vocab = load_doc(vocab_filename)
vocab = vocab.split()
vocab = set(vocab)

# load training data
positive_lines = process_docs('txt_sentoken/pos', vocab, True)
negative_lines = process_docs('txt_sentoken/neg', vocab, True)
sentences = negative_docs + positive_docs
print('Total training sentences: %d' % len(sentences))

# train word2vec model
model = Word2Vec(sentences, size=100, window=5, workers=8, min_count=1)
# summarize vocabulary size in model
words = list(model.wv.vocab)
print('Vocabulary size: %d' % len(words))

# save model in ASCII (word2vec) format
filename = 'embedding_word2vec.txt'
model.wv.save_word2vec_format(filename, binary=False)
```

Running the example loads 58,109 sentences from the training data and creates an embedding for a vocabulary of 25,767 words.
You should now have a file ’embedding_word2vec.txt’ with the learned vectors in your current working directory.

* 运行样例从训练数据中读取58,109个句子，创建一个包含25,767个单词的嵌入。
* 现在你应该在当前工作路径下有一个叫做**embedding_word2vec.txt**的文件，里面保存了学到的向量。

```
Total training sentences: 58109
Vocabulary size: 25767
```

Next, let’s look at using these learned vectors in our model.

* 接下来，我们看看怎么把这些学到的嵌入向量应用到分类模型中。

## Use Pre-trained Embedding

In this section, we will use a pre-trained word embedding prepared on a very large text corpus.
We can use the pre-trained word embedding developed in the previous section and the CNN model developed in the section before that.

* 在此章节，我们使用一个预先训练好的词嵌入模型。
* 我们可以使用上一节预训练的词嵌入模型，上上节开发的CNN模型。

The first step is to load the word embedding as a directory of words to vectors.
The word embedding was saved in so-called ‘word2vec‘ format that contains a header line. We will skip this header line when loading the embedding.
The function below named load_embedding() loads the embedding and returns a directory of words mapped to the vectors in NumPy format.

* 第一步是读取词嵌入模型文件，构造一个从单词到词向量的dict。
* 此前保存的词嵌入模型有表头，读取时跳过表头。
* 下面的**load_embedding()**函数读取词嵌入模型，返回一个从字符串到numpy.ndarr的dict。


```python
# load embedding as a dict
def load_embedding(filename):
    # load embedding into memory, skip first line
    file = open(filename,'r')
    lines = file.readlines()[1:]
    file.close()
    # create a map of words to vectors
    embedding = dict()
    for line in lines:
        parts = line.split()
        # key is string word, value is numpy array for vector
        embedding[parts[0]] = asarray(parts[1:], dtype='float32')
    return embedding
```

Now that we have all of the vectors in memory, we can order them in such a way as to match the integer encoding prepared by the Keras Tokenizer.

Recall that we integer encode the review documents prior to passing them to the Embedding layer. The integer maps to the index of a specific vector in the embedding layer. Therefore, it is important that we lay the vectors out in the Embedding layer such that the encoded words map to the correct vector.

Below defines a function get_weight_matrix() that takes the loaded embedding and the tokenizer.word_index vocabulary as arguments and returns a matrix with the word vectors in the correct locations.

* 现在，我们读取全部的词向量到内存，我们可以把他们按照Keras Tokenizer编码的顺序排列。
* 记住我们在把文档传入嵌入层之前进行了编码。编码后得到的正整数对应嵌入层中的一个索引。
* 因此，把词向量在嵌入层正确的排列很重要，这样才能把编码后的单词映射到正确的向量。
* 下面定义了一个**get_weight_matrix()**函数，读取加载的词向量和**tokenizer.word_index**词汇表作为参数，返回由词向量组成的矩阵。


```python
# create a weight matrix for the Embedding layer from a loaded embedding
def get_weight_matrix(embedding, vocab):
    # total vocabulary size plus 0 for unknown words
    vocab_size = len(vocab) + 1
    # define weight matrix dimensions with all 0
    weight_matrix = zeros((vocab_size, 100))
    # step vocab, store vectors using the Tokenizer's integer mapping
    for word, i in vocab.items():
        weight_matrix[i] = embedding.get(word)
    return weight_matrix
```

Now we can use these functions to create our new Embedding layer for our model.

* 现在我们可以用这些函数来创建嵌入层了。


```python
# load embedding from file
raw_embedding = load_embedding('embedding_word2vec.txt')
# get vectors in the right order
embedding_vectors = get_weight_matrix(raw_embedding, tokenizer.word_index)
# create the embedding layer
embedding_layer = Embedding(vocab_size, 100, weights=[embedding_vectors], input_length=max_length, trainable=False)
```

Note that the prepared weight matrix embedding_vectors is passed to the new Embedding layer as an argument and that we set the ‘trainable‘ argument to ‘False‘ to ensure that the network does not try to adapt the pre-learned vectors as part of training the network.

We can now add this layer to our model. We also have a slightly different model configuration with a lot more filters (128) in the CNN model and a kernel that matches the 5 words used as neighbors when developing the word2vec embedding. Finally, the back-end of the model was simplified.

* 注意，预先准备的权重矩阵**embedding_vectors**被作为参数用来构造**Embedding**对象，同时我们把**trainable**参数设置为**False**，来确保神经网络在训练过程中不会尝试更新这些预先训练的权重。
* 现在我们可以把这一层加入模型中，同时我们对模型结构进行微调。
    * 使用了更多的filter(128);
    * 和word2vec中5-neighbours对应的kernel;
* 最后模型的back-end被简化为一层。

```python
# define model
model = Sequential()
model.add(embedding_layer)
model.add(Conv1D(filters=128, kernel_size=5, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))
print(model.summary())
```

These changes were found with a little trial and error.
The complete code listing is provided below.

* 以上这些模型结构参数通过少量的尝试决定。
* 完整的代码如下。

```python
from string import punctuation
from os import listdir
from numpy import array
from numpy import asarray
from numpy import zeros
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Embedding
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
 
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
def clean_doc(doc, vocab):
    # split into tokens by white space
    tokens = doc.split()
    # remove punctuation from each token
    table = str.maketrans('', '', punctuation)
    tokens = [w.translate(table) for w in tokens]
    # filter out tokens not in vocab
    tokens = [w for w in tokens if w in vocab]
    tokens = ' '.join(tokens)
    return tokens
 
# load all docs in a directory
def process_docs(directory, vocab, is_trian):
    documents = list()
    # walk through all files in the folder
    for filename in listdir(directory):
        # skip any reviews in the test set
        if is_trian and filename.startswith('cv9'):
            continue
        if not is_trian and not filename.startswith('cv9'):
            continue
        # create the full path of the file to open
        path = directory + '/' + filename
        # load the doc
        doc = load_doc(path)
        # clean doc
        tokens = clean_doc(doc, vocab)
        # add to list
        documents.append(tokens)
    return documents
 
# load embedding as a dict
def load_embedding(filename):
    # load embedding into memory, skip first line
    file = open(filename,'r')
    lines = file.readlines()[1:]
    file.close()
    # create a map of words to vectors
    embedding = dict()
    for line in lines:
        parts = line.split()
        # key is string word, value is numpy array for vector
        embedding[parts[0]] = asarray(parts[1:], dtype='float32')
    return embedding
 
# create a weight matrix for the Embedding layer from a loaded embedding
def get_weight_matrix(embedding, vocab):
    # total vocabulary size plus 0 for unknown words
    vocab_size = len(vocab) + 1
    # define weight matrix dimensions with all 0
    weight_matrix = zeros((vocab_size, 100))
    # step vocab, store vectors using the Tokenizer's integer mapping
    for word, i in vocab.items():
        weight_matrix[i] = embedding.get(word)
    return weight_matrix
 
# load the vocabulary
vocab_filename = 'vocab.txt'
vocab = load_doc(vocab_filename)
vocab = vocab.split()
vocab = set(vocab)
 
# load all training reviews
positive_docs = process_docs('txt_sentoken/pos', vocab, True)
negative_docs = process_docs('txt_sentoken/neg', vocab, True)
train_docs = negative_docs + positive_docs
 
# create the tokenizer
tokenizer = Tokenizer()
# fit the tokenizer on the documents
tokenizer.fit_on_texts(train_docs)
 
# sequence encode
encoded_docs = tokenizer.texts_to_sequences(train_docs)
# pad sequences
max_length = max([len(s.split()) for s in train_docs])
Xtrain = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
# define training labels
ytrain = array([0 for _ in range(900)] + [1 for _ in range(900)])
 
# load all test reviews
positive_docs = process_docs('txt_sentoken/pos', vocab, False)
negative_docs = process_docs('txt_sentoken/neg', vocab, False)
test_docs = negative_docs + positive_docs
# sequence encode
encoded_docs = tokenizer.texts_to_sequences(test_docs)
# pad sequences
Xtest = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
# define test labels
ytest = array([0 for _ in range(100)] + [1 for _ in range(100)])
 
# define vocabulary size (largest integer value)
vocab_size = len(tokenizer.word_index) + 1
 
# load embedding from file
raw_embedding = load_embedding('embedding_word2vec.txt')
# get vectors in the right order
embedding_vectors = get_weight_matrix(raw_embedding, tokenizer.word_index)
# create the embedding layer
embedding_layer = Embedding(vocab_size, 100, weights=[embedding_vectors], input_length=max_length, trainable=False)
 
# define model
model = Sequential()
model.add(embedding_layer)
model.add(Conv1D(filters=128, kernel_size=5, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))
print(model.summary())
# compile network
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit network
model.fit(Xtrain, ytrain, epochs=10, verbose=2)
# evaluate
loss, acc = model.evaluate(Xtest, ytest, verbose=0)
print('Test Accuracy: %f' % (acc*100))
```

Running the example shows that performance was not improved.
In fact, performance was a lot worse. The results show that the training dataset was learned successfully, but evaluation on the test dataset was very poor, at just above 50% accuracy.
The cause of the poor test performance may be because of the chosen word2vec configuration or the chosen neural network configuration.

* 运行样例后看到模型表现没有提升。
* 实际上，模型表现变得很差。
* 训练集拟合得还不错，但是验证集的表现及其悲惨，acc略高于50%。
* 这个结果有可能是word2vec的配置，或者是神经网络结构导致的。


```
Epoch 6/10
2s - loss: 0.3306 - acc: 0.8778
Epoch 7/10
2s - loss: 0.2888 - acc: 0.8917
Epoch 8/10
2s - loss: 0.1878 - acc: 0.9439
Epoch 9/10
2s - loss: 0.1255 - acc: 0.9750
Epoch 10/10
2s - loss: 0.0812 - acc: 0.9928
Test Accuracy: 53.000000
```

The weights in the embedding layer can be used as a starting point for the network, and adapted during the training of the network. We can do this by setting ‘trainable=True‘ (the default) in the creation of the embedding layer.
Repeating the experiment with this change shows slightly better results, but still poor.
I would encourage you to explore alternate configurations of the embedding and network to see if you can do better. Let me know how you do.

* 嵌入层的权重可以作为神经网络的初始值，然后再训练过程中更新。
* 我们通过在创建嵌入层时设置**trainable=True**来实现这一点。
* 重复试验，效果略有提升，但是依然悲惨。
* 请多多尝试。


```
Epoch 6/10
4s - loss: 0.0950 - acc: 0.9917
Epoch 7/10
4s - loss: 0.0355 - acc: 0.9983
Epoch 8/10
4s - loss: 0.0158 - acc: 1.0000
Epoch 9/10
4s - loss: 0.0080 - acc: 1.0000
Epoch 10/10
4s - loss: 0.0050 - acc: 1.0000
Test Accuracy: 57.500000
```

It is possible to use pre-trained word vectors prepared on very large corpora of text data.

For example, both Google and Stanford provide pre-trained word vectors that you can download, trained with the efficient word2vec and GloVe methods respectively.

Let’s try to use pre-trained vectors in our model.

* 可以使用在更大的数据集上预训练的词向量。

You can download [pre-trained GloVe vectors][pre-trained-glove-vectors] from the Stanford webpage. Specifically, vectors trained on Wikipedia data:
[pre-trained-glove-vectors]: https://nlp.stanford.edu/projects/glove/

* [glove.6B.zip][glove-6b] (822 Megabyte download)
[glove-6b]: http://nlp.stanford.edu/data/glove.6B.zip

Unzipping the file, you will find pre-trained embeddings for various different dimensions. We will load the 100 dimension version in the file ‘glove.6B.100d.txt‘

The Glove file does not contain a header file, so we do not need to skip the first line when loading the embedding into memory. The updated load_embedding() function is listed below.

* 下载并解压**glove.6B.zip**文件，...

```python
# load embedding as a dict
def load_embedding(filename):
    # load embedding into memory, skip first line
    file = open(filename,'r')
    lines = file.readlines()
    file.close()
    # create a map of words to vectors
    embedding = dict()
    for line in lines:
        parts = line.split()
        # key is string word, value is numpy array for vector
        embedding[parts[0]] = asarray(parts[1:], dtype='float32')
    return embedding
```

It is possible that the loaded embedding does not contain all of the words in our chosen vocabulary. As such, when creating the Embedding weight matrix, we need to skip words that do not have a corresponding vector in the loaded GloVe data. Below is the updated, more defensive version of the get_weight_matrix() function.

```python
# create a weight matrix for the Embedding layer from a loaded embedding
def get_weight_matrix(embedding, vocab):
    # total vocabulary size plus 0 for unknown words
    vocab_size = len(vocab) + 1
    # define weight matrix dimensions with all 0
    weight_matrix = zeros((vocab_size, 100))
    # step vocab, store vectors using the Tokenizer's integer mapping
    for word, i in vocab.items():
        vector = embedding.get(word)
        if vector is not None:
            weight_matrix[i] = vector
    return weight_matrix
```


We can now load the GloVe embedding and create the Embedding layer as before.

```python
# load embedding from file
raw_embedding = load_embedding('glove.6B.100d.txt')
# get vectors in the right order
embedding_vectors = get_weight_matrix(raw_embedding, tokenizer.word_index)
# create the embedding layer
embedding_layer = Embedding(vocab_size, 100, weights=[embedding_vectors], input_length=max_length, trainable=False)
```

We will use the same model as before.

The complete example is listed below.