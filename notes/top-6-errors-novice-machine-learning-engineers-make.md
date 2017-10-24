原文
> https://pan.baidu.com/s/1pL5d7uV

作者
> Christopher Dossman

# Top 6 errors novice machine learning engineers make

* 机器学习工程师新手常犯的六大错误

In machine learning, there are many ways to build a product or solution and each way assumes something different.
Many times, it’s not obvious how to navigate and identify which assumptions are reasonable.
People new to machine learning make mistakes, which in hindsight will often feel silly.
I’ve created a list of the top mistakes that novice machine learning engineers make. Hopefully, you can learn from these common errors and create more robust solutions that bring real value.

* 机器学习中解决同一个问题有多种方法；不同的方法基于不同的假设。
* 很多时候，选择假设、辨别假设是否合理不是件容易事。
* 新手经常犯一些事后想来比较傻的错误。
* 作者列举了新手机器学习工程师的常见错误。

## Taking the default loss function for granted

* 想当然的使用默认的损失函数

Mean squared error is great!
It really is an amazing default to start off with, but when it comes to real-world applications this off-the-shelf loss function is rarely optimum for the business problem you’re trying to solve for.

* MSE很棒！
* MSE适用于起步；但是在实际应用问题中，这个损失函数常常不是最好的选择。


Take for example fraud detection.
In order to align with business objectives what you really want is to penalize false negatives in proportion to the dollar amount lost due to fraud.
Using mean squared error might give you OK results but will never give you state of the art results.

* 以欺诈检测为例。
* 为了和真正的业务目标(降低因欺诈损失中的漏报比例)一致。
* 使用MSE可以得到**OK**的结果，却无法得到**state-of-the-art**的结果。

**Take Away**: Always build a custom loss function that closely matches your solution objectives

* 总是建立符合问题目标的损失函数？

## Using one algorithm/method for all problems

Many will finish their first tutorial and immediately start using the same algorithm that they learned on every use case they can imagine.
It’s familiar and they figure it will work just as well as any other algorithm.
This is a bad assumption and will lead to poor results.
Let your data choose your model for you.
Once you have preprocessed your data, feed it into many different models and see what the results are.
You will have a good idea of what models work best and what models don’t work so well.

* 很多人在完成了第一个tutorial后，立即将其中同样算法应用于他们能想到的每一个场景。
* 这个算法他们比较熟悉，且他们认为这个算法和其他算法会工作得同样好。
* 这是个不好的假设，且会导致糟糕的效果。
* 让数据为你选择模型。
* 一旦完成数据预处理，将其注入不同的模型，观察其结果。
* 这样就可以得知哪些模型效果好，哪些效果不好。


**Becoming a Machine Learning Engineer | Step 2: Pick a process**
> https://medium.com/towards-data-science/becoming-a-machine-learning-engineer-step-2-pick-a-process-942eef6ba8dd

Check out this article and get a handle on your process.

**Take Away**: If you find yourself using the same algorithm over and over again it probably means you’re not getting the best results.

* 若发现你在反复使用同样的算法，也许这意味着你没有得到最好的效果。

## Forget about outliers


Outliers can be important or completely ignored, just based on context.
Take for example revenue forecasting.
Large spikes in revenue can occur and it is a good idea to look at them and understand why they occurred.
In the case of outliers caused by some type of error, it is safe to ignore them and remove from your data.
From a model perspective, some are more sensitive to outliers than other.
Take for example Adaboost, it treats those outliers as **“hard”** cases and puts tremendous weights on outliers while decision trees
might simple count each outlier as one false classification.

* 根据实际情况，离群点可能提供了重要信息，也可能完全可以忽略。
* 以收入预测为例。
* 收入值可能出现巨大的波动，查看这些波动并理解它们出现的原因是个好主意。
* 如果这些离群点因某种错误出现，那么可以安全地忽略它们并将它们从数据中删除。
* 从模型的角度看来，某些模型对离群点格外敏感。
* 以**Adaboost**为例，它把离群点作为**难以学习的样例**，且在决策树种将每个离群点作为错误的分类给予其巨大的权重。


Becoming a Machine Learning Engineer | Step 2: Pick a process
Goes over best practices that you can use to avoid this mistake

**Take Away**: Always look at your data closely before you start your work and determine if outliers should be ignored or looked at more closely
* 在其他工作开始前，仔细查看数据；决定是否要忽略其中的离群点，或进一步分析。


## Not properly dealing with cyclical features

Hours of the day, days of the week, months in a year, and wind direction are all examples of features that are cyclical.
Many new machine learning engineers don’t think to convert these features into a representation that can preserve information such as hour 23 and hour 0 being close to each other and not far.

* 小时之于一天、星期几之于一周、月份之于一年、风向，这些都是周期性特征。
* 很多新手工程师不知道需要把这些特征转化为适当的形式，以保持类似**23点和0点很接近而不是远离**这样的信息。


Keeping with the hour example, the best way to handle this is to calculate the sin and cos component so that you represent your cyclical feature as (x,y) coordinates of a circle.
In this representation hour, 23 and hour 0 are right next to each other numerically, just as they should be.

* 仍以小时为例，处理这种特征的最好方式是以**sin**和**cos**表示。
* 这种表示方式，23点和0点在数值上接近。

Take Away: If you have cyclical features and you are not converting them you are giving your model garbage data to start with.

* 若有周期特征存在，却不将其转化；则是给予模型垃圾数据。

## L1/L2 Regularization without standardization


L1 and L2 regularization penalizes large coefficients and is a common way to regularize linear or logistic regression; however, many machine learning engineers are not aware that is important to standardize features before applying regularization.

* 对大的参数使用L1和L2正则化是一种在线性回归和逻辑回归中常用的方法。
* 然而，很多机器学习工程师没有意识到应在正则化之前对数据归一化。


Assuming you had a linear regression model with a transaction amount as a feature.
Without regularization, if the transaction amount is in dollars, the fitted coefficients are going to be around 100 times larger
than if they were in cents.
This will cause a bias and tend to penalize features that are smaller in scale.
To avoid the problem, standardize all the features and put them on equal footing so regularization is the same all over your features.

* 假设线性回归模型中有一项特征-交易量。
* 若不使用正则化，交易量特征以美元为单位时，参数的数值将是以美分为单位的100倍。
* 这将导致一种惩罚单位较小的特征的倾向。
* 为避免这种情况，将所有特征归一化，令所有特征数值范围相当；这样正则化对所有特征效果一致。


**Take Away**: Regularization is great but can cause headaches if you don’t have standardized features
* 正则化是有效的技术，若不与归一化同时使用，则令人头疼。

## Interpreting absolute value of coeicients from linear or logistic regression as feature importance


Many off-the-shelf linear regressors return p-values for each coefficient, many novice machine learning engineers believe that for linear models, the bigger the value of the coefficient, the more important the feature is.
This is hardly ever true as the scale of the variable changes the absolute value of the coefficient.
If the features are co-linear, coefficients can shift from one feature to the other.
The more features the data set has the more likely the features are co-linear and the less reliable simple interpretations of feature importance are.

* 很多现成的线性回归工具返回每一项参数的p-value。
* 很多新手机器学习工程师认为，这项系数越大，这个特征越重要。
* 因为参数的数值范围影响系数的绝对值，所以这个观点基本不正确。
* 若特征之间有线性相关性，则系数可能从一向特征转移到另一项。
* 数据集的特征数量越多，则特征间有线性相关性的可能性越大，简单地把系数作为特征重要性的理解就越不靠谱。


**Take Away**: Understanding what features are most important to a result is important, but don’t assume that you can look at the coefficients.They often don’t tell the whole story.

* 理解特征重要性很重要，但不可假设直接看这些（线性模型的）系数。
* 这些系数通常无法传达全面的信息（受到特征数值和特征间线性相关性等因素影响，而不是单纯反映单特征的信息量）。

## 小结

Doing a few projects and getting good results can feel like winning a million bucks.
You worked hard and you have the results to prove that you did a good job,
but just like with any other industry the devil is in the details and even fancy plots can hide bias and error.
This list is not meant to be exhaustive, but merely to cause the reader to think about all the small issues that might be hiding in your solution.
In order to achieve good results, it is important to follow your process and always double check that you are not making some common mistakes.

* 做一些项目、得到好的结果，会感觉不错。
* 付出努力，且有结果证明工作的价值。
* 然而，和其他很多产业同样，恶魔藏在细节中，看起来很棒的图表中可能藏着偏差和错误。
* 此列表并不旨在详尽无遗，仅是希望引起读者考虑“你的答案中可能隐藏着种种细小的问题”。
* 为了得到好的结果，反复检查实验过程，确认常见错误是否存在很重要。


If you found this article useful you will get a lot out of my **Becoming a machine learning engineer | Step 2: Picking a Process article.**
It helps you iron out a process that will allow you to catch more simple mistakes and get better results.
