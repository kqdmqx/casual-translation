原文
> https://pan.baidu.com/s/1pL5d7uV

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
