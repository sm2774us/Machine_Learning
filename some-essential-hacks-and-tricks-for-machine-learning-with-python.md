# **Some Essential Hacks and Tricks for Machine Learning with Python**

## We describe some essential hacks and tricks for practicing machine learning with Python.

[](https://heartbeat.fritz.ai/@tirthajyoti?source=post_page-----5478bc6593f2----------------------)

![Tirthajyoti Sarkar](https://miro.medium.com/fit/c/48/48/1*dROuRoTytntKE6LLBKKzKA.jpeg)

[Tirthajyoti Sarkar](https://heartbeat.fritz.ai/@tirthajyoti?source=post_page-----5478bc6593f2----------------------)

Follow

[Apr 19, 2018](https://heartbeat.fritz.ai/some-essential-hacks-and-tricks-for-machine-learning-with-python-5478bc6593f2?source=post_page-----5478bc6593f2----------------------) · 10 min read

![](https://miro.medium.com/max/30/1*oSJ-W0XzUaOQNi_NbpFCbw.png?q=20)

![](https://miro.medium.com/max/2400/1*oSJ-W0XzUaOQNi_NbpFCbw.png)

# “I **_am a student of computer science/engineering. How do I get into the field of machine learning/deep learning/AI?”_**

It’s never been easier to get started with machine learning. In addition to structured MOOCs, there is also a huge number of incredible, free resources available around the web. Here are just a few that have helped me:

1.  Start with some cool videos on YouTube. Read a couple of good books or articles. For example, have you read “[The Master Algorithm: How the Quest for the Ultimate Learning Machine Will Remake Our World](https://www.goodreads.com/book/show/24612233-the-master-algorithm)”? And I can guarantee you’ll fall in love with [this cool interactive page about machine learning?](http://www.r2d3.us/visual-intro-to-machine-learning-part-1/)
2.  Learn to clearly differentiate between buzzwords first — _machine learning, artificial intelligence, deep learning, data science, computer vision, robotics_. Read or listen to the talks, given by experts, on each of them. [Watch this amazing video by Brandon Rohrer](https://www.youtube.com/watch?v=tKa0zDDDaQk), an influential data scientist. Or this [video about the clear definition and difference of various roles](https://www.youtube.com/watch?v=Ura_ioOcpQI) associated with data science.
3.  Have your goal clearly set for what you want to learn. And then, go and take that Coursera course. Or take the [other one from Univ. of Washington](https://www.coursera.org/specializations/machine-learning), which is pretty good too.
4.  1. **Follow some good blogs**: [KDnuggets](https://www.kdnuggets.com/), [Mark Meloon’s blog about data science career](http://www.markmeloon.com/), [Brandon Rohrer’s blog](https://brohrer.github.io/blog.html), [Open AI’s blog about their research](https://blog.openai.com/), and of course, [Heartbeat](http://heartbeat.fritz.ai/)
5.  If you are enthusiastic about taking online MOOCs, [check out this article for guidance](https://towardsdatascience.com/how-to-choose-effective-moocs-for-machine-learning-and-data-science-8681700ed83f).
6.  Most of all, develop a feel for it. Join some good social forums, but **resist the temptation to latch onto sensationalized headlines and news bytes** posted. Do your own reading, understand what it is and what it is not, where it might go, and what possibilities it can open up. Then sit back and think about how you can apply machine learning or imbue data science principles into your daily work. Build a simple [regression model](https://heartbeat.fritz.ai/5-regression-loss-functions-all-machine-learners-should-know-4fb140e9d4b0) to predict the cost of your next lunch or download your electricity usage data from your energy provider and do a simple time-series plot in Excel to discover some pattern of usage. And after you are thoroughly enamored with machine learning, you can watch this video.
https://www.youtube.com/watch?v=IpGxLWOIZy4

# Is Python a good language of choice for Machine Learning/AI?

Familiarity and moderate expertise in at least one high-level programming language is useful for beginners in machine learning. Unless you are a Ph.D. researcher working on a purely theoretical proof of some complex algorithm, you are expected to mostly use the existing machine learning algorithms and apply them in solving novel problems. This requires you to put on a programming hat.

There’s a lot of debate on the ‘_best language for data science_’ (in fact, here’s a take on why [data scientists should learn Swift][(https://heartbeat.fritz.ai/why-data-scientists-should-start-learning-swift-66c3643e0d0d)).

While the debate rages on, grab a coffee and [read this insightful article to get an idea and see your choices](https://medium.freecodecamp.org/which-languages-should-you-learn-for-data-science-e806ba55a81f). Or, check out [this post on KDnuggets](https://www.kdnuggets.com/2017/09/python-vs-r-data-science-machine-learning.html). For now, it’s widely believed that Python helps developers to be more productive from development to deployment and maintenance. Python’s syntax is simpler and of a higher level when compared to Java, C, and C++. It has a vibrant community, open-source culture, hundreds of high-quality libraries focused on machine learning, and a huge support base from big names in the industry (e.g. Google, Dropbox, Airbnb, etc.).

> **_This article will focus on some essential hacks and tricks in Python focused on machine learning_**.

# Fundamental Libraries to know and master

There are few core Python packages/libraries you need to master for practicing machine learning effectively. Very brief description of those are given below,

## Numpy

Short for [Numerical Python](http://numpy.org/), NumPy is the fundamental package required for high performance scientific computing and data analysis in the Python ecosystem. It’s the foundation on which nearly all of the higher-level tools such as [Pandas](https://pandas.pydata.org/) and [scikit-learn](http://scikit-learn.org/) are built. [TensorFlow](https://www.tensorflow.org/) uses NumPy arrays as the fundamental building block on top of which they built their Tensor objects and graphflow for deep learning tasks. Many NumPy operations are implemented in C, making them super fast. For data science and modern machine learning tasks, this is an invaluable advantage.

![](https://miro.medium.com/max/30/1*qvSMwAWOd4cfett-57FUHA.jpeg?q=20)

![](https://miro.medium.com/max/700/1*qvSMwAWOd4cfett-57FUHA.jpeg)

## Pandas

This is the most popular library in the scientific Python ecosystem for doing general-purpose data analysis. Pandas is built upon Numpy array thereby preserving the feature of fast execution speed and offering many **data engineering features** including:

-   Reading/writing many different data formats
-   Selecting subsets of data
-   Calculating across rows and down columns
-   Finding and filling missing data
-   Applying operations to independent groups within the data
-   Reshaping data into different forms
-   Combing multiple datasets together
-   Advanced time-series functionality
-   Visualization through Matplotlib and [Seaborn](https://seaborn.pydata.org/)

![](https://miro.medium.com/max/30/1*MM_2l0tbeDhT_4ftq3gg1Q.png?q=20)

![](https://miro.medium.com/max/700/1*MM_2l0tbeDhT_4ftq3gg1Q.png)

## Matplotlib and Seaborn

Data visualization and storytelling with your data are essential skills that every data scientist needs to communicate insights gained from analyses effectively to any audience out there. This is equally critical in pursuit of machine learning mastery too as often in your ML pipeline, you need to perform exploratory analysis of the data set before deciding to apply particular ML algorithm.

Matplotlib is the most widely used 2-D Python visualization library equipped with a dazzling array of commands and interfaces for producing publication-quality graphics from your data. [Here is an amazingly detailed and rich article](https://heartbeat.fritz.ai/introduction-to-matplotlib-data-visualization-in-python-d9143287ae39) on getting you started on Matplotlib.

![](https://miro.medium.com/max/30/1*ZbyeNyic4ysGwbFgP1kb0Q.png?q=20)

![](https://miro.medium.com/max/700/1*ZbyeNyic4ysGwbFgP1kb0Q.png)

Seaborn is another great visualization library **focused on statistical plotting**. It’s worth learning for machine learning practitioners. Seaborn provides an API (with flexible choices for plot style and color defaults) on top of Matplotlib, defines simple high-level functions for common statistical plot types, and integrates with the functionality provided by Pandas. [Here is a great tutorial on Seaborn for beginners](https://www.datacamp.com/community/tutorials/seaborn-python-tutorial).

![](https://miro.medium.com/max/30/1*GrXULemn0fKH3248z2CNXg.png?q=20)

![](https://miro.medium.com/max/673/1*GrXULemn0fKH3248z2CNXg.png)

**Example of Seaborn plots**

## Scikit-learn

Scikit-learn is the most important general machine learning Python package you must master. It features various [classification](https://en.wikipedia.org/wiki/Statistical_classification), [regression](https://en.wikipedia.org/wiki/Regression_analysis), and [clustering](https://en.wikipedia.org/wiki/Cluster_analysis) algorithms, including [support vector machines](https://en.wikipedia.org/wiki/Support_vector_machine), [random forests](https://en.wikipedia.org/wiki/Random_forests), [gradient boosting](https://en.wikipedia.org/wiki/Gradient_boosting), [_k_-means](https://en.wikipedia.org/wiki/K-means_clustering), and [DBSCAN](https://en.wikipedia.org/wiki/DBSCAN), and is designed to inter-operate with the Python numerical and scientific libraries [NumPy](https://en.wikipedia.org/wiki/NumPy) and [SciPy](https://en.wikipedia.org/wiki/SciPy). It provides a range of supervised and unsupervised learning algorithms via a consistent interface. The vision for the library has a level of robustness and support required for use in production systems. This means a deep focus on concerns such as ease of use, code quality, collaboration, documentation, and performance. [Look at this gentle introduction to machine learning vocabulary](http://scikit-learn.org/stable/tutorial/basic/tutorial.html) as used in the Scikit-learn universe. [Here is another article demonstrating a simple machine learning pipeline](https://towardsdatascience.com/machine-learning-with-python-easy-and-robust-method-to-fit-nonlinear-data-19e8a1ddbd49) method using Scikit-learn.

----------

> Don’t have time to scour the Internet for the latest in deep learning? Don’t worry, we’ve got you covered. [Subscribe](https://www.deeplearningweekly.com/newsletter?utm_campaign=dlweekly-newsletter-timesaver1&utm_source=heartbeat) for our weekly list of updates from the deep learning world.

----------

# Some hidden gems of Scikit-learn

Scikit-learn is a great package to master for machine learning beginners and seasoned professionals alike. However, even experienced ML practitioners may not be aware of all the hidden gems of this package which can aid in their task significantly. I am trying to list few of these relatively lesser known methods/interfaces available in Scikit-learn.

**_Pipeline_**: This can be used to chain multiple estimators into one. This is useful as there is often a fixed sequence of steps in processing the data, for example feature selection, normalization and classification. [Here is more info about it](http://scikit-learn.org/stable/modules/pipeline.html).

**_Grid-search_**: Hyper-parameters are parameters that are not directly learnt within estimators. In Scikit-learn they are passed as arguments to the constructor of the estimator classes. It is possible and recommended to search the hyper-parameter space for the best [cross validation](http://scikit-learn.org/stable/modules/cross_validation.html#cross-validation) score. Any parameter provided when constructing an estimator may be optimized in this manner. [Read more about it here](http://scikit-learn.org/stable/modules/grid_search.html#grid-search).

**_Validation curves_**: Every estimator has its advantages and drawbacks. Its generalization error can be decomposed in terms of bias, variance and noise. The **bias** of an estimator is its average error for different training sets. The **variance** of an estimator indicates how sensitive it is to varying training sets. Noise is a property of the data. It is very helpful to plot the influence of a single [hyperparameter](https://heartbeat.fritz.ai/tuning-machine-learning-hyperparameters-40265a35c9b8) on the training score and the validation score to find out whether the estimator is overfitting or underfitting for some hyperparameter values. [Scikit-learn has a built-in method to help here](http://scikit-learn.org/stable/modules/learning_curve.html).

![](https://miro.medium.com/max/30/1*94PW-U1fK7LoDCNKGMMBsw.png?q=20)

![](https://miro.medium.com/max/700/1*94PW-U1fK7LoDCNKGMMBsw.png)

**_One-hot encoding of categorial data_**: It is an extremely common data preprocessing task to transform input categorical features in one-in-k binary encodings for using in classification or prediction tasks (e.g. logistic regression with mixed numerical and text features). Scikit-learn offers [powerful yet simple methods](http://scikit-learn.org/stable/modules/preprocessing.html#encoding-categorical-features) to accomplish this. They operate directly on Pandas dataframe or Numpy arrays thereby freeing the user to write any special map/apply function for these transformations.

**_Polynomial feature generation_**: For countless regression modeling tasks, often it is useful to add complexity to the model by considering nonlinear features of the input data. A simple and common method to use is polynomial features, which can get features’ high-order and interaction terms. [Scikit-learn has a ready-made function](http://scikit-learn.org/stable/modules/preprocessing.html#generating-polynomial-features) to generate such higher-order cross-terms from a given feature set and user’s choice of highest degree of polynomial.

**_Dataset generators_**: Scikit-learn includes various random sample generators that can be used to build artificial datasets of controlled size and complexity. [It has functions for classification, clustering, regression, matrix decomposition, and manifold testing.](http://scikit-learn.org/stable/datasets/index.html#sample-generators)

![](https://miro.medium.com/max/30/1*6-6-1-k5q8_ZN7RrkL8_tw.png?q=20)

![](https://miro.medium.com/max/700/1*6-6-1-k5q8_ZN7RrkL8_tw.png)

# Practicing Interactive Machine Learning

Project Jupyter was born out of the [IPython Project](https://ipython.org/) in 2014 and evolved rapidly to support interactive data science and scientific computing across all major programming languages. There is no doubt that it has left one of the biggest degrees of impact on how a data scientist can quickly test and prototype his/her idea and showcase the work to peers and open-source community.

However, **learning and experimenting with data become truly immersive when the user can interactively control the parameters of the model** and see the effect (almost) real-time. Most of the common rendering in Jupyter are static.

> But you want more control, **you want to change variables at the simple swipe of your mouse, not by writing a for-loop**. What should you do? You can use **IPython widget**.

Widgets are eventful python objects that have a representation in the browser, often as a control like a slider, text box, etc., through a front-end (HTML/JavaScript) rendering channel.

In [this article](https://towardsdatascience.com/a-very-simple-demo-of-interactive-controls-on-jupyter-notebook-4429cf46aabd), I demonstrate a simple curve fitting exercise using basic widget controls. In a follow-up article, that is [extended further in the realm of interactive machine learning techniques](https://towardsdatascience.com/interactive-machine-learning-make-python-lively-again-a96aec7e1627).
