
# Interactive Machine Learning: Make Python ‘Lively’ Again

## Notebooks come alive when interactive widgets are used. Users can visualize and control changes in the data and the model. Learning becomes an immersive, plus fun, experience.

[](https://towardsdatascience.com/@tirthajyoti?source=post_page-----a96aec7e1627----------------------)

![Tirthajyoti Sarkar](https://miro.medium.com/fit/c/96/96/1*dROuRoTytntKE6LLBKKzKA.jpeg)

[Tirthajyoti Sarkar](https://towardsdatascience.com/@tirthajyoti?source=post_page-----a96aec7e1627----------------------)

Follow

[Dec 16, 2017](https://towardsdatascience.com/interactive-machine-learning-make-python-lively-again-a96aec7e1627?source=post_page-----a96aec7e1627----------------------) · 4 min read

![](https://miro.medium.com/max/60/1*AGKCwMcNxpeolzxTzia5Ng.jpeg?q=20)

![](https://miro.medium.com/max/700/1*AGKCwMcNxpeolzxTzia5Ng.jpeg)

You have coded in Jupyter, the ubiquitous notebook platform for coding and testing cool ideas in virtually all major programming languages. You love it, you use it regularly.

> But you want more control, **you want to change variables at the simple swipe of your mouse, not by writing a for-loop**. What should you do? You can use IPython widget. Read on…

## What is Python Widget?

Project Jupyter was born out of the [IPython Project](https://ipython.org/) in 2014 and evolved rapidly to support interactive data science and scientific computing across all major programming languages. There is no doubt that it has left one of the biggest degrees of impact on how a data scientist can quickly test and prototype his/her idea and showcase the work to peers and open-source community.

However, learning and experimenting with data become truly immersive when user can interactively control the parameters of the model and see the effect (almost) real-time. Most of the common rendering in Jupyter are static. However, there is a [big effort to introduce elements called **_ipywidgets_**](http://jupyter.org/widgets.html), which renders fun and interactive controls on the Jupyter notebook.

> Widgets are eventful python objects that have a representation in the browser, often as a control like a slider, textbox, etc., through a front-end (HTML/Javascript) rendering channel.

In a [previous article](https://towardsdatascience.com/a-very-simple-demo-of-interactive-controls-on-jupyter-notebook-4429cf46aabd), I demonstrated a simple curve fitting exercise using basic widget controls. **Please read that article for instructions related to the installation of this widget package**. In this article, that is extended further in the realm of interactive machine learning techniques.

----------

## Interactive Linear Regression

We demonstrate simple linear regression of single variable using interactive control elements. Note, the idea can be extended for complex multi-variate, nonlinear, kernel based regression easily. However, just for simplicity of visualization, we stick to single variable case in this demo.

The boiler plate code is [**available in my Github repository**](https://github.com/tirthajyoti/Widgets). We show the interactivity in two stages. First, we show the data generation process as a function of input variables and statistical properties of the associated noise. Here is a video of the process where user can dynamically generate and plot the nonlinear function using simple slide-bar controls.

Here, the generating function (aka ‘ _ground truth_’) is a 4th degree polynomial and the noise comes from a Gaussian distribution. Next, we write a linear regression function using [scikit-learn’s](http://scikit-learn.org/stable/) **_polynomial features generation_** and **_pipeline methods_**. A detailed [**step-by-step guide of such a machine learning pipeline process is given here**](https://towardsdatascience.com/machine-learning-with-python-easy-and-robust-method-to-fit-nonlinear-data-19e8a1ddbd49)**.** Here, we wrap the whole function inside another interactive control widget to be able to dynamically alter the various parameters of the linear model.

![](https://miro.medium.com/max/60/1*8yenrFpLoMgPPkXdrDcv2w.png?q=20)

![](https://miro.medium.com/max/700/1*8yenrFpLoMgPPkXdrDcv2w.png)

We introduce interactive control for the following hyperparameters.

-   Model complexity (degree of polynomial)
-   Regularization type — [LASSO or Ridge](https://www.analyticsvidhya.com/blog/2017/06/a-comprehensive-guide-for-linear-ridge-and-lasso-regression/)
-   Size of the test set (fraction of total sample data used in test)

Following video shows the user interaction with the linear regression model. Note, how the test and training scores are also updated dynamically to show a trend of over-fitting or under-fitting as the model complexity changes. One can go back to the data generation control and increase of decrease the noise magnitude to see its impact on the fitting quality and bias/variance trade-off.

----------

## Summary

We presented a brief overview of a Jupyter notebook with embedded interactive control objects which allow the user/programmer to dynamically play with the generation and modeling of a data set. Current demo allows the user to introduce noise, change model complexity, and examine the impact of regularization, all on the fly and see the resulting model and predictions instantly. But the whole idea is explained in a step-by-step manner in the notebook, which should help interested reader to experiment with these widgets and to come up with lively, interactive machine learning or statistical modeling projects.
