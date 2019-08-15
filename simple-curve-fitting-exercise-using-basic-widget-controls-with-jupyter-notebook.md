
# A very simple demo of interactive controls on Jupyter notebook

## Notebooks come alive when interactive widgets are used. Users can visualize and control changes in the data. Learning becomes an immersive, plus fun, experience. Researchers can easily see how changing inputs to a model impacts the results.

[](https://towardsdatascience.com/@tirthajyoti?source=post_page-----4429cf46aabd----------------------)

![Tirthajyoti Sarkar](https://miro.medium.com/fit/c/96/96/1*dROuRoTytntKE6LLBKKzKA.jpeg)

[Tirthajyoti Sarkar](https://towardsdatascience.com/@tirthajyoti?source=post_page-----4429cf46aabd----------------------)

Follow

[Dec 10, 2017](https://towardsdatascience.com/a-very-simple-demo-of-interactive-controls-on-jupyter-notebook-4429cf46aabd?source=post_page-----4429cf46aabd----------------------) · 5 min read

![](https://miro.medium.com/max/60/1*OV9Z2nnWNUHlfeTqyZsotg.jpeg?q=20)

![](https://miro.medium.com/max/700/1*OV9Z2nnWNUHlfeTqyZsotg.jpeg)

P[roject Jupyter/IPython](http://jupyter.org/about.html) has left one of the biggest degrees of impact on how a data scientist can quickly test and prototype his/her idea and showcase the work to peers and open-source community. It is a non-profit, open-source project, born out of the [IPython Project](https://ipython.org/) in 2014, which rapidly evolved to support interactive data science and scientific computing across all major programming languages.

![](https://miro.medium.com/max/60/1*Y4TFSxWLTLB4hF6STdY-uw.png?q=20)

![](https://miro.medium.com/max/700/1*Y4TFSxWLTLB4hF6STdY-uw.png)

Jupyter allows a data scientist/analyst to play with a complex data set and test the model using any of the [leading programming paradigms (Python, C++, R, Julia, Ruby, Lua, Haskell, and many more)](https://github.com/jupyter/jupyter/wiki/Jupyter-kernels) of his/her choice. Behind its apparent simplicity, Jupyter offers powerful rendering capability to produce beautiful markup text and crisp graphics to make a data science/machine learning project come alive in all its glory. [Here is an example of the entire basic set algebra tutorial written in Jupyter notebook](https://nbviewer.jupyter.org/github/tirthajyoti/StatsUsingPython/blob/master/Set_Algebra_with_Python.ipynb).

However, learning and experimenting with data become truly immersive when user can interactively control the parameters of the model and see the effect (almost) real-time. Most of the common rendering in Jupyter are static. However, there is a [big effort to introduce elements called **_ipywidgets_**](http://jupyter.org/widgets.html), which renders fun and interactive controls on the Jupyter notebook. Based on these core elements, several 2D and 3D dynamic data visualization projects (e.g. [bqplot](https://github.com/bloomberg/bqplot), [ipyleaflet](https://github.com/ellisonbg/ipyleaflet)) are also growing in size and scope.

In this article, I am showing a demo notebook with one of the simplest interactive control elements and how it can be integrated in a data modeling task to dynamically visualize the impact of model parameter tuning.

----------

Widgets are eventful python objects that have a representation in the browser, often as a control like a slider, textbox, etc., through a front-end (HTML/Javascript) rendering channel. They use Jupyter ‘comms’ API, which is a symmetric, asynchronous, fire and forget style messaging API allowing the programmer to send JSON-able blobs between the front-end and the back-end and hiding the complexity of the web server, ZMQ, and web-sockets. [Here is a detailed discussion](https://ipywidgets.readthedocs.io/en/stable/examples/Widget%20Low%20Level.html).

![](https://miro.medium.com/max/60/1*zxuFZYWnzygKO0VwBMAYLA.png?q=20)

![](https://miro.medium.com/max/700/1*zxuFZYWnzygKO0VwBMAYLA.png)

## Installing widgets

You have two options for installing ipywidgets. They differ slightly, so note the difference. Basically, if you use pip, you also have to enable the ipywidget extension in your notebook to render it next time you start the notebook. You can enable it within any virtual environment you use, so that the extension does not impact any other environment.

pip install ipywidgets  
jupyter nbextension enable --py widgetsnbextension

Or, you can do a Conda install (if you use Anaconda), it will be enabled automatically.

conda install -c conda-forge ipywidgets

## Using ‘_Interact_’ object/control

The `interact` function (`ipywidgets.interact`) automatically creates user interface (UI) controls for exploring code and data interactively. It is the easiest way to get started using IPython’s widgets.

![](https://miro.medium.com/max/60/1*Xhuo5G0FLNTidNkhOgS5gA.png?q=20)

![](https://miro.medium.com/max/700/1*Xhuo5G0FLNTidNkhOgS5gA.png)

In addition to `interact`, IPython provides another function, `interactive`, that is useful when you want to reuse the widgets that are produced or access the data that is bound to the UI controls. Note that unlike `interact`, the return value of the function will not be displayed automatically, but you can display a value inside the function with `IPython.display.display`.

## Demo code for a simple curve fitting exercise

Boiler plate code is [available on my GitHub repository](https://github.com/tirthajyoti/Widgets). Please feel free to fork/download and play with it. We are essentially generating a Gaussian data with noise and playing with various model parameters with the help of interactive slide-bar or drop-down menu controls. After passing the function and rendering the display with the object, returned by the ipywidget.interactive (), here is what it looks like…

![](https://miro.medium.com/max/60/1*ns46ZzRFzwADAo0M1q6qDA.png?q=20)

![](https://miro.medium.com/max/700/1*ns46ZzRFzwADAo0M1q6qDA.png)

Next, we call [curve-fitting optimizer function from Scipy package](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html) and fit a ideal Gaussian distribution function to the data. The code describes exactly how to extract the data from the ipywidget object and pass it on to the optimizer. You can play with various model parameters and see how the optimizer fitted/estimated parameters differ from the ideal values as the noise parameters are changed, **all by sliding the controls left or right**!

![](https://miro.medium.com/max/60/1*V1ATrNc5hHjszl5S8r_EzA.png?q=20)

![](https://miro.medium.com/max/700/1*V1ATrNc5hHjszl5S8r_EzA.png)

----------

## Summary

We discussed some basics about IPython widgets or interactive controls and why they can turn a dry data science code notebook into a fun, living document. We also showed a simple demo code to illustrate the idea in a statistical model fitting exercise. In future, we hope to show some more cool applications of widgets in machine learning model tuning or deep learning hyperparameter optimization.
