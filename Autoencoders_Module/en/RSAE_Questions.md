---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.11.2
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

<!-- #region id="HQOsPSWZyQZx" -->
# Movie recommendation using Autoencoder 
## IVADO - Online course on Recommender Systems 

Colab notebook content originally created for IVADO's workshop on Recommender Systems, August 2019. 


<b> Authors: </b>

David Berger 

Laurent Charlin 

Nissan Pow 
<!-- #endregion -->

<!-- #region id="qYuwYEztyQZ0" -->
# 1. Introduction

In this workshop, we will see how to implement recommendation systems based on an autoencoder (AE), a classic architecture in deep learning. As with the *Matrix factorization* tutorial, we will use the <a href="https://grouplens.org/datasets/movielens/">MovieLens</a> database to train the models and conduct experiments.
<!-- #endregion -->

<!-- #region id="XDKq8-yXyQZ1" -->
## 1.1 Installing libraries

Before we begin, we must make sure to install the libraries for the tutorial using `pip`. To do this, run the following cell by selecting it and then press the `shift`+`Enter` keys.
<!-- #endregion -->

```python id="and193IDyQZ2" colab={"base_uri": "https://localhost:8080/"} outputId="d28084dc-4660-4d24-fb72-902d66eaa737"
!rm -rf RS-Workshop
!git clone https://github.com/davidberger2785/RS-Workshop
```

<!-- #region id="jXnUNiZPyQZ8" -->
To ensure that the installation has been done properly, import all the libraries and modules that we will use for this workshop by running the next cell.
<!-- #endregion -->

```python id="ffIsFyBmyQZ8"
import numpy as np
import pandas as pd

# Data vizualisation
import matplotlib.pyplot as plt
import seaborn as sns

import os
import sys

# Homemade functions
sys.path += ['RS-Workshop/Tutoriels - En/']
import utilities as utl
```

<!-- #region id="pjO04i-MyQZ_" -->
We have also coded some boilerplate functions that we have grouped together in the `utilities` library. In fact, these functions probably already exist in Python, but we often simply ignore they exist...
<!-- #endregion -->

<!-- #region id="bt2C99HCyQZ_" -->
## 1.2 Goal

The purpose of recommender systems is, as the name implies, to make personalized recommendations. Ideally, these recommendations will have to be good, although this concept may seem fuzzy. Unlike other tasks in machine learning, for example cat images recognition or stock market predictions, making recommendations to help a user is more complex. Is the aim to present a specific user with suggestions that reinforce their previous choices? Do we want to present complementary or totally independent suggestions for the items previously considered? Will we try instead to present them items to which they have not yet been exposed? Each of these options are correct and can be modeled. Without loss of generality, the diagram below simply models the issue of recommendation systems from the perspective of machine learning.

![title](https://github.com/davidberger2785/RS-Workshop/blob/master/Images/High_level_1.png?raw=1)



Nevertheless, in the context of this workshop, by using the example of movies available on streaming platforms such as Netflix or Amazon Prime, we can reduce the problem to a relatively simple task: recommend movies that the user will like according to their past interests. In order to carry out this task, we will use all the preferences of the users, certain associated sociodemographic variables as well as certain characteristics of the movies. Finally, we can refine the diagram as follows:

![title](https://github.com/davidberger2785/RS-Workshop/blob/master/Images/High_level_2.png?raw=1)

<!-- #endregion -->

<!-- #region id="Z8kkOcZLyQaA" -->
## 1.3 The MoviesLens dataset(s)

The data used here consists of approximately 100k movie ratings by 943 users. Over 1,6k movies are available. Beyond the 100k ratings, additional information related to users and movies is available.

We will use three different datasets to carry out our analyses:

*   Users : related to users' characteristics,
*   Movies : related to movies' characteristics,
*   Ratings : containing over 100k ratings.

We are using the <a href="https://pandas.pydata.org/">Pandas</a> library in order to download and manipulate the datasets.
<!-- #endregion -->

<!-- #region id="apGD8Z_CyQaA" -->
### 1.3.1 Users: Download and preprocessing
<!-- #endregion -->

```python id="F6pLOxHjyQaB"
# Download
ROOT_DIR = 'RS-Workshop/'
DATA_DIR = os.path.join(ROOT_DIR, 'data/ml-100k/')

users = pd.read_csv(os.path.join(DATA_DIR, 'u.user'), 
                        sep='|', header=None, engine='python', encoding='latin-1')

# Different variables according to the information provided in the readme file
users.columns = ['Index', 'Age', 'Gender', 'Occupation', 'Zip code']

# Quick overview
users.head()
```

<!-- #region id="6xg1jIhbyQaD" -->
Before presenting some descriptive statistics related to the population, we will format the users' data as a <a href="https://en.wikipedia.org/wiki/List_(abstract_data_type)">list</a> to handle it more easily. The occupation of each user is a string, so we have recoded the 21 possible occupations in boolean.
<!-- #endregion -->

```python id="C7KCPwC9yQaE"
# Number of users
nb_users = len(users)

# Gender
gender = np.where(np.matrix(users['Gender']) == 'M', 0, 1)[0]

# Occupation
occupation_name = np.array(pd.read_csv(os.path.join(DATA_DIR, 'u.occupation'), 
                                            sep='|', header=None, engine='python', encoding='latin-1').loc[:, 0])

# Boolean transformation of user's occupation
occupation_matrix = np.zeros((nb_users, len(occupation_name)))

for k in np.arange(nb_users):
    occupation_matrix[k, occupation_name.tolist().index(users['Occupation'][k])] = 1

# Concatenation of the sociodemographic variables 
user_attributes = np.concatenate((np.matrix(users['Age']), np.matrix(gender), occupation_matrix.T)).T.tolist()
```

<!-- #region id="MupJaXiZyQaG" -->
We then explore the descriptive statistics of the users. These include information related to *age* (continuous variable), *gender* (binary variable) and *occupation* of each user (21, all binary).
<!-- #endregion -->

<!-- #region id="cMm2LDjSyQaG" -->
Descriptive statistics related to <i>Age.
<!-- #endregion -->

```python id="ZF4Y5TUQyQaH"
pd.DataFrame(users['Age'].describe()).T
```

<!-- #region id="LeX_TqtCyQaJ" -->
Barplot graph related to <i>Gender.
<!-- #endregion -->

```python id="ktjctlNPyQaJ"
utl.barplot(['Women', 'Men'], np.array([np.mean(gender) , 1 - np.mean(gender)]) * 100, 
            'Sex', 'Percentage (%)', "User's gender", 0)
```

<!-- #region id="Tv5sHwVZyQaO" -->
Barplot graph related to <i>Occupation.
<!-- #endregion -->

```python id="z0_b6pd9yQaP"
attributes, scores = utl.rearrange(occupation_name, np.mean(occupation_matrix, axis=0) * 100)
utl.barplot(attributes, scores, 'Occupation', 'Percentage (%)', "User's occupation", 90)
```

<!-- #region id="PyIfNODfyQaR" -->
### 1.3.2 Movies: Download and preprocesssing

We will now process and explore the data associated with movies. For each movie, we have the *title*, the *release date* (in North America), as well as the corresponding *genres*.
<!-- #endregion -->

```python id="Em0nC2FpyQaR"
movies = pd.read_csv(os.path.join(DATA_DIR, 'u.item'), sep='|', header=None, engine='python', encoding='latin-1')

# Number of movies
nb_movies = len(movies)

# Genres
movies_genre = np.matrix(movies.loc[:, 5:])
movies_genre_name = np.array(pd.read_csv(os.path.join(DATA_DIR, 'u.genre'), sep='|', header=None, engine='python', encoding='latin-1').loc[:, 0])

# Quick overview
movies.columns = ['Index', 'Title', 'Release', 'The Not a Number column', 'Imdb'] + movies_genre_name.tolist()
movies.head()
```

<!-- #region id="c0uDnLOYyQaT" -->
We present the ratio of movies according to the genre as a descriptive statistic.
<!-- #endregion -->

```python id="eqaRcdBtyQaU"
attributes, scores = utl.rearrange(movies_genre_name, 
                                   np.array(np.round(np.mean(movies_genre, axis=0) * 1, 2))[0])
utl.barplot(attributes, np.array(scores) * 100, xlabel='Genre', ylabel='Percentage (%)', 
            title=" ", rotation = 90)
```

<!-- #region id="8BIFt6pxyQaW" -->
### 1.3.3 Ratings: Download and preprocessing

The dataset based on users ratings consists of approximately 100k lines (one evaluation per line) where the following are presented: 

*   the *user identification number*
*   the *identification number of the movie*
*   its associated *rating*
*   *a time marker*

The training and test sets were provided as is (we do not need to build them ourselves) and have 80k and 20k evaluations respectively.

For practical reasons, we convert the database as a list using our `convert` util function.
<!-- #endregion -->

```python id="SlMXfiltyQaX"
training_set = np.array(pd.read_csv(os.path.join(DATA_DIR, 'u1.base'), delimiter='\t'), dtype='int')
testing_set = np.array(pd.read_csv(os.path.join(DATA_DIR, 'u1.test'), delimiter='\t'), dtype='int')

train_set = utl.convert(training_set, nb_users, nb_movies)
test_set = utl.convert(testing_set, nb_users, nb_movies)
```

<!-- #region id="5zvO5UCTyQaZ" -->
As we did before, we can get descriptive statistics associated with the evaluations. At first, it might be interesting to study the average trends of users.

##### Question 1

1. What other types of statistics might be interesting?
<!-- #endregion -->

```python id="qu-AEw28yQaZ"
train_matrix = np.array(train_set)
shape = (len(train_set), len(train_set[0]))
train_matrix.reshape(shape)
train_matrix_bool = np.where(train_matrix > 0 , 1, 0)

user_watch = np.sum(train_matrix_bool, axis=1)
pd.DataFrame(user_watch).describe().T
```

<!-- #region id="T9Dsm_LZyQac" -->
Histogram of the number of movies watched per user.
<!-- #endregion -->

```python id="XQhKtRRFyQad"
sns.set(rc={'figure.figsize':(12,8)})
sns.set(font_scale = 1.5)

plt.title('Empirical distribution of \n the number of movies watched per user')
plt.xlabel('Number of movies watched')
plt.ylabel('Number of users')
plt.hist(user_watch, 100);
```

<!-- #region id="NJNTBhdeyQag" -->
Descriptive statistics related to the movies' evaluation.
<!-- #endregion -->

```python id="vZXYJ4eryQah"
movie_frequency = np.mean(train_matrix_bool, axis=0)
pd.DataFrame(movie_frequency).describe().T
```

<!-- #region id="67pC2NGKyQaj" -->
##### Question 2

a. What statistics or observations might we consider relevant? Why?

b. What kind of statistics might be more appropriate in such a context?
<!-- #endregion -->

```python id="Z4KxdnsTyQaj"
plt.xlabel('Proportion of the population who watched the movie')
plt.ylabel('Number of Movies')
plt.hist(movie_frequency, 100);
```

<!-- #region id="Qfpcfvk_yQal" -->
#### Individual preferences according to the type of movie

We could also look at the behavior of a particular individual. Among other things, we could study if there is a bias associated with their evaluation scheme or what are their cinematographic preferences according to the score awarded.
<!-- #endregion -->

```python id="74RuNFKtyQal"
def stats_user(data, movies_genre, user_id):
    
    ratings = data[user_id]
    stats = np.zeros(6)
    eva = np.zeros((6, movies_genre.shape[1]))

    for k in np.arange(len(ratings)):
        index = int(ratings[k])
        stats[index] += 1
        eva[index, :] = eva[index, :] + movies_genre[k]

    return stats, eva
```

```python id="J9sH_M2HyQan"
user_id = 0
stats, eva = stats_user(train_set, movies_genre, user_id)
utl.barplot(np.arange(5) + 1, stats[1:6] / sum(stats[1:6]), xlabel='Number of stars', ylabel='Percentage of movies (%)', 
            title=" ", rotation = 0)
```

<!-- #region id="zuIGfhUFyQaq" -->
##### Question 3

3. How can we test to the existence of bias associated to individual's assessment scheme?
<!-- #endregion -->

<!-- #region id="s4jcmvzwyQaq" -->
## 1.4 Construction of the training and validation sets

In machine learning, we manipulate <a href="https://blogs.nvidia.com/blog/2018/04/15/nvidia-research-image-translation/">complex databases</a> for which we attempt to define equally complex function spaces in order to accomplish a specific task. That being said, these function spaces are defined by a set of parameters whose number tends to increase with the complexity of the data. Once space is defined by a set of fixed parameters, we can vary the different values of the hyperparameters in order to empirically explore function spaces. To choose the set of optimal parameters and hyperparameters, we define a metric that allows us to evaluate the model; for example, how much the image of a cat seems likely.

To the extent that we want to develop a model that can generalize, the evaluation of its performance must be done on an independent dataset, coming from the same distribution from the set on which the model was trained. This set is called the validation set.

**! Note !** 

The notion of training and test set in the framework of recommender systems is somewhat different from what is usually seen with so-called supervised problems. If in the context of a supervised problem, the test set consists essentially of new observations (lines from a file) which are independent of observations previously observed in the training set. The paradigm is significantly different when we work with recommendation systems.

Indeed, because of the mathematical model on which the recommendation systems are based, the data belonging to the test set are not linked to a new individual, but rather to new evaluations made by the same set of individuals. As a result, the data associated with the training, validation and test sets are no longer independent as assumed (the famous *iid* hypothesis) which complicates things theoretically.

Since the purpose of the workshop is not to study the notion of bias associated with the type of dependence between the different assessments in referral systems, we will naively assume that each of the assessments is independent of each other. Nevertheless, in a practical framework, ignoring this kind of considerations will possibly bias the algorithms.
<!-- #endregion -->

```python id="vvqL3zJmyQaq"
def split(data, ratio, tensor=False):
    train = np.zeros((len(data), len(data[0]))).tolist()
    valid = np.zeros((len(data), len(data[0]))).tolist()

    for i in range(len(data)):
        for j in range(len(data[i])):
            if data[i][j] > 0:
                if np.random.binomial(1, ratio, 1):
                    train[i][j] = data[i][j]
                else:
                    valid[i][j] = data[i][j]

    return [train, valid]

train_0 = split(train_set, 0.8)
test = test_set
```

<!-- #region id="0tYJWzOVyQaw" -->
# 2. Autoencoder as recommender systems
<!-- #endregion -->

<!-- #region id="XPQ8-o41yQax" -->
## 2.1 Model

In general, auto-encoders are a class of neural networks that allow unsupervised learning of the latent characteristics of the data being studied. To do this, the AE will attempt to predict, or copy, the input observations using (multiple) hidden layer. In its simplest form, the architecture of an AE can be summarized in the diagram below.

![title](https://github.com/davidberger2785/RS-Workshop/blob/master/Images/AE.png?raw=1)

Looking more closely, the AE consists of an encoder, the function $h(\cdot)$ defined by:

$$
\begin{align}
    h(\mathbf{x}) = \frac{1}{1+ \exp(-\mathbf{W} \mathbf{x})}.
\end{align}
$$

This function takes as input the observations and will consist of recoding it as a hidden layer so as to reduce their size (fewer neurons). Afterwards, an encoder defined by:

$$
\begin{align}
    f(h(\mathbf{x})) = \mathbf{W}^\top h(\mathbf{x})
\end{align}
$$

will attempt *to reconstruct* the input observations from the hidden layer. In this sense, the AE tries to estimate the observations used as input.

We can advantageously use the estimates made by the auto-encoders in such a way as to present new recommendations to the users.

For example, suppose that the set of evaluations made by a user is defined by the vector:

$$
\begin{align}
    \mathbf{x} = [3, \ 0, \ 0, \ 1, \ ..., \ 2, \ 4].
\end{align}
$$

We are noticing that the movie occupying the first position, *Toy Story*, was moderately appreciated, while the next two movies, *Golden Eye* and *Four Room*, have not been viewed at all. Suppose once again that in this same set of assessments, the EA will present the following estimates:

$$
\begin{align}
    \mathbf{\hat{x}} = [3.2, \ 1.3, \ 4, \ 0.5, \ ..., \ 3, \ 1].
\end{align}
$$

As a result, we will be able to use the estimates associated with initially unvisited movies as recommendations. Thus, the movie *Four Rooms*, in third position, seems a good suggestion for the user, while *Golden Eye* is definitely not a convincing recommendation.
<!-- #endregion -->

<!-- #region id="tb8oH65vyQax" -->
## 2.2 Deep learning with Pytorch

In order to build a recommendation system based on autoencoders, we will use the
<a href="https://pytorch.org/">Pytorch</a> library. It provides two extremely interesting features:
<ul>
<li> Manipulation of tensors (kind of multidimensional matrices) to perform calculations with GPU. </li>
<li> Automatic differentiation (!!!) with the <a href="http://pytorch.org/docs/master/autograd.html">autograd class</a> to easily calculate the gradient descent. </li>
<!-- #endregion -->

```python id="5dJgvL_1yQay"
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data

from torch.nn import functional
from torch.autograd import Variable
```

<!-- #region id="_8KzAwHKyQaz" -->
Since we will work with Pytorch, let's turn the MoviesLens dataset into a tensor object.
<!-- #endregion -->

```python id="EjAhxXU6yQa0"
train = torch.FloatTensor(train_0)
train, valid = train[0], train[1]
test = torch.FloatTensor(test_set)
```

<!-- #region id="MTz99dilyQa2" -->
**! Note !** 

Although the documentation available for Pytorch is detailed (compared to other deep learning libraries), it is easy to get lost. Nevertheless, for this workshop, it is not necessary to enter all the details associated with the different commands. In fact, the key is to understand the ins and outs of the key steps presented.
<!-- #endregion -->

<!-- #region id="FgHOkq9yyQa2" -->
## 2.3 Implementation

We can decline in 5 steps the implementation of an AE as a recommendation system:

1. Initialization of the AE,
2. Propagation of the message,
3. Estimation: cost calculation and retropropagation,
4. Learning loop,
5. Evaluation.
<!-- #endregion -->

<!-- #region id="8afngqjXyQa2" -->
### 2.3.1 Initialization

First, let's define the autoencoder class using the <a href="http://pytorch.org/docs/master/nn.html#module"> torch.nn</a> module. In PyTorch, any neural network must inherit this class. The autoencoder uses other common classes in Pytorch, such as <a href = "http://pytorch.org/docs/master/nn.html#torch.nn. Linear "> torch.nn.Linear (in_features, out_features)</a>. The latter implements a fully connected linear layer (as its name suggests).

<!-- #endregion -->

<!-- #region id="Pw5N1JF0Ca1L" -->
##### Question 4

4. Complete the initialization of the autoencoder class according to the architecture diagram presented above.
<!-- #endregion -->

```python id="-RVCLuGayQa3"
class AE(nn.Module):
    def __init__(self, inputs, outputs, features, criterion=None):
        """
        Args: 
           self: class name
           nb_inputs: number of neurons on the input layer
           nb_outputs: number of neurons on the output layer
           nb_features: number of neurons on the hidden layer
           criterion: loss function used for learning  
        """
        
        super(AE, self).__init__()
        # Complete this line 
        # And this one too
        
        self.criterion = criterion
```

<!-- #region id="ozLRChh6yQa5" -->
We then define:

1. The number of neurons input,
2. The number of neurons output,
3. The number of neurons desired in the hidden layer.
<!-- #endregion -->

<!-- #region id="GXv9KoeQyQa5" -->
##### Question 5




<!-- #endregion -->

<!-- #region id="qRMLG7iqDMfX" -->
###### Question 5.1
Initialize the autoencoder with the correct parameter values.
<!-- #endregion -->

```python id="VJXLOMYZyQa6"
nb_inputs = ?
nb_outputs = ?
nb_features = ?

ae = AE(?)
```

<!-- #region id="jJWe85RbDDlX" -->
###### Question 5.2

Is it relevant that the hidden layer has more neurons than the input layer?
<!-- #endregion -->

<!-- #region id="TyH3ax3HyQa9" -->
### 2.3.2 Propagation

During the propagation phase, the `forward` function associated with the propagation of the message defines the operations to be performed in order to calculate the elements of the output. This function is essential and must match the initialization of the model in the previous step to allow proper backpropagation. 
   
Note the use of the <a href="http://pytorch.org/docs/master/nn.html#torch-nn-functional">torch.nn.functional</a> method which define a set of functions which can be applied to the layers of a neural network. In this workshop, we will use non-linear functions like <a href="http://pytorch.org/docs/master/nn.html#id36">sigmoid</a> and cost functions such as the mean squared error <a href="http://pytorch.org/docs/master/nn.html#mse-loss">mse_loss</a>.
<!-- #endregion -->

<!-- #region id="LOpI_qz9yQa-" -->
##### Question 6

6. Write down the `forward` function.
<!-- #endregion -->

```python id="w4qq6sRCyQa-"
def forward(model, x):
    """
    Args:
        model: name of the autoencoder as initialized
        x: input layer, here made up of 1682 neurons
    Return:
        predictions: output layer
    """

    # TO-DO
    
    return ?
```

<!-- #region id="vXgbMhgVyQbB" -->
### 2.2.2 Weight estimates

Although neural networks have breathtaking predictive capabilities, the complexity of their architecture can become high very quickly. From a computational point of view, among other things, this translates into the impossibility of obtaining an optimum regards to the loss function. Of course, estimating the weights analytically is mostly impossible, as can be done in a linear model under normality assumption for example. Nevertheless, if no global optimum is guaranteed, and if incidentally no analytical form can be calculated, the associated weights can be estimated.

That being said, the (stochastic) gradient descent (and its derivatives) is an efficient optimization technique that is mainly used in deep learning. This technique uses three key concepts:

1. Cost function,
2. Optimizer's type,
3. Gradient backpropagation (implemented in the learning loop).
<!-- #endregion -->

<!-- #region id="ZyXo169lyQbB" -->
#### 2.2.2.1 Loss function

As we saw in the previous workshop, the loss function plays an important role in the construction of a predictive model. In fact, it is this same loss function that we try to minimize (or maximize) by iteratively adjusting the weights of the AE. Thus, two different loss functions will most likely result in two different models. As usual, Pytorch offers a large amount of <a href="http://pytorch.org/docs/master/nn.html#id42">loss functions</a> that you can explore at your leisure.

Since ratings vary between 1 and 5, the mean square error (MSE) seems an interesting first option. Formally, as part of a recommendation system, we will define the MSE as follows:

$$
\begin{align}
\textit{MSE}(\mathbf{R}, \hat{\mathbf{R}}) = \frac{1}{n} \sum_{r_{ui} \neq 0} (r_{ui} - \hat{r}_{ui})^2, 
\end{align}
$$

where $\mathbf{R}$ and $\hat{\mathbf{R}} $ are respectively the matrices of the observed and predicted ratings and $n$ is the number of estimates. In the same way, $r_{ui}$ and $\hat{r}_{ui} $ are scalars associated respectively with the observed evaluation and the estimate of the user $u$ for the item $i$.

Since we have encoded the loss function as an attribute of the autoencoder class, we define it with the following command.
<!-- #endregion -->

```python id="EEXLzacKyQbC"
ae.criterion = nn.MSELoss()
```

<!-- #region id="RGMAyx5TyQbD" -->
##### Question 7

7. The MSE is an interesting loss function for recommendation system with explicit data. Which relevant loss function could have been implemented if the data had been preference-based binary ratings?
<!-- #endregion -->

<!-- #region id="plmRF7m8yQbE" -->
#### 2.2.2.2 Optimizer

PyTorch provides several <a href="http://pytorch.org/docs/master/optim.html#algorithms">optimization methods</a> more or less derived from the gradient descent via the `torch 'class. optim`. Among these techniques:

<ul>
<li> SGD (Stochastic Gradient Descent): implementation of SGD.
<li> Adam (Adaptive Moment Estimation): variation of the gradient descent method where the learning rate is adjusted for each parameter.
<li> RMSprop: actual optimizer for this workshop. For more details <a href="http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf"> here</a>.
</ul>

In any case, when we use iterative optimization methods, we must provide a learning rate and a weight decay value, for reasons similar to those mentioned in the <i>Matrix factorization</i> workshop.
<!-- #endregion -->

```python id="AUF_1V9FyQbE"
learning_rate = 0.02
weight_decay = 0.2

optimizer = optim.RMSprop(ae.parameters(), lr=learning_rate, weight_decay=weight_decay)
```

<!-- #region id="gFiTJAkkyQbH" -->
#### 2.2.2.3 Backpropagation

In Pytorch, the gradient's backpropagation is simplified thanks to the automatic differentiation and the <a href="http://pytorch.org/docs/master/notes/autograd.html">autograd</a> class. This is done in two steps:

1. Calculation of the loss function with the function previously defined in the class of the AE.
2. Automatic differentiation of the loss function with the `backward()` function.

The entire backpropagation process is directly implemented in the learning loop defined below.
<!-- #endregion -->

<!-- #region id="5IZE4qcSyQbH" -->
### 2.2.3 Learning loop

When an autoencoder (and generally a deep learning-based architecture) is used as a recommendation system, the learning loop differs somewhat from the matrix factorization-based models. Thus, each rating is no longer considered individually, as it was the case before, but is considered on the whole of the ratings provided by a specific user.

<!-- #endregion -->

<!-- #region id="5l865Yw4JC4g" -->
##### Question 8

a. Complete the propagation phase.

b. At the end of each epoch, which statistic would it be better to calculate? Code it. Note: initialize objects at the beginning of the function (see line 4)

c. Implement the backpropagation phase.

d. Since data from the training, validation or test set can be used in the fit function, what conditions should we put on line 22?
<!-- #endregion -->

```python id="mdAUWt8vyQbH"
def fit(model, x, y, valid=False):
    
    nb_obs, nb_items = len(x), len(x[0])
    average_loss, s = 0, 0.

    for id_user in range(nb_obs):

        inputs = Variable(x[id_user]).unsqueeze(0)
        target = Variable(y[id_user]).unsqueeze(0)

        if torch.sum(target > 0) > 0:
            
            # Question 8.1: Estimate
            estimate = ?
            # TO-DO
            target.require_grad = False
            
            # Question 8.3: Retropropagation
            loss = ?
            
            # Question 8.4: Condition
            if ?:
                loss.backward()
                optimizer.step()

            # Question 8.2: Statistics
            # TO-DO
            s += 1.

    return model, average_loss, s
```

<!-- #region id="yYyw2GVlyQbJ" -->
## 2.4 Training

The autoencoder and the associated functions now implemented, we can start to train the model. Once again, the goal here is not to tune the parameters so as to obtain the best possible model, but simply to understand the role that they can play according to the model's predictive ability.


<!-- #endregion -->

<!-- #region id="AvN6ja4XJJx1" -->
##### Question 9

9. Finish implementing the training phase.
<!-- #endregion -->

```python id="t-ydW5_KyQbJ"
nb_epoch = 20

for epoch in range(1, nb_epoch + 1):
    
    # TO-DO
 
    print("epoch: ", "{:3.0f}".format(epoch), "   |   train: ", "{:1.8f}".format(train_loss.numpy() / train_s), \
                    '   |   valid: ', "{:1.8f}".format(valid_loss.numpy() / valid_s))
```

<!-- #region id="IrsVTVVeyQbM" -->
You can now manipulate the various parameters and hyperparameters of the AE. Among the various modifications that you can make, here is a (short and not exhaustive) list of easily implementable modifications:

1. Change the hyperparameters (a bit boring).
2. Increase the size of the hidden layer (more interesting).
3. Add hidden layers to the model, making sure to initialize them and adapt the forward function.
4. Dichotomize the ratings using a threshold (3 for example) and run the whole code by adapting or not the cost function as discussed previously.
<!-- #endregion -->

<!-- #region id="4tcquuEUyQbM" -->
Finally, we can evaluate the performance of our model on the test set.
<!-- #endregion -->

```python id="NHCODYiGyQbN"
ae, test_loss, test_s = fit(model=ae, x=test, y=test, valid=True)
print('test: ',"{:1.8f}".format(test_loss.numpy() / test_s))
```

<!-- #region id="D-ZvlGKkyQbQ" -->
## 2.5 Analysis


### 2.5.1 Exploration of the latent layer

In a similar way to what was presented in the matrix factorization workshop, we can explore the latent layer of AE. Insofar as the input layer represents the set of evaluations for a given individual, each neuron in the latent layer will be associated with a latent attribute of an individual.

As an example, let's say $\mathbf{H}_{|U| \times k}$ the matrix associated with the tent layer where each row represents the latent representation of the preferences for a given user and where each column represents a neuron of that same layer. Suppose now that the first two neurons of the latent layer have the following values:

$$
\begin{align}
h_1 &= [1.0, \ 0.0, \ -0.5, \ ..., \ 1.0, \ -1.0]
\qquad \text{and} \qquad
h_2 = [1.0, \ 0.0, \ 0.5, \ ..., \ -1.0, \ -0.8].
\end{align}
$$

And that to these values correspond the following users:

1. Serena,
2. Kali,
3. Neil,
4. Marie,
5. David.

We can then map users based on the values ​​associated with $h_1$ and $h_2$:

![title](https://github.com/davidberger2785/RS-Workshop/blob/master/Images/hidden_4.png?raw=1)


This approach could allow us to perform clusters of individuals based on latent attributes. In fact, a quick glance allows us to think that the preferences in terms of cinema of Serena are the opposite of David, which is not very surprising.

With the following commands, we propose to further explore the latent structure of AE.
<!-- #endregion -->

```python id="rzlHsEykyQbQ"
x = train
hidden = torch.sigmoid(ae.fc1(x)).detach().numpy()
```

<!-- #region id="u_oQUsAByQbT" -->
We could be interested, for example, in measures of association between the different hidden layers, or a hidden layer and a socio-demographic characteristics. To do this, an interesting avenue would be to simply calculate the associated correlations.

First, by simply exploring the correlations between different neurons.
<!-- #endregion -->

```python id="ebCk3BucyQbT"
df = pd.DataFrame(np.array(hidden))
f = plt.figure(figsize=(6, 6))
plt.matshow(df.corr(), fignum=f.number)
plt.xticks(range(df.shape[1]), df.columns, fontsize=10, rotation=0)
plt.yticks(range(df.shape[1]), df.columns, fontsize=10)
cb = plt.colorbar()
cb.ax.tick_params(labelsize=10)
```

<!-- #region id="p-r0v_i7yQbV" -->
##### Question 10

10. Which neuron of the hidden layer seems <i>a priori </i> the most interesting? Why?
<!-- #endregion -->

<!-- #region id="lsY2vDgayQbV" -->
##### Question 11

11. Latent layer study was done here according to the users of the system. Would it be possible to study the latent layers associated with individuals. If so, how?
<!-- #endregion -->

<!-- #region id="3NNUvFz2yQbW" -->
# 3. Applications

One of the primary objectives of the recommendation systems is to make personalized recommendations. Therefore, it might be interesting to study the recommendations made by our model for a specific individual. It would also be preferable that the recommendations made suggest only films not viewed by the user.


<!-- #endregion -->

<!-- #region id="s-5m512lJoXl" -->
##### Question 12

12. Implement a short function which presents the best *k* recommendations that have not yet been viewed by a specific user.
<!-- #endregion -->

```python id="IPXEtoAdyQbW"
def recommendations(model, data, titles, k):
    """
     Args:
         model: name of the autoencoder as initialized
         data: assessments associated with the user
         titles: list of titles of potentially recommended objects
         k: number of recommendations wanted
     Return:
         names: names of the recommendations
         scores: the score associated with them
     """
    # TO-DO
    
    return ?
```

<!-- #region id="gkPtOvMIyQbY" -->
Call of the function with a few manipulations...
<!-- #endregion -->

```python id="l1PVUkKyyQbY"
user_id = 0
k=10
names, scores = recommendations(ae, (train + valid + test)[user_id], movies['Title'], k)

df = pd.DataFrame(np.matrix((names[-k:], scores[-k:])).T, (np.arange(k) + 1).tolist())
df.columns = ['Title', 'Predicted rating']
df
```

<!-- #region id="B8JoYDYdyQba" -->
As we saw in the workshop on matrix-based recommendation systems, we can customize algorithms with different parameters. For example, we could implement a system with the best recommendations based on:

1. Of a particular genre of movie.
2. A minimum preferred quality: a minimum predicted score strictly greater than 4.5, for example.
<!-- #endregion -->

<!-- #region id="ohnEsX7myQba" -->
##### Question 13

13. Do different algorithms make identical recommendations? 

<!-- #endregion -->

<!-- #region id="9T5Zy189yQbb" -->
# 4. Other modeling ideas

So far, we have only considered the evaluations in our model. It might be interesting to consider other types of modeling.

For example, instead of using film ratings made by an individual as an input layer (hence 1682 neurons), we could use the ratings of individuals for a particular movie (and thus 943 input layer neurons). In this modeling, we could incorporate the different types of films and/or their year of release.

Finally, we could simply get ourselves away from autoencoders and ogle other types of architectures. Considering the different functions and class previously coded, we could implement a multilayer perceptron (MLP). To do this, the inputs of the network would be exactly the same, with the difference that the targets would consist only of unvisited films.
<!-- #endregion -->

<!-- #region id="3kWSAmBEyQbc" -->
## 4.1 Use of socio-demographic informations

It might be interesting to check whether the use of socio-demographic informations improves or not the predictive capabilities of the model. To the extent that such information would only improve the capabilities of the model very little, they could be useful when a new user intends to use the recommendation system already in place. Although imperfect, the information associated with the age, gender and occupation of a user could be useful for presenting the first recommendations.

In order to observe how the recommendation system behaves with such data, we must first modify the different sets so that they present the socio-demographic information of each user.
<!-- #endregion -->

```python id="Mzq15h9TyQbc"
train_inputs = torch.FloatTensor(utl.inner_concatenation(user_attributes, train_0[0]))
train_outputs = torch.FloatTensor(train_0[0])

valid_inputs = torch.FloatTensor(utl.inner_concatenation(user_attributes, train_0[1]))
valid_outputs = torch.FloatTensor(train_0[1])

test_inputs = torch.FloatTensor(utl.inner_concatenation(user_attributes, test_set))
test_outputs = torch.FloatTensor(test_set)
```

<!-- #region id="a-KippqOyQbh" -->
##### Question 14

14. Initialize the autoencoder.
<!-- #endregion -->

```python id="F07Bdo-xyQbh"
#TO-DO
```

<!-- #region id="z6MteKllyQbj" -->
##### Question 15

15. Code the training phase.
<!-- #endregion -->

```python id="uGREkaBCyQbk"
#TO-DO
```

<!-- #region id="mYhkBa1ryQbm" -->
##### Question 16

16. Calculate the performance on the test set.
<!-- #endregion -->

```python id="G6XQvsfwyQbm"
#TO-DO
```

<!-- #region id="0IwUVsx4yQbo" -->
### 4.1.1 Cold start problem

As previously mentioned, beyond improving the performance of the model according to the chosen metric, the use of socio-demographic variables in the model makes it possible to make recommendations to a new user simply according to its attributes. This modeling can counteract the cold start problem.

<!-- #endregion -->

<!-- #region id="Ef4d7Y_oJ1zb" -->
##### Question 17

1. Set the age, gender and occupation of an individual.

2. Consider that the latter has not yet evaluated any movie.

3. Show her/him, according to the estimated model, the best movie recommendations.



<!-- #endregion -->

```python id="43OfivOOyQbo"
# Question 17.1: 
age = [?]
gender = [?]

occupation = np.zeros(len(occupation_name))

occupation[occupation_name.tolist().index('artist')] = 1

# Question 17.2: 
  #TO-DO

# Question 17.3: 
  #TO-DO
```
