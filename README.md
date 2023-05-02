Download Link: https://assignmentchef.com/product/solved-cpsc-585-project-2-an-mlp-for-emnist-letters
<br>
This project has the same platform requirements as <u>​</u><a href="https://docs.google.com/document/d/1G20FKK8uuAnuuh-YTyaV8hTWkQkf8PxNRwKlrDDie8Y/edit?usp=sharing">Project 1</a><u>​</u><a href="https://docs.google.com/document/d/1G20FKK8uuAnuuh-YTyaV8hTWkQkf8PxNRwKlrDDie8Y/edit?usp=sharing">,</a> but access to a GPU or TPU will speed up training significantly. If you do not have access to a physical machine with a GPU, there are cloud services that offer free access to GPU or TPU-enabled versions of Jupyter, including ​<a href="https://colab.research.google.com/">Google Colaboratory</a>​ (<a href="https://colab.research.google.com/notebooks/gpu.ipynb">GP</a><u>​    </u><a href="https://colab.research.google.com/notebooks/gpu.ipynb">U</a><u>​</u><a href="https://colab.research.google.com/notebooks/gpu.ipynb">,</a> <u>​</u><a href="https://colab.research.google.com/notebooks/tpu.ipynb">TPU</a>​), <u>​</u><a href="https://www.kaggle.com/docs/notebooks">Kaggle Notebooks</a>​ (<u>​</u><a href="https://www.kaggle.com/docs/efficient-gpu-usage">GPU</a><u>​</u><a href="https://www.kaggle.com/docs/efficient-gpu-usage">,</a> ​<a href="https://www.kaggle.com/docs/tpu">TPU</a>​), and ​<a href="https://medium.com/@HelloPaperspace/introducing-gradient-community-notebooks-easily-run-ml-notebooks-on-free-gpus-f3fa36336b3c">Gradient</a> <a href="https://medium.com/@HelloPaperspace/introducing-gradient-community-notebooks-easily-run-ml-notebooks-on-free-gpus-f3fa36336b3c">Community Notebooks</a>​.

You will need <u>​</u><a href="https://keras.io/">Keras</a><u>​</u> and <u>​</u><a href="https://www.tensorflow.org/">TensorFlow</a><u>​</u><a href="https://www.tensorflow.org/">.</a> While Keras is capable of using multiple ​<a href="https://keras.io/backend/">backends</a><u>​</u><a href="https://keras.io/backend/">,</a> the <a href="https://keras.io/#multi-backend-keras-and-tfkeras">current recommendation</a><u>​</u> from the Keras team is to use the ​<a href="https://www.tensorflow.org/guide/keras">tf.keras</a>​ module built into TensorFlow.

You may also wish to use other Python libraries such as <u>​</u><a href="https://scikit-learn.org/">scikit-learn</a>​ and <u>​</u><a href="https://pandas.pydata.org/">pandas</a>​.

Examples in the Nielsen book use the MNIST dataset of handwritten digits, which is also <a href="https://www.tensorflow.org/api_docs/python/tf/keras/datasets/mnist">included with Keras</a>​. Since the dataset is reasonably small and easily available, you may wish to use this dataset while setting up and getting familiar with Keras and your notebook environment.

Keras includes several examples using MNIST, in particular ​<a href="https://github.com/keras-team/keras/blob/master/examples/mnist_mlp.py">mnist_mlp.py</a><u>​</u><a href="https://github.com/keras-team/keras/blob/master/examples/mnist_mlp.py">.</a> Try running this code (but don’t forget to switch to ​tf.keras​ instead) and verifying that you can match the accuracy of 98.40% claimed in the comments.

Compare the network architecture in ​mnist_mlp.py​ with the architecture that Nielsen describes in the section on ​<a href="http://neuralnetworksanddeeplearning.com/chap3.html#dropout_explanation">dropout</a><u>​</u> in Chapter 3 of the Nielsen book

<h1>Dataset</h1>

The Extended MNIST or <u>​</u><a href="https://www.nist.gov/itl/products-and-services/emnist-dataset">EMNIST dataset</a><u>​</u> expands on the original MNIST, adding handwritten letters as well as additional samples of handwritten digits. There are several “splits” of the data by various characteristics. We will be using the “EMNIST Letters” dataset, which contains data split into 26 classes, one for each letter in the English alphabet.

Extracting and loading the dataset

Download the ​<a href="http://www.itl.nist.gov/iaui/vip/cs_links/EMNIST/matlab.zip">Matlab format dataset</a>​ and extract ​matlab/emnist-letters.mat​. This file can be opened by ​<a href="https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.loadmat.html">scipy.io.loadmat()</a><u>​</u><a href="https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.loadmat.html">,</a> but you will need some guidance in figuring out how to navigate the structure. See <u>​</u><a href="https://stackoverflow.com/a/53547262">this answer</a>​ on StackOverflow for details on retrieving the training, validation, and test sets.

Dataset classes

The EMNIST Letters dataset folds together both upper- and lowercase letters into a single class. The ​data[‘mapping’]​ field maps from class numbers to ASCII codes of the corresponding letters. For example, class 1 maps to ASCII codes 65 and 97 (​’A’​ and ​’a’​). This may affect your network design.

<h1>An MLP for EMNIST Letters</h1>

Begin by applying the network architecture from ​mnist_mlp.py​ to the EMNIST Letters data, modifying the number of classes. What accuracy do you achieve? How does this compare with the accuracy for MNIST?

Keeping the same number of <u>​</u><a href="https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense">Dense</a><u>​</u> layers in the network (i.e. a standard MLP with one hidden layer), modify the architecture to improve the accuracy. You will need to decide on an appropriate number of neurons in the hidden layer. Keep in mind that:

<ul>

 <li>There are 26 classes rather than 10, so you will likely need a larger hidden layer than the network for recognizing digits.</li>

 <li>In addition to having more classes, the EMNIST Letters data mixes upper- and lowercase letters within each class, so even with enough neurons in the hidden layer, your accuracy is likely to be lower. See the details in the <u>​</u><a href="https://arxiv.org/abs/1702.05373">EMNIST paper</a>​ for the kind of performance you might reasonably expect.</li>

</ul>

Once you have settled on the size of the hidden layer, use the techniques you learned in

Chapter 3 of the Nielsen book to obtain the highest accuracy you can on the validation set.

When finished, evaluate your results on the test set. Compare the performance on the test set of your original and final networks for EMNIST Letters.