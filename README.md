![Fabric](https://i.pinimg.com/originals/e3/fa/32/e3fa32b8d27bbb6f39161378dbe9e0c9.gif)

# Identifying-Defects-in-the-Various-Fabrics-using-Convolutional-Neural-Networks
The main objective of this project is to identify whether there is any fault in the fabrics or not, this is done using convolutional neural networks where the model is trained on both defected and non-defect fabric dataset. The dataset was obtained from private data repository and some pre-processing has been performed to make sure the data is clean for modelling.

# Base Paper
+ https://www.researchgate.net/publication/316687014_Fabric_Fault_Detection_Using_Image_Processing

# Algorithm Description
So, we have used Convolutional neural networks to identify whether a person has brain tumour or not, as we all know, how sophisticated CNNs are and how they can learn almost anything like a brain does, this can help us save a lot of time and also giving almost accurate predictions for the disease. As we discussed convolutional neural networks are very sophisticated and more advanced version of neural networks, these are very superior to other neural networks which works better with images and audio/speech input signal. A CNN network comprises of 3 important layers such as a convolutional layer, pooling layer and fully connected layer. we can have as many layers as possible depending on the domain and project we are working on.

**Reference:**

![Neural Network](https://john.sisler.info/wp-content/uploads/sites/2/2018/07/Neural-Network-Diagram.png?4ea638&4ea638)

https://www.ibm.com/cloud/learn/convolutional-neural-networks

# _How to Execute?_
So, before execution we have some pre-requisites that we need to download or install i.e., anaconda environment, python and a code editor.
**Anaconda**: Anaconda is like a package of libraries and offers a great deal of information which allows a data engineer to create multiple environments and install required libraries easy and neat.

**Download link:**

![Anaconda](https://media.giphy.com/media/aO4sY5KYVip8I/giphy.gif)

https://www.anaconda.com/

**Python**: Python is a most popular interpreter programming language, which is used in almost every field. Its syntax is very similar to English language and even children and learning it nowadays, due to its readability and easy syntax and large community of users to help you whenever you face any issues.

**Download link:**

![Python](https://media.giphy.com/media/jJkRqLUoaic9i/giphy.gif)

https://www.python.org/downloads/

**Code editor**: Code editor is like a notepad for a programming language which allows user to write, run and execute program which we have written. Along with these some code editors also allows us to debug, which usually allows users to execute the code line by line and allows them to see where and how to solve the errors. But I personally feel visual code is very good to work with any programming language and makes a great deal of attachment with user.

**Download links:**

![Vs code](https://www.thisprogrammingthing.com/assets/headers/vscode@400.png) ![Pycharm](https://www.esoftner.com/wp-content/uploads/2019/12/PyCharm-Logo.png)

+ https://code.visualstudio.com/Download, 
+ https://www.jetbrains.com/pycharm/download/#section=windows

# How to create a new environment and configure jupyter notebook with it.
Let us define an environment and why we need different environments. An environment is a collection of libraries that are required to run our project. When we already have an environment with the necessary libraries, why do we need a new environment?
To avoid version mismatches, we create a new environment for each project. For example, in your previous project, you used "tf env" with tensorflow 2.4 and keras 2.4, but in your current project, you must use tensorflow 2.6 and keras 2.6. If you continue your project in the "tf env" environment, there will be a version mismatch and you will need to update tensorflow and keras, but this will cause problems with the previous project's execution. To avoid this, we create a new environment with tensorflow 2.6 and keras 2.6 and resume our project.

Let us now see how to create an environment in anaconda.
+ Type “conda create –n <<name_of_your_env>>”
example: conda create -n env
+ It will ask to proceed with the environment location, type ‘y’ and press enter.
+ When you press ‘y’, the environment will be created. To activate your environment type conda activate <<your_env_name>> . E.g., conda activate myenv.
+ You can see that the environment got changed after conda activate myenv line. It changed from “base” to “myenv” which means you are now working in “myenv” environment.
+ To install a library in your virtual environment type pip install <library_name>.
e.g., pip install pandas
+ Instead of installing libraries one by one you can even install by bunch, i.e., we have a txt file called requirements.tx which consists of all the libraries required to proceed with the project, so we can use it.
+ so, before installing requirements.txt, make sure you are in the specific path where your requirements.txt is located, basically this file is located in the folder where our executable files are located, so we need to move to that directory by following command.
**cd A:\folder_name**
+ Here A -> drive, folder name -> path where your executable file is saved
+ I go to that file path in anaconda using cd command 
1.	Go to drive where your project file is.
2.	Go to the path of your project using cd <path>
3.	Type pip install –r requirements.txt 
+ And all your required libraries will be downloaded and you can start your project.
+ But if you want to use jupyter notebook on the new environment you have to set it up for the new environment.
+ After you have installed all the libraries and created an environment, you need an editor to run the code, that is starting jupyter notebook, as soon as you enter jupyter notebook in the terminal you will definitely get this error. “Jupiter” is not recognized as an internal or external command.
So, to solve it it we have 2 commands.
1.	conda install –c conda-forge jupyterlab
2.	conda install –c anaconda python
Now you are ready to use jupyter on this environment and start with your project!

### **Credits to owner who gave detailed explanation of installation procedure.**
+ https://github.com/PaVaNTrIpAtHi
+ https://www.linkedin.com/in/pavan-tripathi-3993641a1/

# Steps to Run the code.
**Note:** Make sure you have added path while installing the software’s.
1. Install the prerequisites mentioned above.
2. open anaconda prompt and create a new environment.
  - conda create -n "env_name"
  - conda activate "env_name"
3. Install necessary libraries from requirements.txt file provided.
4. Run pip install -r requirements.txt or conda install requirements.txt (Requirements.txt is a text file consisting of all the necessary libraries required for executing this python file. If it gives any error while installing libraries, you might need to install them individually.)
5. Run main.ipynb runt eh final code, and make sure to change the path of the model and image folders. You can even see preprocess.ipynb to get a feel of how the images are pre-processed.

# Data Description
So, the dataset in the project was collected from a private repository and some were collected from Kaggle data repository. It consists of two classes of images each class consists of more than 100 images. Defected fabric and normal. Below are some sample images of 2 classes.

**Credits to the owners of the dataset.**

NORMAL

![Normal_fabric](https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse1.mm.bing.net%2Fth%3Fid%3DOIP.zNLqVGx1oDkoPaPm8iku8gAAAA%26pid%3DApi&f=1)

DEFECT

![Defect](https://www.bmsvision.com/sites/default/files/Defect_Local.png)

# _Issues Faced._
1. Pre-processing and training the model takes lot of time since the dataset is being trained on huge amount of data.
2. We might face an issue while installing specific libraries.
3. Make sure you have the latest version of python, since sometimes it might cause version mismatch.
4. Adding path to environment variables in order to run python files and anaconda environment in code editor, specifically in visual studio code.
5. Refer to the Below link to get more details on installing python and anaconda and how to configure it.
+ https://techieyantechnologies.com/2022/06/get-started-with-creating-new-environment-in-anaconda-configuring-jupyter-notebook-and-installing-libraries-using-requirements-txt-2/
6. Make sure to change the paths of the model and dataset in the code.

# _Note:_
**All the required data hasn't been provided over here. Please feel free to contact me for dataset or any issues.**

### **Let’s Connect**
https://www.linkedin.com/in/abhinay-lingala-5a3ab7205/

![Connect](https://media0.giphy.com/media/l9ZWFT0IjZbzi/200w.gif)

# ___**Yes, you now have more knowledge than yesterday, Keep Going.**___
![Congrats](https://media1.tenor.com/images/43d14c43c382da190601052f59a87072/tenor.gif?itemid=4115466)
