---
layout: default
nav_exclude: true
---

# Exam template for 02476 Machine Learning Operations

This is the report template for the exam. Please only remove the text formatted as with three dashes in front and behind
like:

```--- question 1 fill here ---```

where you instead should add your answers. Any other changes may have unwanted consequences when your report is auto
generated in the end of the course. For questions where you are asked to include images, start by adding the image to
the `figures` subfolder (please only use `.png`, `.jpg` or `.jpeg`) and then add the following code in your answer:

```markdown
![my_image](figures/<image>.<extension>)
```

In addition to this markdown file, we also provide the `report.py` script that provides two utility functions:

Running:

```bash
python report.py html
```

will generate an `.html` page of your report. After deadline for answering this template, we will autoscrape
everything in this `reports` folder and then use this utility to generate an `.html` page that will be your serve
as your final handin.

Running

```bash
python report.py check
```

will check your answers in this template against the constrains listed for each question e.g. is your answer too
short, too long, have you included an image when asked to.

For both functions to work it is important that you do not rename anything. The script have two dependencies that can
be installed with `pip install click markdown`.

## Overall project checklist

The checklist is *exhaustic* which means that it includes everything that you could possible do on the project in
relation the curricilum in this course. Therefore, we do not expect at all that you have checked of all boxes at the
end of the project.

### Week 1

* [x] Create a git repository
* [x] Make sure that all team members have write access to the github repository
* [x] Create a dedicated environment for your project to keep track of your packages
* [x] Create the initial file structure using cookiecutter
* [x] Fill out the `make_dataset.py` file such that it downloads whatever data you need and
* [x] Add a model file and a training script and get that running
* [x] Remember to fill out the `requirements.txt` file with whatever dependencies that you are using
* [x] Remember to comply with good coding practices (`pep8`) while doing the project
* [x] Do a bit of code typing and remember to document essential parts of your code
* [x] Setup version control for your data or part of your data
* [x] Construct one or multiple docker files for your code
* [x] Build the docker files locally and make sure they work as intended
* [x] Write one or multiple configurations files for your experiments
* [x] Used Hydra to load the configurations and manage your hyperparameters
* [ ] When you have something that works somewhat, remember at some point to to some profiling and see if
      you can optimize your code
* [x] Use Weights & Biases to log training progress and other important metrics/artifacts in your code. Additionally,
      consider running a hyperparameter optimization sweep.
* [x] Use Pytorch-lightning (if applicable) to reduce the amount of boilerplate in your code

### Week 2

* [x] Write unit tests related to the data part of your code
* [x] Write unit tests related to model construction and or model training
* [x] Calculate the coverage.
* [x] Get some continuous integration running on the github repository
* [x] Create a data storage in GCP Bucket for you data and preferable link this with your data version control setup
* [x] Create a trigger workflow for automatically building your docker images
* [x] Get your model training in GCP using either the Engine or Vertex AI
* [x] Create a FastAPI application that can do inference using your model
* [ ] If applicable, consider deploying the model locally using torchserve
* [x] Deploy your model in GCP using either Functions or Run as the backend

### Week 3

* [ ] Check how robust your model is towards data drifting
* [ ] Setup monitoring for the system telemetry of your deployed model
* [ ] Setup monitoring for the performance of your deployed model
* [ ] If applicable, play around with distributed data loading
* [ ] If applicable, play around with distributed model training
* [ ] Play around with quantization, compilation and pruning for you trained models to increase inference speed

### Additional

* [ ] Revisit your initial project description. Did the project turn out as you wanted?
* [ ] Make sure all group members have a understanding about all parts of the project
* [ ] Uploaded all your code to github

## Group information

### Question 1
> **Enter the group number you signed up on <learn.inside.dtu.dk>**
>
> Answer:

--- 37 ---

### Question 2
> **Enter the study number for each member in the group**
>
> Example:
>
> *sXXXXXX, sXXXXXX, sXXXXXX*
>
> Answer:

--- s213637, s212599, s220285, s220279 ---

### Question 3
> **What framework did you choose to work with and did it help you complete the project?**
>
> Answer length: 100-200 words.
>
> Example:
> *We used the third-party framework ... in our project. We used functionality ... and functionality ... from the*
> *package to do ... and ... in our project*.
>
> Answer:

--- We used the PyTorch Image Models (TIMM) framework. This framewrok contained an implementation of the resnet18 model in pytorch with pretrained weights. To use the model for our application, we used the model with the pretrained wieghts, but cut off the last layer. In it's place we put a fully connected layer with only 29 nodes, as these were the possible outputs we expected from the model. Additionally, to best utilized these pretrained wieght, and speed up training, we froze all the weights and biases except for the newly added fully connected layer. Thus, only the weights were trained during the training process.

Additionally we used the Pytorch Lightning framework to quickly set up our model and the training loop. ---

## Coding environment

> In the following section we are interested in learning more about you local development environment.

### Question 4

> **Explain how you managed dependencies in your project? Explain the process a new team member would have to go**
> **through to get an exact copy of your environment.**
>
> Answer length: 100-200 words
>
> Example:
> *We used ... for managing our dependencies. The list of dependencies was auto-generated using ... . To get a*
> *complete copy of our development enviroment, one would have to run the following commands*
>
> Answer:

--- We used anaconda to manage our environments. We created a virtual environment where we installed our required packages with pip install.
In order for all team members to use the same environment, we created an environment.yaml file. As not all required dependencies were included with the normal command conda env export and some issues were faced between different operating systems, we used a script called conda_env_export.py to save the environment correctly. To keep track of all the required packages of the project, we created a requirements.txt file. This file is created using pipreqs. If a new user wants to use our project code, the repository should be cloned and an anaconda environment should be created and activated with: conda env create -f environment.yml
conda activate MLOPS_project
After that, all packages need to be installed with the command: pip install -r requirements.txt
Then, everything is ready to use.---

### Question 5

> **We expect that you initialized your project using the cookiecutter template. Explain the overall structure of your**
> **code. Did you fill out every folder or only a subset?**
>
> Answer length: 100-200 words
>
> Example:
> *From the cookiecutter template we have filled out the ... , ... and ... folder. We have removed the ... folder*
> *because we did not use any ... in our project. We have added an ... folder that contains ... for running our*
> *experiments.*
> Answer:

--- From the cookiecutter template we have used the src/data folder for all the scripts redarding the creation and processing of the dataset. The folder src/models is used for the definition of our model and the training and predict scripts. The src/features and src/visualization folders are not used since no features of the data are extracted before the training and no extra visualizations functions are created. In data/raw we are storing the raw compressed dataset that can be pulled from dvc. When running the make_dataset.py script, the data is extracted in data/interim and finally processed to torch.tensors, saved in data/processed. The folder data/external is not used. The directory reports is used for the report hand in and for figures. In models, the trained model files are stored. The folders notebooks and references are not used since no belonging files where necessary. We have added...---

### Question 6

> **Did you implement any rules for code quality and format? Additionally, explain with your own words why these**
> **concepts matters in larger projects.**
>
> Answer length: 50-100 words.
>
> Answer:

--- There are a number of rules and checks that we added as pre-commit hooks. We have a number of linters, that format the code nicely, such as black, trailing-whitespace, pycln, and isort. After this, we run a few checks, such as mypy, flake8, and interrogate to make sure the code is formatted correctly and also documented. We additionally created a custom hook that auto generates the latest version of the requirements file. These are all important to streamline the process of developing, commiting and getting started with the process. e.g. One doesn't have to pay attention to formatting. ---

## Version control

> In the following section we are interested in how version control was used in your project during development to
> corporate and increase the quality of your code.

### Question 7

> **How many tests did you implement?**
>
> Answer:

--- We implemented a test to check if the make_dataset.py file creates data of the correct shape. Also, we implemented a test of the model.py to check the output of the model after passing dummy data. ---

### Question 8

> **What is the total code coverage (in percentage) of your code? If you code had an code coverage of 100% (or close**
> **to), would you still trust it to be error free? Explain you reasoning.**
>
> **Answer length: 100-200 words.**
>
> Example:
> *The total code coverage of code is X%, which includes all our source code. We are far from 100% coverage of our **
> *code and even if we were then...*
>
> Answer:

--- The total coverage of our code is 44%, which includes.... This and that is missing.... ---

### Question 9

> **Did you workflow include using branches and pull requests? If yes, explain how. If not, explain how branches and**
> **pull request can help improve version control.**
>
> Answer length: 100-200 words.
>
> Example:
> *We made use of both branches and PRs in our project. In our group, each member had an branch that they worked on in*
> *addition to the main branch. To merge code we ...*
>
> Answer:

--- During the course of the project we created a number of branches. This was espacially useful for parallelizing the implementation of different features. For example, one of us had the task of making the dateset, while another one of us was responsible for the model creation. We strived to keep the master branch clean, and only update it using pull requests, but towards the end we did end up pushing directly to master at times. Overall we all gained a higher proficiency in using version control systems, and towards the end could readily create and merge branches with no hassle. Whenever a feature branch was completed, we used Pull Requests push the changes to the master branch, when this happened the automated github workflows ran to ensure the unittests were all passing. ---

### Question 10

> **Did you use DVC for managing data in your project? If yes, then how did it improve your project to have version**
> **control of your data. If no, explain a case where it would be beneficial to have version control of your data.**
>
> Answer length: 100-200 words.
>
> Example:
> *We did make use of DVC in the following way: ... . In the end it helped us in ... for controlling ... part of our*
> *pipeline*
>
> Answer:

--- We used DVC to store and version control our data. In that way, everyone from the team can push and pull the data with dvc from the Google S3 bucket and access it. Since the data.dvc file is pushed to the project's github and tagged with the data version, we can access the data from older versions. In case accidentaly data gets lost or data is updated, the old files can be restored. At the same time, it is useful that the data is stored in a compressed way in our google cloud bucket instead of pushing big data files to the github repository. ---

### Question 11

> **Discuss you continues integration setup. What kind of CI are you running (unittesting, linting, etc.)? Do you test**
> **multiple operating systems, python version etc. Do you make use of caching? Feel free to insert a link to one of**
> **your github actions workflow.**
>
> Answer length: 200-300 words.
>
> Example:
> *We have organized our CI into 3 separate files: one for doing ..., one for running ... testing and one for running*
> *... . In particular for our ..., we used ... .An example of a triggered workflow can be seen here: <weblink>*
>
> Answer:

--- At the moment we have on github actions workflow. Here, we run the previously written tests, and evaluates them on the codebase. This runs on push events found in the master, main, and make_data branches. We have setup the corresponding .yaml file to spin up a ubuntu-latest machine running python 3.10. Following this, the required modules as described in the requirements.txt and requirements_test.txt are intalled. The requirements were separated into these two files, because some modules are only required for testing and would thus make the install size of those only wanting to use the repository for production. Following this, the github action executes the tests found in the test directory with pytest. While we did not use caching in our workflows, we recognize the utility and efficiency increase it would bring when runnning our actions. This would come in exceptional useful for use as pytorch is a huge dependency, and installing it everytime has a huge time overhead. This workflow is found described by the test.yaml file in the github/workflows directory. An interesting thing we found when setting up the worflow is the string parsing of yaml files. As we were using python 3.10 we set the value of the python version to 3.10, without quatiation marks. The yaml parser thought this is a number, so it took a few minutes for us to realize that this was the problem. This was easily solved ---

## Running code and tracking experiments

> In the following section we are interested in learning more about the experimental setup for running your code and
> especially the reproducibility of your experiments.

### Question 12

> **How did you configure experiments? Did you make use of config files? Explain with coding examples of how you would**
> **run a experiment.**
>
> Answer length: 50-100 words.
>
> Example:
> *We used a simple argparser, that worked in the following way: python my_script.py --lr 1e-3 --batch_size 25*
>
> Answer:

--- We used hydra to organize our training configurations. For each configuration, we created a exp.yaml file where the hyperparameters are stored. These parameters are passed to the training script with hydra through the default_config.yaml. In this file, the chosen experiment configuration is defined, where exp1 is default. If the user wants to choose another configuration, it can be passed using the agparser in the command line: python src/models/train_model.py experiment=exp2 ---

### Question 13

> **Reproducibility of experiments are important. Related to the last question, how did you secure that no information**
> **is lost when running experiments and that your experiments are reproducible?**
>
> Answer length: 100-200 words.
>
> Example:
> *We made use of config files. Whenever an experiment is run the following happens: ... . To reproduce an experiment*
> *one would have to do ...*
>
> Answer:

--- As mentioned in the previous questions, we made use of config files using the hydra framework and saved the configurations using the weights and bias framework. So, whenever an experiment is run the following happens: the experiment reads the configuration from the configuration file, then the configuration is saved in wandb, where it can be viewed together with the experiment's results. To reproduce an experiment one would have to check the wandb experiment and use the same configurations in the config file. Additionally, during the start of the training, we set the torch seed to zero, so all the random initializations have the same start and the experiment can be reproducible. ---

### Question 14

> **Upload 1 to 3 screenshots that show the experiments that you have done in W&B (or another experiment tracking**
> **service of your choice). This may include loss graphs, logged images, hyperparameter sweeps etc. You can take**
> **inspiration from [this figure](figures/wandb.png). Explain what metrics you are tracking and why they are**
> **important.**
>
> Answer length: 200-300 words + 1 to 3 screenshots.
>
> Example:
> *As seen in the first image when have tracked ... and ... which both inform us about ... in our experiments.*
> *As seen in the second image we are also tracking ... and ...*
>
> Answer:

--- As seen in the first image, we have tracked the validation and training loss as well as the training and validation accuracy. which informs us about how the training of the model is progressing, by showing how the difference between the model's predicted values and the ground truth, which we get with the loss function, decreases along the training. The same can be said about the accuracy, which we check in the training set, where we see that the model is learning from the dataset it is being trained on, but also on the validation set, where we can see how accurate the model is in data that it has not seen during training.

As seen in the second image we are also tracking the confusion matrix in the validation set at the end of each epoch which is a better visualization of how the model is making predictions and comparing it with the true label of the input. So we can see, in each of the experiments, how the confusion matrix changed in every epoch during the training. The matrix can show us which labels have higher or lower accuracy, and also which labels the model gets more "confused" about.

<img width="1440" alt="Screenshot 2023-01-19 at 10 55 17" src="https://user-images.githubusercontent.com/75242605/213412289-138d4d5e-6a4f-4d03-b9d0-3d46163e8c75.png">
<img width="1105" alt="Screenshot 2023-01-19 at 10 58 21" src="https://user-images.githubusercontent.com/75242605/213412594-699b77eb-9ad5-4cd9-b887-4fac25c7d642.png"> ---

### Question 15

> **Docker is an important tool for creating containerized applications. Explain how you used docker in your**
> **experiments? Include how you would run your docker images and include a link to one of your docker files.**
>
> Answer length: 100-200 words.
>
> Example:
> *For our project we developed several images: one for training, inference and deployment. For example to run the*
> *training docker image: `docker run trainer:latest lr=1e-3 batch_size=64`. Link to docker file: <weblink>*
>
> Answer:

--- For our project we developed several images. One is used for local training ([local dockerfile](https://github.com/AnnaGr-Git/asl_alphabet_classification/blob/master/trainer_local.dockerfile)), one for pushing the image to a GCP bucket to use it for training with Vertex AI ([cloud dockerfile](https://github.com/AnnaGr-Git/asl_alphabet_classification/blob/master/trainer_cloud.dockerfile)), and one for creating the FastAPI app ([FastAPI dockerfile](https://github.com/AnnaGr-Git/asl_alphabet_classification/blob/master/fastapi_app.dockerfile)). The local file can be used from developers who would like to create their own local docker image to share it. The cloud file gets triggered every time changes get pushed to main, so always the newest docker image is available. The image always gets the newest version of the data by `dcv pull`, preprocesses the data and runs a training. This image can be accessed a described in the readme, and can be used for training with Vertex AI. For using this image for training locally, do `docker run trainer:latest`. The FastAPI file gets triggered every time changes get pushed to main, and creates an image that can be accessed by the end user through the API.  ---

### Question 16

> **When running into bugs while trying to run your experiments, how did you perform debugging? Additionally, did you**
> **try to profile your code or do you think it is already perfect?**
>
> Answer length: 100-200 words.
>
> Example:
> *Debugging method was dependent on group member. Some just used ... and others used ... . We did a single profiling*
> *run of our main code at some point that showed ...*
>
> Answer:

--- Debugging varied from group member to group member, and the file in which the problem occured. For python files we mainly relied on the python interpreters verbose error logging. If this wasn't enough we used a compination of debugging as well as the python REPL to gain an understanding of what went wrong. Although when we used click, the debugger wasn't a big help, so we resorted to using the good old method of writing print() statements. Throughout the course of the project we developed better understanding of the python debugger and increased our reliance on it as the days progressed.

We didn't do much in terms of profiling, any implemented speed impovements were the result of obviously not performant code. As at this stage the project is in what would be considered the MVP stage, we didn't place much value in optimizing the code. ---

## Working in the cloud

> In the following section we would like to know more about your experience when developing in the cloud.

### Question 17

> **List all the GCP services that you made use of in your project and shortly explain what each service does?**
>
> Answer length: 50-200 words.
>
> Example:
> *We used the following two services: Engine and Bucket. Engine is used for... and Bucket is used for...*
>
> Answer:

--- We used the following GCP services: Cloud Storage (Buckets), Container Registry, Vertex AI, Cloud Build and Cloud Run. We have three buckets: one to storaé the data that gets accessed by dvc, one for storing the docker images, and one for saving the model files. In the container registry, we have access to the train images and the fastapi images. Vertex AI is used to train models in the cloud as described in the readme of the repository. This accesses the train image stored in the Cloud Storage. Cloud Build is used to create a new docker image and save it to the cloud storage every time someone pushes code to the main branch. The trigger is directly connected to the github repository. Cloud Run is used to create a service that builds a docker container based on the FastAPI docker file in the repository, which allows the end user to access the API through simple http or curl commands. ---

### Question 18

> **The backbone of GCP is the Compute engine. Explained how you made use of this service and what type of VMs**
> **you used?**
>
> Answer length: 50-100 words.
>
> Example:
> *We used the compute engine to run our ... . We used instances with the following hardware: ... and we started the*
> *using a custom container: ...*
>
> Answer:

--- We did not use VM instances by themselves, however, we used Vertex AI which has a Compute Engine VM as the underlying infrastructure. There we used just a CPU VM to save costs while testing. Also, building the images in GCP works based on a VM. We also used a Linux CPU VM. We used a 4 CPU 16GB virtual machine to deploy the model. ---

### Question 19

> **Insert 1-2 images of your GCP bucket, such that we can see what data you have stored in it.**
> **You can take inspiration from [this figure](figures/bucket.png).**
>
> Answer:

--- ![image](https://user-images.githubusercontent.com/75242605/213456976-8bfc0801-f26d-4cb0-b677-10764af7898d.png) ---

### Question 20

> **Upload one image of your GCP container registry, such that we can see the different images that you have stored.**
> **You can take inspiration from [this figure](figures/registry.png).**
>
>
> Answer:

--- ![image](https://user-images.githubusercontent.com/75242605/213458122-757a7d6b-52a8-429d-8105-60d2516a0403.png) ---

### Question 21

> **Upload one image of your GCP cloud build history, so we can see the history of the images that have been build in**
> **your project. You can take inspiration from [this figure](figures/build.png).**
>
> Answer:

--- ![image](https://user-images.githubusercontent.com/75242605/213458561-e2b13504-c08a-40f8-bd7c-e802cd1a3573.png) ---

### Question 22

> **Did you manage to deploy your model, either in locally or cloud? If not, describe why. If yes, describe how and**
> **preferably how you invoke your deployed service?**
>
> Answer length: 100-200 words.
>
> Example:
> *For deployment we wrapped our model into application using ... . We first tried locally serving the model, which*
> *worked. Afterwards we deployed it in the cloud, using ... . To invoke the service an user would call*
> *`curl -X POST -F "file=@file.json"<weburl>`*
>
> Answer:

--- question 22 fill here ---

### Question 23

> **Did you manage to implement monitoring of your deployed model? If yes, explain how it works. If not, explain how**
> **monitoring would help the longevity of your application.**
>
> Answer length: 100-200 words.
>
> Example:
> *We did not manage to implement monitoring. We would like to have monitoring implemented such that over time we could*
> *measure ... and ... that would inform us about this ... behaviour of our application.*
>
> Answer:

--- question 23 fill here ---

### Question 24

> **How many credits did you end up using during the project and what service was most expensive?**
>
> Answer length: 25-100 words.
>
> Example:
> *Group member 1 used ..., Group member 2 used ..., in total ... credits was spend during development. The service*
> *costing the most was ... due to ...*
>
> Answer:

--- During the development of the project, as of January 19, only 3 dollars were spent in total. The service that cost the most was the Cloud storage, with $2.58 being spent, followed by $0.26 with Vertex AI, $0.11 in Cloud run, and $0.05 with the Cloud Machine Learning  Engine and Cloud build combined. ---

## Overall discussion of project

> In the following section we would like you to think about the general structure of your project.

### Question 25

> **Include a figure that describes the overall architecture of your system and what services that you make use of.**
> **You can take inspiration from [this figure](figures/overview.png). Additionally in your own words, explain the**
> **overall steps in figure.**
>
> Answer length: 200-400 words
>
> Example:
>
> *The starting point of the diagram is our local setup, where we integrated ... and ... and ... into our code.*
> *Whenever we commit code and puch to github, it auto triggers ... and ... . From there the diagram shows ...*
>
> Answer:
--- 
![Alt text](figures/overview_mlops.png?raw=true "Flowchart")
The starting point of our diagram is our local setup. Each developer can clone the github project and work on their local machine. For a clear organization of our project, we used the cookie cutter for data science template. In order to run the code and have the same setup on every machine, the packages and python requirements are organized with an anaconda environment. In order to simplify the code and avoid boilerplating, we use the pytorch lightning library. For reproducability, the hyperparameters of every training should be logged. For that reason, hydra is used to read and pass the config-files of different experiments where configuration can be specified or overridden from the command line. In order to log and visualize the training experiments, we used "Weights&Biases" where every team member has access to the shared project. In order to version control the data without pushing everyting to the github repository, dvc is used with the Google Cloud storage. Each version of the data can be pushed and pulled with DVC. For a continuous integration, we added pre-commit hooks where the format of the code is checked and corrected using flake8 and black among others before being committed and pushed to the github repository. Additionally, pytest test functions were created where the code functions are checked for its correctness. Github actions runs this pytests automatically, as soon as there are push or pull requests on the main branch.
Docker images are used to ensure that the project can be run from anywhere, also from the cloud without having the same hardware settings. We create 2 images of our project. One contains all requirements and code for training the model and one image for deploying the model. A cloud build trigger ensures that everytime a new update is pushed to the main branch in github, the two docker images are newly created and saved in the Google Container Registry to guarantee the most recent version. In order to train our model, we use the Google Vertex AI. In order to provide the functionality of our model to any user, we use Cloud Run and FastApI where our model is deployed in the cloud where the user can insert a picture to be classified by the model.
---
### Question 26

> **Discuss the overall struggles of the project. Where did you spend most time and what did you do to overcome these**
> **challenges?**
>
> Answer length: 200-400 words.
>
> Example:
> *The biggest challenges in the project was using ... tool to do ... . The reason for this was ...*
>
> Answer:

--- An interesting we ran into on the first few days of the project was that the free credits on google cloud were suddenly used up. We didn't have any services running that would have warranted such a deduction, so we spent some time looking into what could have caused it. In the end we coudn't definitely prove what becamo of those cloud credits, but we suspect an attack with some relation to crypto mining, as we found crypto related files on our S3 bucket.

On the more technical side, integrating all the component systems into one cohesive product was a complicated enngagement, but fun nonetheless. Individually, each module was a small challenge in itself to sufficiently find our way around it, but that challenge grew exponentially when we attempted to integrate them all with each other. We found that the group aspect of this ocurse was one of the best yet. We absolutely felt like a team in a company, with each of us responsible for a different aspect of the product. Due to this, we could overcome the challenge of integrating the different parts with each other.

Speaking of the group aspect, although we felt that on the technical side we could solve anything, we lacked the proper managment structures to extract the most potential. While we have al used them, none of us took it upon ourselves to create a kanban board for tracking tasks, and at time we didn't utilize git branches to their full potential, often commiting directly to master.

And last but not least, one member of our group overcame great emotional struggle when he got stuck in the elavator of building 310. ---

### Question 27

> **State the individual contributions of each team member. This is required information from DTU, because we need to**
> **make sure all members contributed actively to the project**
>
> Answer length: 50-200 words.
>
> Example:
> *Student sXXXXXX was in charge of developing of setting up the initial cookie cutter project and developing of the*
> *docker containers for training our applications.*
> *Student sXXXXXX was in charge of training our models in the cloud and deploying them afterwards.*
> *All members contributed to code by...*
>
> Answer:

--- question 27 fill here ---
