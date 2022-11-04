## INTRODUCTION
An important part of a company’s HR Department’s work is evaluating resumes in order to shortlist candidates for hiring as employees. It is extremely challenging to go through a huge pile
of resumes. These resumes are often heavily populated with irrelevant details, as applicants do not necessarily know what the various employers are seeking.
In order to simplify this, the NER model can be used to facilitate the process of evaluation of resumes at a quick glance thus the effort required to shortlist candidates from a pile of resumes of
applicants can be reduced.
Named Entity Recognition (NER) is a task in which one identifies and categorizes key information which are referred to as entities in text data. This is also referred to as extraction of
information or identification of important bits of information from the data. NER is a Natural Language Processing problem, its objective is to locate and classify the
recognised entities into certain predefined categories. These categories can be person names, organizations, locations, events, time expressions, item quantities, monetary values, numerical
data etc.

## PROBLEM STATEMENT
The sheer quantity of applicants for each job opening is in large numbers and shortlisting resumes which are an appropriate fit for the role is a tedious task. The resumes contain a lot of
information, a large portion of it is irrelevant in order to select the candidate for the subsequent process of hiring. The recruiters only seek for important bits of information which indicate that
the applicant is a good fit for the role. A summary of the resume which only contains important chunks of information is required.

## OBJECTIVE
To train and test a custom named entity recogniser model for the domain of resumes.

## HARDWARE SPECIFICATION
The recommended hardware specifications are a minimum of 8GB RAM and a CPU of 7th generation (Intel Core i7 processor) or higher and a dedicated graphics card. The software
requires no specific hardware component/ interface.

## SOFTWARE SPECIFICATION
Operating System – Windows/ Linux, Jupyter Notebook/ Colab Notebook, Python 3, spaCy, sci-kit learn libraries.

## METHODOLOGY
#### Dataset:
The foremost task is the creation of manually annotated training data in order to train the model. To achieve this a total of 220 resumes were downloaded from an online jobs platform and
uploaded to an online annotation tool to manually annotate the documents by an open sourced third party. The tool parses the documents automatically and allows annotations of relevant
entities specific to the use case, which is resume in our case, and generates json formatted training data with each line containing the text corpus along with the annotations. We train the model with
200 resume data and test it on 20 resume data.
#### Training the Model:
In this project we use python’s spaCy module for training the NER model. spaCy’s models are statistical and every “decision” is a prediction, say for example, the part-of-speech tag which is to
be assigned, or recognising a named entity – is a prediction. These predictions are based on the examples that the model has seen during training.
The model is then shown the unlabelled text data and will make a prediction based on the training. We are aware of the correct answer thus, we can give the model feedback on its prediction in the
form of an error gradient of the loss function which calculates the difference between the training example and the expected output, the greater this difference is, the more significant the gradient
and the updates to the model.
When we train a model, we do not want it to memorize our examples, in which case it will only be able to predict correctly for the samples that were fed to the model during training, instead we
want the model to come up with a theory that can be generalized across other examples so that it can “learn” and apply it to new unseen data and predict with a considerable accuracy. In order to
understand this let us look at an example, we do not want the model to learn that the one instance of “Amazon” in the sample resume is a company, we want it to learn that “Amazon”, in contexts
like this, is most likely a company. In order to tune the accuracy, we process our training examples in batches, and experiment with minibatch sizes and dropout rates.
Evidently it is not enough to only show a model a single example once, specifically in the case where the training examples are less in number, thus we’ll have to train for a number of iterations.
At each iteration, the training data is shuffled to ensure that the model does not make any generalizations based on the order of examples in the training dataset.
Another technique to improve the learning results is to set a dropout rate, which is the rate at which there is randomly a “drop” of individual features and representations. This makes it harder
for the model to memorize the training data. For example, a 0.25 dropout means that each feature or internal representation has a 1/4 likelihood of being dropped. Here we train the model for 10
epochs and keep the dropout rate as 0.2.

## PROJECT SPECIFICATION
In order to build this project which includes training a custom pre-trained convolutional neural network in the specific domain of resumes in order to fine tune it so that the important bits of
information can be identified and summarized to provide a glance over the resumes and reduce the workload of HR personnels the following tech stack has been employed.
#### Tech Stack

Python 3
Python is a high-level, object-oriented, general-purpose, interactive, and interpreted programming language. Between 1985 and 1990, Guido van Rossum designed it. Python source code is also
accessible under the GNU General Public License, just like Perl (GPL). Python is not a snake; it is named after the TV programme Monty Python's Flying Circus.
2008 saw the introduction of Python 3.0. Although this version was designed to be incompatible with earlier versions, many of its crucial features have now been backported to work with version
2.7. This tutorial provides sufficient background knowledge for the Python 3 programming language. Python is a powerful, interactive, object-oriented, and interpreted scripting language. Python has
been created to be very readable. It has fewer syntactic structures than other languages and typically employs English keywords rather than punctuation.

spaCy
SpaCy is a Python package for high-level Natural Language Processing (NLP), which is open-source and free. It assists you in creating programmes that process and "understand" massive amounts of text
because it was created expressly for use in production environments. Systems for information extraction or natural language understanding can be created using it.
The training pipelines of spaCy can be set up as Python packages. They are therefore a part of your application, just like any other module, in this sense. They can be listed as a dependent in your
requirements.txt and are versioned. Installing trained pipelines can be done manually, using pip, from a local directory or a download URL. Their information is spread all across your file system.
SpaCy has a lightning-fast statistical entity detection system that labels token spans that are adjacent to one another. Companies, regions, organizations, and items are just a few of the named and numeric
entities that the default trained pipelines can recognise. The entity recognition system can be expanded with any classes, and new examples can be added to the model.
A named entity is a "real-world item" that has been given a name, such as a person, nation, thing, book, etc. By requesting a prediction from the model, spaCy is able to identify several kinds of named entities
in a document. This doesn't always work precisely and may require some adjustment later, depending on your use case, because models are statistical and heavily depend on the instances they were trained on.

scikit-learn
Scikit-learn is probably the most useful library for machine learning in Python. The sklearn library contains a lot of efficient tools for machine learning and statistical modeling including classification,
regression, clustering and dimensionality reduction. We have utilized the functionalities of classification report, precision recall fscore support, accuracy
score etc. A classification report is a performance evaluation metric in machine learning. It displays the model’s precision, recall, F1 score and support. It provides a better understanding of the overall performance of
the trained model. 
Precision - Precision is defined as the ratio of true positives to the sum of true and false positives.
Recall - Recall is defined as the ratio of true positives to the sum of true positives and false negatives.
F1 Score - The F1 is the weighted harmonic mean of precision and recall. The closer the value of the F1 score is to 1.0, the better the expected performance of the model is.
Support - Support is the number of actual occurrences of the class in the dataset. It doesn’t vary between models, it just diagnoses the performance evaluation process.

Jupyter Notebook
Project Jupyter is a project with goals to develop open-source software, open standards, and services for interactive computing across multiple programming languages. The Jupyter Notebook is a web-based
interactive computing platform. The notebook combines live code, equations, narrative text, visualizations, etc. Jupyter Notebook allows users to compile all aspects of a data project in one place making it easier to show the entire process of a project to your intended audience. Through the web-based application, users
can create data visualizations and other components of a project to share with others via the platform.
