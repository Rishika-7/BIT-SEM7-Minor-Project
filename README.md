## INTRODUCTION
An important part of a company’s HR Department’s work is evaluating resumes in order to shortlist candidates for hiring as employees. It is extremely challenging to go through a huge pile of resumes. These resumes are often heavily populated with irrelevant details, as applicants do not necessarily know what the various employers are seeking. 
In order to simplify this, the NER model can be used to facilitate the process of evaluation of resumes at a quick glance thus the effort required to shortlist candidates from a pile of resumes of applicants can be reduced.
Named Entity Recognition (NER) is a task in which one identifies and categorizes key information which are referred to as entities in text data. This is also referred to as extraction of information or identification of important bits of information from the data. 
NER is a Natural Language Processing problem, its objective is to locate and classify the recognised entities into certain predefined categories. These categories can be person names, organizations, locations, events, time expressions, item quantities, monetary values, numerical data etc. 

## OBJECTIVE
The aim of this project is to create a named entity recogniser, by training and testing a custom model and fine tuning it for achieving better accuracy for the specific domain of resumes. This project will use the Python open-source software library, spaCy.

## Dataset
livecareer.com resume Dataset
A collection of 2400+ Resume Examples taken from livecareer.com for categorizing a given resume into any of the labels defined in the dataset.

## Inside the CSV
ID: Unique identifier and file name for the respective pdf.
Resume_str : Contains the resume text only in string format.
Resume_html : Contains the resume data in html format as present while web scrapping.
Category : Category of the job the resume was used to apply.

## Present Categories
HR, Designer, Information-Technology, Teacher, Advocate, Business-Development, Healthcare, Fitness, Agriculture, BPO, Sales, Consultant, Digital-Media, Automobile, Chef, Finance, Apparel, Engineering, Accountant, Construction, Public-Relations, Banking, Arts, Aviation

## Jobzilla skill patterns
The jobzilla skill dataset is jsonl file containing different skills that can be used to create spaCy entity_ruler. The data set contains label and pattern-> diferent words used to descibe skills in various resume.

## Loading
We load the spaCy model, Resume Dataset, and Jobzilla skills dataset directly into the entity ruler.

## Resume Dataset
Using Pandas read_csv to read the dataset containing text data about Resume.
We are going to randomize Job categories so that 200 samples contain various job categories instead of one and we are going to limit our number of samples to 200 as processing 2400+ takes time.

## Loading spaCy model
To download the spaCy model run python -m spacy en_core_web_lg. Then load spacy model into nlp.

## Entity Ruler
To create an entity ruler we need to add a pipeline and then load the .jsonl file containing skills into the ruler. We have successfully added a new pipeline entity_ruler. Entity ruler helps us add additional rules to highlight various categories within the text, such as skills and job description in our case.

## Skills
We will create two python functions to extract all the skills within a resume and create an array containing all the skills. Later we are going to apply this function to our dataset and create a new feature called skill. This will help us visualize trends and patterns within the dataset.
get_skills is going to extract skills from a single text, and unique_skills will remove duplicates.

## Cleaning Resume Text
We are going to use nltk library to clean our dataset in a few steps:
- We are going to use regex to remove hyperlinks, special characters, or -punctuations.
- Lowering text
- Splitting text into array based on space
- Lemmatizing text to its base form for normalizations
- Removing English stopwords
- Appending the results into an array

## Applying functions
We then apply all the functions we have created previously
- creating Clean_Resume columns and adding cleaning Resume data.
- creating skills columns, lowering text, and applying the get_skills function.
- removing duplicates from skills columns.

## Visualization
Now that we have everything we want, we visualize Job distributions and skill distributions

## Jobs Distribution
Our random 200 samples contain a variety of job categories. Accountants, Business development, and Advocates are the top categories.

## Skills
We use the Deepnote input cell to create category variables and then visualize the distribution of skills based on selected Job Descriptions.

## Entity Recognition
We then display various entities within our raw text by using spaCy displacy.render. 

## Custom Entity Recognition
In our case, we have added a new entity called SKILL and is displayed in gray color. 
- Adding Job-Category into entity ruler.
- Adding custom colors to all categories.
- Adding gradient colors to SKILL and Job-Category

## Match Score
Recruiters can add skills and get a percentage of match skills. This can help them filter out hundreds of Resumes with just one button.
Please add the skills that are required by the job description without space in between commas and it will print out the percentage of match skills within the resume.

## Conclusion
In this project, we have used an entity ruler to create additional entities and then displayed them using custom colors. We have also visualized categories and skills distributions and allowed the user to add resumes directly which includes skills match percentage. Finally, we have used LDA for topic modeling and used pyLDAvis to visualize various topics.

