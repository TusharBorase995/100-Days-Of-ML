# Challenges in Machine Learning

**Lecture:** Challenges in Machine Learning | Problems in Machine Learning
**Source:** CampusX — 100 Days of Machine Learning
**Video:** https://www.youtube.com/watch?v=WGUNAJki2S4

---

## Table of Contents

1. [Data Collection](#1-data-collection)
2. [Insufficient Data](#2-insufficient-data)
3. [Non-Representative Data](#3-non-representative-data)
4. [Poor Quality Data](#4-poor-quality-data)
5. [Irrelevant Features](#5-irrelevant-features)
6. [Overfitting](#6-overfitting)
7. [Underfitting](#7-underfitting)
8. [Software Integration](#8-software-integration)
9. [Offline Learning / Deployment](#9-offline-learning--deployment)
10. [Cost Involved](#10-cost-involved)
11. [Quick Revision](#11-quick-revision)

---

# Introduction

Machine Learning is powerful, but building a successful ML system is not simply about choosing an algorithm and training it.

There are several challenges involved, from **collecting the data** to **deploying the final model as a usable software product**.

The lecture discusses **10 major challenges**:

```text
1. Data Collection
2. Insufficient Data
3. Non-Representative Data
4. Poor Quality Data
5. Irrelevant Features
6. Overfitting
7. Underfitting
8. Software Integration
9. Offline Learning / Deployment
10. Cost Involved
```

---

# 1. Data Collection

## What is the problem?

The first challenge is finding **relevant data** for the problem we want to solve.

When learning Machine Learning in college, getting data often seems easy.

For example, a teacher may simply provide:

```text
data.csv
```

and we can start building the model.

But real-world projects are different.

When a company wants to solve a specific problem, it may not already have a suitable dataset.

Therefore, the first challenge becomes:

> **Where will we get the required data from?**

---

## How can data be collected?

Some possible sources include:

* Web scraping
* APIs
* Government/departmental data
* Existing company data
* Other external sources

For example, suppose we want to build an image-classification model.

We need a large collection of images belonging to the classes we want to identify.

We may have to collect those images ourselves.

### [IMAGE HERE]

*Suggested image: Different sources of ML data → APIs, websites, databases, sensors, etc.*

---

## Why is this difficult?

Collecting a small amount of data may be easy.

The real difficulty is collecting:

* **Large amounts of data**
* **Relevant data**
* **Accurate data**
* **Data suitable for the particular ML problem**

So:

> **Data collection is one of the first major challenges in a real-world ML project.**

---

# 2. Insufficient Data

Suppose we have identified our data source.

The next question is:

> **Do we have enough data?**

Having too little data can make it difficult for a machine learning algorithm to learn properly.

---

## Simple Example

Suppose we train two models to solve the same problem.

### Model A

```text
Training data = 1,000 examples
```

### Model B

```text
Training data = 1,00,000 examples
```

Assuming the data is relevant and of similar quality, the model trained with more data generally has a better opportunity to learn the underlying patterns.

This is why **data quantity matters**.

---

## The problem of labelled data

There is another important issue.

Sometimes we can collect a lot of raw data, but the data is **not labelled**.

For example, suppose we want to build an animal-image classifier.

We can collect thousands of images from the internet.

But the model needs to know which image belongs to which class:

```text
Image 1 → Cat
Image 2 → Dog
Image 3 → Horse
Image 4 → Cat
...
```

Someone has to label these images.

For a very large dataset, manually labelling data can require a huge amount of human effort.

Therefore:

```text
Raw data
   ↓
Collect
   ↓
Label
   ↓
Training dataset
```

Both **collecting data** and **getting labelled data** can be challenging.

---

## Important idea

There is an observation in Machine Learning often summarized as:

> **When you have a very large amount of data, the choice of algorithm can become less important than the data itself.**

This idea is associated with the well-known phrase **“The Unreasonable Effectiveness of Data.”**

However, in many practical situations, we still don't have unlimited data.

---

# 3. Non-Representative Data

Suppose we have solved the previous problem.

We now have **sufficient data**.

But there is another question:

> **Does our data actually represent the population or situation we are trying to model?**

Having a large dataset is not enough.

The data must be **representative**.

---

## Cricket Example

Suppose we want to predict:

> **Which country will win the Cricket World Cup?**

We decide to conduct a survey.

Imagine that we ask only people from India:

```text
1000 Indians
```

and ask:

> "Which country do you think will win?"

A large number of people may say:

```text
India
```

But can we use this survey to conclude that India is actually the most likely winner?

No.

The sample is biased because we only asked people from one country.

---

## What should we do?

We should collect opinions from people belonging to different countries.

For example:

```text
100 Indians
100 Pakistanis
100 Australians
100 English
100 South Africans
...
```

Now the sample has a better representation of different groups.

---

## Sampling Bias

When the way we collect our sample causes the sample to systematically favor a particular group or outcome, we have **sampling bias**.

### Example

```text
Population
   ↓
People from many countries

Bad sample
   ↓
Mostly Indians
   ↓
Biased result
```

The important point is:

> **Even a very large dataset can be problematic if it is not representative.**

---

## Two separate questions

When collecting data, always ask:

### Question 1

**Do I have enough data?**

→ Data quantity.

### Question 2

**Does my data represent the real-world population/problem?**

→ Data representation.

Both are important.

---

### [IMAGE HERE]

*Suggested image: Representative sample vs biased sample.*

---

# 4. Poor Quality Data

Suppose we have:

* Enough data
* Representative data

There is still another problem:

> **Is the data of good quality?**

Data may contain:

* Incorrect values
* Missing values
* Noise
* Incorrect formats
* Other inconsistencies

If poor-quality data is directly given to a machine learning algorithm, the algorithm may not be able to learn the desired patterns properly.

---

## Example

Suppose a dataset contains:

```text
Age
21
23
25
27
-500
29
```

The value:

```text
Age = -500
```

is clearly problematic.

Similarly, a dataset may contain missing values:

```text
Age
21
23
NaN
27
29
```

Such issues need to be handled before using the data.

---

## Data Cleaning

Before training a model, we generally need to clean the dataset.

This can involve:

```text
Raw Data
   ↓
Find errors
   ↓
Handle missing values
   ↓
Handle incorrect values
   ↓
Correct formats
   ↓
Clean Data
```

The lecture emphasizes that a significant amount of time in ML projects can be spent cleaning and preparing data.

---

## Important principle

> **If the input data is poor, the machine learning model will also struggle.**

This is commonly expressed as:

> **Garbage In → Garbage Out**

In other words:

```text
Poor data
   ↓
Poor learning
   ↓
Poor predictions
```

---

# 5. Irrelevant Features

Another challenge is having **irrelevant features** in our dataset.

## What is a feature?

A feature is an input variable used by the machine learning model.

For example, if we want to predict something about a person, our dataset could contain:

```text
Age
Weight
Height
Nationality
```

Each of these can be considered a feature.

---

## What is an irrelevant feature?

An irrelevant feature is a feature that does not meaningfully contribute to solving the prediction problem.

For example, suppose a feature has no useful relationship with the target.

Giving such information to the model can make the learning process unnecessarily complicated.

---

## Garbage In → Garbage Out

The same principle applies here.

If we provide irrelevant information to the model, it can negatively affect the resulting model.

Therefore, we should try to provide **useful features**.

---

## Feature Selection

One approach is to remove unnecessary features.

For example:

```text
Feature 1 ✓
Feature 2 ✓
Feature 3 ✓
Feature 4 ✗
Feature 5 ✗
```

We keep the useful features and remove the irrelevant ones.

This process is called **feature selection**.

---

## Main idea

The goal is not:

> Give the model as many features as possible.

The goal is:

> **Give the model useful features that help it learn the required pattern.**

---

# 6. Overfitting

Now we move from problems with the **data** to problems with the **model**.

One of the most important challenges is **overfitting**.

---

## What is Overfitting?

Overfitting occurs when a machine learning model learns the training data **too closely**.

Instead of learning the general pattern, the model becomes overly dependent on the particular training examples.

As a result, it may perform well on the training data but poorly on **new, unseen data**.

---

## Simple Example

Imagine a student who memorizes the exact questions and answers from previous exams.

They perform very well when the same questions appear again.

But if the teacher asks a new question testing the same concept in a different way, they struggle.

Why?

Because they memorized the examples instead of understanding the underlying concept.

That is similar to overfitting.

---

## Machine Learning Example

Suppose the training data contains:

```text
Training examples
A
B
C
D
E
```

The model learns these examples extremely closely.

Now we give it:

```text
New example → F
```

The model may perform poorly because it has not learned the general relationship.

---

## Generalization

The real objective of Machine Learning is **generalization**.

A good model should learn patterns from the training data and use those patterns to make good predictions on new data.

```text
Training Data
     ↓
Learn Pattern
     ↓
Generalize
     ↓
New / Unseen Data
     ↓
Good Prediction
```

Overfitting breaks this process.

---

### [IMAGE HERE]

*Suggested image: A graph showing a model fitting the general trend vs a model fitting individual training points.*

---

## How can overfitting happen?

A model can become too closely fitted to the training data.

The result is:

```text
Training Data
    ↓
Very good performance

New Data
    ↓
Poor performance
```

Therefore:

> **Good training performance alone does not guarantee a good machine learning model.**

---

# 7. Underfitting

Underfitting is essentially the opposite problem.

## What is Underfitting?

Underfitting occurs when the model is **unable to learn the underlying pattern** in the training data.

The model is too simple or otherwise fails to capture the relationship present in the data.

---

## Example

Suppose the actual relationship in the data is complex.

But we use a model that is too simple.

The model cannot capture the actual pattern.

Therefore:

```text
Actual Pattern
      ↓
Complex

Model
      ↓
Too Simple
      ↓
Fails to learn the pattern
```

---

## Overfitting vs Underfitting

### Overfitting

The model learns the training data **too closely**.

```text
Learns too much from training examples
        ↓
Poor generalization
```

### Underfitting

The model does **not learn enough**.

```text
Fails to capture underlying pattern
        ↓
Poor predictions
```

---

### [IMAGE HERE]

*Suggested image: Three curves — underfitting, appropriate fitting, and overfitting.*

---

## The goal

We want a model that learns the underlying pattern properly.

```text
Underfitting ← GOOD FIT → Overfitting
```

The ideal model should generalize well to new data.

---

# 8. Software Integration

Suppose we have successfully trained a machine learning model.

Is the project finished?

**No.**

The model must ultimately become part of a **software product** that users can actually use.

This creates another challenge:

> **How do we integrate the ML model into software?**

---

## Different Platforms

Software can run on different platforms, such as:

```text
Windows
Android
Linux
etc.
```

A machine learning model that works in one environment may not automatically work properly in another environment.

---

## Different Programming Languages

Machine learning models are often developed using technologies such as Python.

But the software in which the model needs to be integrated might use another programming language.

This creates additional integration challenges.

---

## Libraries and Compatibility

Machine learning relies heavily on libraries.

A model may depend on particular:

* Libraries
* Versions
* Runtime environments
* Operating systems

Therefore, making the model work correctly inside the final software product can be difficult.

---

## Main idea

Training the model is only one part of the problem.

```text
Train ML Model
      ↓
Integrate with Software
      ↓
Make it usable by users
```

A model that cannot be successfully integrated into the actual product is not very useful.

---

# 9. Offline Learning / Deployment

After integrating the model, we have to **deploy it** so that users can actually use it.

This creates another challenge.

---

## Deployment

Deployment means making the trained ML model available in a real environment where it can receive inputs and produce predictions.

For example:

```text
User
  ↓
Software
  ↓
Server
  ↓
ML Model
  ↓
Prediction
  ↓
User
```

---

## Real-Time Updates

A machine learning model may need to deal with new data over time.

The real world keeps changing.

Therefore, the model may need to be:

```text
Updated
Retrained
Redeployed
```

when necessary.

---

## Offline Learning

In offline learning, the model is trained using a fixed collection of data.

When new data becomes available, the model does not automatically learn from it.

Instead, we may need to:

```text
New Data
   ↓
Retrain Model
   ↓
New Model
   ↓
Deploy Again
```

This makes maintaining the system more complicated.

---

## Why is deployment difficult?

The challenge is not simply:

> "Can I train the model?"

It is:

> **Can I continuously operate the model as a real software product?**

The system may need:

* Deployment
* Updates
* Monitoring
* Retraining
* Redeployment

---

# 10. Cost Involved

The final challenge discussed in the lecture is **cost**.

Machine Learning can become expensive when moving from a small experiment to a large production system.

---

## Where does the cost come from?

Costs can arise from:

```text
Data
+
Training
+
Computing resources
+
Servers
+
Deployment
+
Maintenance
```

For a small college project, you may run a model on your laptop.

But imagine the same model being used by millions of users.

Now the company needs significantly more infrastructure.

---

## Example

Suppose:

```text
100 users
```

use an ML application.

The computational requirement may be manageable.

But suppose:

```text
10 million users
```

use it.

Now the company needs infrastructure capable of handling a huge number of requests.

This increases the cost substantially.

---

## Hidden Costs

The cost of ML is not limited to training the model.

There can also be costs associated with:

* Deploying the model
* Running servers
* Serving predictions
* Updating the model
* Maintaining the infrastructure

The lecture points toward research discussing these **hidden costs of machine learning systems**.

### [IMAGE HERE]

*Suggested image: ML project cost breakdown — data, compute, deployment, maintenance.*

---

# 11. Complete Picture

The challenges discussed in the lecture can be viewed as a progression:

```text
                MACHINE LEARNING PROJECT

                       DATA
                        │
                        ▼
               ┌─────────────────┐
               │ Data Collection │
               └────────┬────────┘
                        ▼
               ┌─────────────────┐
               │ Enough Data?    │
               │                 │
               │ Insufficient    │
               │ Data Problem    │
               └────────┬────────┘
                        ▼
               ┌─────────────────┐
               │ Representative?│
               └────────┬────────┘
                        ▼
               ┌─────────────────┐
               │ Data Quality    │
               └────────┬────────┘
                        ▼
               ┌─────────────────┐
               │ Relevant        │
               │ Features?       │
               └────────┬────────┘
                        ▼
               ┌─────────────────┐
               │ Train Model     │
               └────────┬────────┘
                        ▼
               ┌─────────────────┐
               │ Overfitting /   │
               │ Underfitting    │
               └────────┬────────┘
                        ▼
               ┌─────────────────┐
               │ Software        │
               │ Integration     │
               └────────┬────────┘
                        ▼
               ┌─────────────────┐
               │ Deployment      │
               └────────┬────────┘
                        ▼
               ┌─────────────────┐
               │ Cost            │
               └─────────────────┘
```

---

# 12. Quick Revision

| Challenge                         | What does it mean?                                                            |
| --------------------------------- | ----------------------------------------------------------------------------- |
| **Data Collection**               | Finding and collecting relevant data for the problem                          |
| **Insufficient Data**             | Not having enough training data or labelled data                              |
| **Non-Representative Data**       | Data does not properly represent the real population/problem                  |
| **Poor Quality Data**             | Data contains errors, missing values, noise, etc.                             |
| **Irrelevant Features**           | Features that don't meaningfully help the model                               |
| **Overfitting**                   | Model learns training data too closely and fails to generalize                |
| **Underfitting**                  | Model fails to learn the underlying pattern                                   |
| **Software Integration**          | Difficulty integrating the ML model into the actual software                  |
| **Offline Learning / Deployment** | Difficulty updating, deploying, and maintaining the model as new data arrives |
| **Cost**                          | Data, computation, deployment, and maintenance can become expensive           |

---

# 13. Key Takeaways

### 1. Data is the foundation

Without relevant data, an ML model cannot learn effectively.

### 2. More data is not enough

The data must also be representative and of good quality.

### 3. Features matter

Irrelevant features can negatively affect the learning process.

### 4. The model must generalize

A model should perform well on new data, not just the training data.

### 5. Training is not the end

The model must be integrated into software and deployed for users.

### 6. Production ML has real costs

Running ML systems at scale requires infrastructure and ongoing maintenance.

---

# Final Mental Model

Remember the lecture using this simple chain:

```text
Can I GET the data?
        ↓
Do I have ENOUGH data?
        ↓
Is the data REPRESENTATIVE?
        ↓
Is the data of GOOD QUALITY?
        ↓
Are the FEATURES RELEVANT?
        ↓
Is the model OVERFITTING?
        ↓
Is the model UNDERFITTING?
        ↓
Can I INTEGRATE it into software?
        ↓
Can I DEPLOY and UPDATE it?
        ↓
Can I AFFORD to run it?
```

> **Machine Learning is not just about training an algorithm. The real challenges begin with data and continue all the way to deploying and maintaining the final product.**
