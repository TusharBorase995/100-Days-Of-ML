# Day 3: Types of Machine Learning

## Introduction
Machine Learning algorithms can be categorized based on various criteria. The most prominent and fundamental way to categorize them is **based on the amount of supervision required** during the training phase. 

Based on this categorization, Machine Learning is divided into four main types:
1. **Supervised Learning**
2. **Unsupervised Learning**
3. **Semi-Supervised Learning**
4. **Reinforcement Learning**

---

## 1. Supervised Machine Learning
Supervised Learning is all about learning from labeled data. 
* **Definition:** You provide the machine learning algorithm with a dataset that contains both the **Input (Features/Independent variables)** and the corresponding **Output (Label/Dependent variable)**. 
* **The Goal:** The algorithm's job is to mathematically figure out the relationship or pattern between the inputs and the output so that when given a completely new input in the future, it can accurately predict the output.

**Example: Placement Predictor**
If you have a dataset of 5,000 students, containing:
* *Input 1:* IQ Level
* *Input 2:* CGPA
* *Output:* Placement Status (Yes / No)

The algorithm trains on this data. Later, if you ask it, "Will a student with an IQ of 105 and a CGPA of 8.2 get placed?", it uses the learned patterns to predict the answer.

### Sub-types of Supervised Learning
Supervised Learning is further split into two distinct categories depending purely on the *data type of the output column*.

#### A. Regression
* **Trigger Condition:** Used when your output (Target Column) is **Numerical/Continuous data**.
* **Examples:**
  * Predicting the Salary/Package of a student based on CGPA (Output = 4.5 LPA, 12 LPA, etc.).
  * Predicting the Price of a House based on the number of rooms and location.
  * Predicting a person's weight based on their height.

#### B. Classification
* **Trigger Condition:** Used when your output (Target Column) is **Categorical/Discrete data**.
* **Examples:**
  * Will the student get placed? (Output = Yes / No).
  * Is this email Spam? (Output = Spam / Not Spam).
  * Does the image contain a Dog or a Cat? (Output = Category).

---

## 2. Unsupervised Machine Learning
In Unsupervised Learning, the dataset provided to the algorithm has **Inputs only, with no corresponding Output/Labels.**
* **The Goal:** Since there is no target to predict, the algorithm tries to find hidden structures, patterns, or relationships internally within the raw input data itself.

### The 4 Major Sub-types of Unsupervised Learning:

#### A. Clustering
* **Concept:** Grouping unlabelled data into different categories (clusters) based on their similarities.
* **Example:** In an e-commerce platform, grouping customers based on their purchasing behavior. A clustering algorithm might automatically find groups like "High spenders but rare visits" vs. "Frequent visits but low spenders". You can then use this to send targeted marketing emails.

#### B. Dimensionality Reduction (Feature Extraction)
* **Concept:** Used when you have an overwhelming number of input columns (e.g., 1000+ features in image or text data), which slows down the algorithm and causes overfitting. This technique compresses the data while retaining the maximum variance/information.
* **Example:** If predicting a house price, instead of using "Number of bedrooms" and "Number of washrooms" as two separate columns, a dimensionality reduction technique (like PCA) will merge them into a single new mathematical feature (e.g., "Size factor"), reducing processing load without losing logic. 
* **Visualization use case:** It allows us to visualize 100-dimensional data by squashing it down to 2D or 3D scatter plots.

#### C. Anomaly Detection
* **Concept:** Finding data points that deviate significantly from the normal behavior of the dataset (Outliers).
* **Example:** Credit Card Fraud Detection. The algorithm maps out what normal transactions look like. If suddenly a massive, out-of-character transaction occurs at 3 AM in a different country, the algorithm flags it as an anomaly because it falls far outside the established cluster.

#### D. Association Rule Learning
* **Concept:** Finding hidden relationships or rules between variables in large databases (often used in market basket analysis).
* **Example:** The classic Walmart *Diaper and Beer* case study. By scanning thousands of receipts, the unsupervised algorithm found a strong statistical association that customers buying baby diapers on weekends were also highly likely to buy beer. Consequently, Walmart placed these two items next to each other, significantly boosting sales.

---

## 3. Semi-Supervised Learning
* **The Problem it Solves:** Creating labeled data (Output columns) is very expensive and requires manual human effort. Gathering raw data (Inputs like images) is cheap and easy. 
* **Concept:** A hybrid approach where you have a massive dataset, but only a very tiny fraction of it is labeled. You use Unsupervised Learning (Clustering) to group similar data, and then manually label one data point per cluster. The system then propagates that label to all other items in that cluster.
* **Example:** Google Photos. You take thousands of photos of people. Google groups similar faces together (Unsupervised Clustering). Then, it asks you to manually label just *one* photo ("This is Dad"). Instantly, it applies the label "Dad" to the entire cluster of similar faces.

---

## 4. Reinforcement Learning
* **The Core Difference:** In Supervised, Unsupervised, and Semi-Supervised learning, you need historical data. In Reinforcement Learning (RL), **there is no historical data.** * **Concept:** The algorithm (called an **Agent**) is placed into an environment and learns from scratch by taking actions and observing the results through a system of **Rewards and Punishments**. 
* **The Mechanism:** 1. The Agent takes an action.
  2. If the action is good, the Environment gives it a Reward (+1).
  3. If the action is bad, the Environment gives it a Punishment (-1).
  4. The Agent updates its internal "Policy" to maximize long-term rewards.
* **Examples:**
  * Training a dog (give a treat for sitting, scold for biting).
  * Self-Driving Cars learning to navigate simulations.
  * **AlphaGo by DeepMind:** An RL agent that defeated the world champion in the highly complex board game 'Go'.

---

### Verification Checklist for GitHub Repository
- [x] Categorized ML based on the amount of supervision.
- [x] Defined Supervised Learning and distinguished between Regression and Classification based on data types.
- [x] Detailed the 4 core sub-techniques of Unsupervised Learning (Clustering, PCA, Anomaly, Association).
- [x] Explained the cost-saving benefit of Semi-Supervised Learning using the Google Photos example.
- [x] Documented the Agent-Environment reward loop logic of Reinforcement Learning.
