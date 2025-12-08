# AI-Driven Prediction of Movie Ratings
#### Members
Théo Przybylski, Computer Science Department, Hanyang University, matameltheo@gmail.com  
Antoine Maia-Sudre, Computer Science Department, Hanyang University, antoine.maia05@gmail.com  
Joel Suhner, Finance Department, Hanyang University, joel.suhner@gmail.com  
Yannick Matteo Reichle, Information Systems Department, Hanyang University, yannick.reichle@gmail.com  

## Table of Contents
[I. Introduction](#i-introduction) 
- [Motivation](#motivation)  
- [Expected Outcome](#expected-outcome)  

[II. Datasets](#ii-datasets)  
- [Describing your dataset](#describing-your-dataset)  

[III. Methodology](#iii-methodology)  
- [Choice of Algorithms](#choice-of-algorithms)  
- [Features or Code Explanation](#features-or-code-explanation)  

[IV. Evaluation & Analysis](#iv-evaluation--analysis)  
- [Graphs, Tables, Statistics](#graphs-tables-statistics)  

[V. Related Work](#v-related-work)  
- [Tools, Libraries, Documentation Used](#tools-libraries-documentation-used)  

[VI. Conclusion](#vi-conclusion)  
- [Discussion](#discussion)

---

# I. Introduction 

## Motivation
The central motivation for choosing this project was to create a practical example of a complete machine learning workflow using a real-world dataset. Movie data is well-suited for this purpose because it contains a variety of feature types-numerical values, categorical labels, multi-label lists, and free text-allowing us to demonstrate how different preprocessing techniques can be combined in one pipeline.

We selected movie rating prediction specifically because it provides:

- a clear target variable that is easy to interpret,  
- a broad selection of input attributes such as genres, cast, companies, language, and a text overview,  
- a realistic application scenario commonly used in media-related machine learning,  
- and a manageable problem size that makes it feasible to build the full workflow end-to-end.

## Expected Outcome
By the end of the project, the goal is to deliver a complete system that can estimate a movie’s rating based on its metadata. This includes:

- **A cleaned dataset** where incomplete or invalid records are removed and only relevant fields remain.  
- **A preprocessing pipeline** that performs basic text vectorization, encodes categorical and multi-label fields, and normalizes numerical values into a feature matrix.  
- **A trained regression model** that uses this feature matrix to produce stable rating predictions.  
- **General evaluation outputs** that reflect the model’s training behavior and prediction quality.  
- **A simple prediction interface** that allows movie metadata to be entered and evaluated directly.

---

# II. Datasets

The dataset used in this project is a cleaned movie metadata collection obtained from a public source on Hugging Face:  
https://huggingface.co/datasets/wykonos/movies  

It contains information about films such as genres, actors, production companies, language, financial details and audience ratings. The raw dataset was processed to remove incomplete or inconsistent entries, resulting in a structured CSV file suitable for machine learning tasks.

## Dataset Size

- **Rows (instances):** 9,270 movies  
- **Columns (features):** 12 attributes  

This size provides a solid foundation for identifying general relationships between movie characteristics and audience ratings.

## Main Features

Below are the key attributes that contribute to predicting a movie’s rating:

- **title** – Name of the movie.  
- **genres** – One or several genre categories combined into a compact format.  
- **original_language** – Primary language in which the film was released.  
- **production_companies** – List of studios involved in producing the movie.  
- **credits** – Main actors associated with the film.  
- **budget** – Reported production budget.  
- **revenue** – Earnings made by the movie; provides context for performance.  
- **runtime** – Duration of the film in minutes.  
- **popularity** – Score representing public interest.  
- **vote_average** – Target variable: audience rating on a 0–10 scale.  
- **vote_count** – Number of votes; indicates rating reliability.  
- **overview** – Short text summary describing the film.

The dataset brings together textual, numerical and categorical features. This combination allows for a richer predictive model but also requires appropriate preprocessing to ensure compatibility with machine learning methods.

---

# III. Methodology

## Choice of Algorithms
*(Section content goes here)*

## Features or Code Explanation
*(Section content goes here)*

---

# IV. Evaluation & Analysis

## Graphs, Tables, Statistics
*(Section content goes here)*

---

# V. Related Work

## Tools, Libraries, Documentation Used
*(Section content goes here)*

---

# VI. Conclusion

## Discussion
*(Section content goes here)*
