# Principal Component Analysis (PCA) — A Detailed Guide

Principal Component Analysis (PCA) is one of the most important techniques in statistics, machine learning, data analysis, and scientific research.

PCA is a method for reducing the number of variables in a dataset while preserving as much of the important information (variation) as possible.

It helps answer questions such as:

* Can I represent 100 features using just 2 or 3 features?

* How can I visualize high-dimensional data?

* Which patterns explain the most variation in my data?

* Are many of my variables redundant or correlated?

* Can I simplify a dataset before applying a machine-learning algorithm?

I'll teach it progressively—from intuition to mathematics, calculations, implementation, interpretation, and practical pitfalls.

### 1\. The intuition behind PCA

Imagine you have a dataset of students containing two features:

* Height

* Weight

These two variables are often correlated: taller students tend to weigh more.

![PCA Unlocked: The Math, Intuition, and Applications | by Abdelrahman Rabah | Medium](https://images.openai.com/static-rsc-4/sfl1P5boYyIxsYBcNPG3QTdm4L4jFcLPm7kOkvx6aWu1nHXIgamaM-MUKCM_-nBokkBIlgOFJ-RniWthSrx9bRAghw6q9yCLoADYEAySJSfj-beKp4B_AKHZqLBdPDhbwl0_7wf52EeI1hhBUDEP4h2CMf0LAAjeD1G0x7VDdC3oiPPSwLXHryUVfc70uU7I?purpose=fullsize)

![線形代数と機械学習 | 東京科学大学デジタル創作同好会traP](https://images.openai.com/static-rsc-4/0Hcdu2RpG_Ldk0T95HtxqKauoISe-crWpbyu3HDqTezJ82rcUsVpvdWEn9PGdklnJTVP1Lcfdtu2LqYtwlfSYAysAWAE5fpaDld5OodcLaHjtKUwV3XdxaGW3Y5Dj2LeOIdAzz_m5LPjEqf30Q_9_j5hL53Fpuu0KS91LkyZA_Vw3FQP5I5A1wvkHvaaAu04?purpose=fullsize)

![Google Colab](https://images.openai.com/static-rsc-4/cM7OD2Nwu1ru71hODWAKvUQNpHZhRHyTyu8-0THeAIiJZprx0l5BJuci5sbVY7WvWdIDFkXwQP3YWyBD-jSH3W20WSoQzTezQnAV-aNSrvbHawasO1Z0lER_aSjIYxT4fHUb6PS1FmCUgngSzz5wSBrvZFtgBnXcXCAK4GDLUBQ30njADtW0ZZZK8ooUSC3G?purpose=fullsize)

Suppose you plot every student on a graph:

* X-axis: height

* Y-axis: weight

The data points may form an elongated cloud diagonally.

The important observation is that the data varies mostly along one diagonal direction. Although you have two variables, most of the variation may be captured by a single direction.

PCA discovers this direction.

Instead of describing each student by:

* Height

* Weight

PCA creates a new variable called Principal Component 1 (PC1), which is a combination of height and weight.

For example:

PC1=0.7(\\text{Height})+0.7(\\text{Weight})

PC1 captures the direction of greatest variance in the dataset.

PCA may also discover a second direction, PC2, perpendicular to PC1, capturing the remaining variation.

If PC1 captures 95% of the variation, perhaps we can keep only PC1 and discard PC2, thereby reducing the data from two dimensions to one with relatively little information loss.

That's the core idea of PCA.

### 2\. What exactly is dimensionality reduction?

Suppose your dataset contains 10 variables:

X=
\\begin{bmatrix}
x_1 & x_2 & x_3 & \\cdots & x_{10}
\\end{bmatrix}

Each observation has 10 features.

Your dataset might contain 1,000 observations and 10 features:

X\\in\\mathbb{R}^{1000\\times10}

PCA transforms those 10 features into a smaller number of new features, perhaps 2.

X\\in\\mathbb{R}^{1000\\times10}
\\quad\\longrightarrow\\quad
Z\\in\\mathbb{R}^{1000\\times2}

Before PCA

10 original features per observation

Feature 1 · Feature 2 · Feature 3 · ... · Feature 10

After PCA

2 new features per observation

PC1 · PC2

Important: PCA doesn't simply select two of the original variables. It creates new variables from combinations of the original variables.

These new variables are called principal components.

### 3\. What is variance, and why does PCA care about it?

PCA is fundamentally a variance-based technique.

### 3.1 Understanding variance

Variance measures how spread out values are around their mean.

For a variable , the sample variance is:

s^2=\\frac{1}{n-1}\\sum_{i=1}^{n}(x_i-\\bar{x})^2

Where:

* = number of observations

* = individual observation

* = mean

If a variable changes substantially across observations, it has high variance.

If it barely changes, it has low variance.

### 3.2 Why high variance matters

Consider two features:

| Feature | Variance |
| ------- | -------- |
| Height  | 100      |
| Weight  | 400      |

A direction along which the data varies more contains more of the dataset's statistical spread.

PCA seeks directions with maximum variance.

However, high variance is not automatically the same as high predictive value, importance, or usefulness.

PCA is unsupervised: it does not know which features are important for predicting your target.

### 4\. The mathematical foundation of PCA

Let's build the algorithm step by step.

Suppose we have a dataset:

X\\in\\mathbb{R}^{n\\times p}

Where:

* = number of samples

* = number of features

We want to transform these features into components, where:

k<p

There are several mathematical steps.

### Step 1: Center the data

For each feature, subtract its mean.

For feature :

\\bar{x}_j=\\frac{1}{n}\\sum_{i=1}^{n}x_{ij}

Then:

x_{ij}^{centered}=x_{ij}-\\bar{x}_j

This ensures that every feature has mean zero.

Why centering is necessary

PCA searches for directions of variation around the mean. If you don't center the data, the location of the dataset relative to the origin can distort the analysis.

The centered data matrix is:

X_c=X-\\mathbf{1}\\mu^T

Where is the vector of feature means.

### Step 2: Standardize the data when appropriate

Centering and scaling are different operations.

Centering:

x_{ij}'=x_{ij}-\\bar{x}_j

Standardization:

z_{ij}=\\frac{x_{ij}-\\bar{x}_j}{s_j}

Where is the standard deviation of feature .

When should you standardize?

Usually standardize

Variables have different units or scales.

Examples: income, age, temperature, height.

Sometimes don't scale

All features share meaningful comparable units and relative variance is intentionally important.

Example: several measurements in the same units from a specific physical system.

### Example

Suppose you have:

| Feature            | Typical scale  |
| ------------------ | -------------- |
| Age                | 20–80          |
| Income             | 20,000–200,000 |
| Satisfaction score | 1–5            |

Without standardization, income could dominate PCA simply because its numerical values are much larger.

Standardization helps give each feature a comparable starting scale.

Scaling is not universally correct. Whether to standardize depends on the scientific meaning of your data.

### Step 3: Calculate the covariance matrix

The covariance matrix describes how variables vary together.

For a centered dataset, the sample covariance matrix is:

C=\\frac{1}{n-1}X_c^TX_c

If there are 3 features, the covariance matrix is:

C=
\\begin{bmatrix}
\\operatorname{Var}(X_1)&\\operatorname{Cov}(X_1,X_2)&\\operatorname{Cov}(X_1,X_3)\\\\
\\operatorname{Cov}(X_2,X_1)&\\operatorname{Var}(X_2)&\\operatorname{Cov}(X_2,X_3)\\\\
\\operatorname{Cov}(X_3,X_1)&\\operatorname{Cov}(X_3,X_2)&\\operatorname{Var}(X_3)
\\end{bmatrix}

What does covariance tell us?

\\operatorname{Cov}(X,Y)
=
\\frac{1}{n-1}
\\sum_{i=1}^{n}(x_i-\\bar{x})(y_i-\\bar{y})

* Positive covariance: variables tend to increase together.

* Negative covariance: one tends to increase while the other decreases.

* Near-zero covariance: no strong linear co-variation is indicated.

### Example covariance matrix

C=
\\begin{bmatrix}
4&3\\\\
3&4
\\end{bmatrix}

This suggests that the two variables have:

* Variance of 4 each

* Positive covariance of 3

The two features are strongly related.

### Step 4: Find eigenvalues and eigenvectors

This is the heart of PCA.

PCA computes the eigenvalues and eigenvectors of the covariance matrix.

Mathematically:

Cv=\\lambda v

Where:

* = covariance matrix

* = eigenvector

* = eigenvalue

An eigenvector represents a direction.

An eigenvalue represents the variance captured along that direction.

### The intuition

Imagine rotating your coordinate axes.

Your original X-axis and Y-axis might not align with the directions in which the data varies most.

PCA finds new axes:

* PC1: direction with maximum variance.

* PC2: direction with maximum remaining variance, orthogonal to PC1.

* PC3: direction with maximum remaining variance, orthogonal to PC1 and PC2.

* And so on.

![Understanding the Covariance Matrix — Data Mining](https://images.openai.com/static-rsc-4/t-xKeWEn0g7PNq061cpMRvOsO3mZzTyuO4rAqWDP9Ga2gjyZDKURh_wHURzC8TLyrZVoQc8huRBOYlr7j9EZrKfOkKrN1xA3Y_pJFCDaLHhJ_M6vNujlTAWcLqce1FyQhL2Fq1SnO6XTOandfo-YWCBfhUBVpVrzTxyb5lb_YKHAR3R3_wPd7I7f4T7FNZ2R?purpose=fullsize)

![Week 6: Corner Detection](https://images.openai.com/static-rsc-4/94cQJHbLvTcyVEN-6B0MaFsbH_nBtpBgKIto1abswNng7pKhvlh4cKTo0a8CJdTqHHn9DD1bPmSDBJsfnY_sNOLOFATcerMhW6DCLJDhNJFZqrbpeizaVAPqsYkYgdx5K3isulp-VBF4AlHOvsU1bWCMj_AtRGZiG52Kk1jNrGo76tklvbA_sNmDUiQzqO8X?purpose=fullsize)

![Evaluation Report — NVIDIA NeMo Microservices](https://images.openai.com/static-rsc-4/eGvM_dqcP4GjMj8vwzzuSDvrrzZuih9FwBe1qmIixeikj4FkdV4JmBCwP2-zPk9pUYDMrbF62oEY8VGdVWx8QgT1UahXLpXvitTpOX_0sh8hFzv-Lh9DcSAhQI-jhKMeR7qwwmdmefnGUnEIQJHkmAHHug_HOUJdb7N1FQj3FHNrKg_GLhKHPDrLfTyakZOM?purpose=fullsize)

### Eigenvector

An eigenvector describes the direction of a principal component.

For example:

v_1=
\\begin{bmatrix}
0.707\\\\
0.707
\\end{bmatrix}

This direction combines the two variables positively and equally.

### Eigenvalue

If:

\\lambda_1=7

Then the first principal component captures a variance of 7 in the corresponding covariance-based PCA space.

If:

\\lambda_2=1

The second component captures variance of 1.

Since:

\\lambda_1>\\lambda_2

PC1 captures more variance than PC2.

### Step 5: Sort the eigenvalues

The eigenvalues are sorted from largest to smallest:

\\lambda_1\\geq\\lambda_2\\geq\\lambda_3\\geq\\cdots\\geq\\lambda_p

The associated eigenvectors are sorted in the same order.

This determines the ranking of principal components.

| Component | Eigenvalue     | Interpretation                  |
| --------- | -------------- | ------------------------------- |
| PC1       | Largest        | Captures the most variance      |
| PC2       | Second largest | Captures the next most variance |
| PC3       | Third largest  | Captures the next most variance |
| ...       | ...            | ...                             |

### Step 6: Calculate explained variance

Each principal component captures a portion of the total variance.

The explained variance ratio of component is:

EVR_j=\\frac{\\lambda_j}{\\sum_{i=1}^{p}\\lambda_i}

The total variance is:

\\sum_{i=1}^{p}\\lambda_i

### Example

Suppose the eigenvalues are:

\\lambda_1=7\\lambda_2=1

The total variance is:

7+1=8

Therefore:

EVR_1=\\frac{7}{8}=0.875EVR_2=\\frac{1}{8}=0.125

PC1

### 87.5%

Variance explained

PC2

### 12.5%

Variance explained

In this example, keeping only PC1 would preserve 87.5% of the total variance.

### Step 7: Project the data onto the principal components

After identifying the principal component directions, we transform the original data.

Let:

V_k=
\\begin{bmatrix}
|&|&|\\\\
v_1&v_2&\\cdots&v_k\\\\
|&|&|
\\end{bmatrix}

This matrix contains the first eigenvectors.

The transformed dataset is:

Z=X_cV_k

If the data has been standardized, substitute the standardized matrix for .

The PCA transformation

Z=X_cV_k

Original centered data × selected principal component directions = transformed data.

The transformed features are called:

* Principal component scores

* PCA scores

* Transformed coordinates

Each row of represents one observation in the new PCA coordinate system.

### 5\. A complete numerical example

Let's perform PCA on a simple two-variable dataset.

Suppose we have five observations:

| Observation | X   | Y   |
| ----------- | --- | --- |
| A           | 2   | 1   |
| B           | 3   | 2   |
| C           | 4   | 3   |
| D           | 5   | 4   |
| E           | 6   | 5   |

These variables are highly correlated.

### Step 1: Calculate the means

\\bar{X}=\\frac{2+3+4+5+6}{5}=4\\bar{Y}=\\frac{1+2+3+4+5}{5}=3

### Step 2: Center the dataset

Subtract the mean of each variable.

| Observation | X centered | Y centered |
| ----------- | ---------- | ---------- |
| A           | -2         | -2         |
| B           | -1         | -1         |
| C           | 0          | 0          |
| D           | 1          | 1          |
| E           | 2          | 2          |

The centered matrix is:

X_c=
\\begin{bmatrix}
-2&-2\\\\
-1&-1\\\\
0&0\\\\
1&1\\\\
2&2
\\end{bmatrix}

### Step 3: Calculate the covariance matrix

The variance of X is:

\\operatorname{Var}(X)=
\\frac{4+1+0+1+4}{4}
=\\frac{10}{4}=2.5

The variance of Y is also 2.5.

The covariance is:

\\operatorname{Cov}(X,Y)=
\\frac{4+1+0+1+4}{4}
=2.5

Therefore:

C=
\\begin{bmatrix}
2.5&2.5\\\\
2.5&2.5
\\end{bmatrix}

### Step 4: Find the eigenvalues

Solve:

\\det(C-\\lambda I)=0\\det
\\begin{bmatrix}
2.5-\\lambda&2.5\\\\
2.5&2.5-\\lambda
\\end{bmatrix}
=0(2.5-\\lambda)^2-6.25=0

This gives:

\\lambda_1=5\\lambda_2=0

The dataset has no variance along the second principal component.

This is because X and Y are perfectly linearly related in this dataset.

### Step 5: Find the eigenvectors

The first eigenvector is:

v_1=
\\begin{bmatrix}
1/\\sqrt{2}\\\\
1/\\sqrt{2}
\\end{bmatrix}

The second eigenvector is:

v_2=
\\begin{bmatrix}
1/\\sqrt{2}\\\\
-1/\\sqrt{2}
\\end{bmatrix}

The first principal component is therefore:

PC1=\\frac{X_c+Y_c}{\\sqrt{2}}

The second principal component is:

PC2=\\frac{X_c-Y_c}{\\sqrt{2}}

### Step 6: Calculate explained variance

EVR_1=\\frac{5}{5+0}=1EVR_2=\\frac{0}{5+0}=0

| Principal component | Eigenvalue | Explained variance |
| ------------------- | ---------- | ------------------ |
| PC1                 | 5          | 100%               |
| PC2                 | 0          | 0%                 |

We can reduce the two-dimensional dataset to one dimension without losing any variance in this example.

### Step 7: Transform the data

Using:

PC1=\\frac{X_c+Y_c}{\\sqrt{2}}

For observation A:

PC1_A=\\frac{-2+(-2)}{\\sqrt{2}}PC1_A=-2\\sqrt{2}\\approx-2.828

For observation B:

PC1_B=\\frac{-1+(-1)}{\\sqrt{2}}PC1_B=-\\sqrt{2}\\approx-1.414

For observation C:

PC1_C=0

For observation D:

PC1_D=\\sqrt{2}\\approx1.414

For observation E:

PC1_E=2\\sqrt{2}\\approx2.828

Original data

Each observation has two coordinates (X, Y).

Reduced data

Each observation now has only one coordinate (PC1).

| Observation | PC1    |
| ----------- | ------ |
| A           | -2.828 |
| B           | -1.414 |
| C           | 0      |
| D           | 1.414  |
| E           | 2.828  |

We have successfully reduced two variables to one.

### 6\. What are loadings?

Loadings describe how the original variables contribute to the principal components.

Consider:

PC1=0.707X+0.707Y

The coefficients:

* 0.707 for X

* 0.707 for Y

are the weights in the component direction, often referred to as eigenvector coefficients or PCA weights.

They indicate that both X and Y contribute equally to this component.

Now suppose another component is:

PC1=0.9X+0.1Y

This component is more strongly oriented toward X.

### Interpreting loadings

Suppose PCA produces:

| Variable   | PC1  | PC2  |
| ---------- | ---- | ---- |
| Income     | 0.80 | 0.10 |
| Education  | 0.75 | 0.20 |
| Age        | 0.05 | 0.95 |
| Experience | 0.30 | 0.15 |

We might interpret:

* PC1 is primarily related to income and education.

* PC2 is primarily related to age.

These are tentative interpretations, not definitive causal explanations.

Be careful: The term loading is used inconsistently across PCA literature and software. Eigenvector coefficients, correlations between variables and components, and scaled loadings are not always the same quantity. Always check the definition being used.

### 7\. How do you choose the number of principal components?

One of the most practical questions in PCA is:

> How many components should I keep?

There are several approaches.

### Method 1: Cumulative explained variance

Suppose PCA produces:

| Component | Explained variance |
| --------- | ------------------ |
| PC1       | 40%                |
| PC2       | 25%                |
| PC3       | 15%                |
| PC4       | 10%                |
| PC5       | 5%                 |
| PC6       | 3%                 |
| PC7       | 2%                 |

Cumulative variance:

| Components | Cumulative variance |
| ---------- | ------------------- |
| PC1        | 40%                 |
| PC1–PC2    | 65%                 |
| PC1–PC3    | 80%                 |
| PC1–PC4    | 90%                 |
| PC1–PC5    | 95%                 |
| PC1–PC6    | 98%                 |
| PC1–PC7    | 100%                |

If your threshold is 90%, you would retain four components.

40+25+15+10=90\\%

Typical thresholds may be 80%, 90%, or 95%, depending on the context.

These thresholds are heuristics, not universal scientific rules.

### Method 2: Scree plot

A scree plot displays eigenvalues or explained variance against component number.

![PCA: Master the Algorithm That Makes High-Dimensional Data Manageable - perfecXion.ai](https://images.openai.com/static-rsc-4/Ce23BcOVzKgkMZ6XgNiUFWU8sQuvTzonfpEbiBT9zwHnB8TvPoK6KheIopCvAKIX_QDiZSNLjJesPp9KDrwGUHhWK7Q2Utcy2MgKet41b55fJL8K1vkWL7E6lKeUKH0-sa7lcMC0_ylGwgbZFnTUS9TDQ50_uQm1gWBX2AfBiCNY8twMCqOF2opbqBkeOFHw?purpose=fullsize)

![Comprehensive Statistical Analysis for Characterizing Water Quality Assessment in the Mekong Delta: Trends, Variability, and Key Influencing Factors](https://images.openai.com/static-rsc-4/REiPFq4TCv5vTf_TIM5nO-CGANjV6GSAuOtFV0cyISCXF3ybW6DyLvRnflsZ2fD0BvsJHxT1Pt21mPngUcWUZxTFtet7n9IPcKVUDpeCWSja-0TBYrwKIKZgDvpXr7uvMDWH1QhkVHKKo-ElSldHNJQDRF2Tys1lXFxUtRkJnXpffS9kxLtngFnof85MVOoQ?purpose=fullsize)

![Section 8 Principal Component Analysis (PCA) | ASTERICS: User documentation](https://images.openai.com/static-rsc-4/wK9M0DJR4ayc4YK8YBqAhZG24lOo68cc5xWX9MLLmfjiHM_GZAqBKZ-u-wK5YPjeQErTdw1dO8gZkTVoCI8JIuIJBzJkhVblIy3FCQI88r2nYyp6xUY27LzPv4A_ST6sfjTzTdektiKIL9vT-y0HRSlNU5PKdqeA-saxYZZjhv4iv1QcI_kWjSBShkvxw9yL?purpose=fullsize)

![How does PCA work? Maths behind PCA | Analytics Vidhya](https://images.openai.com/static-rsc-4/sjAzUOZB_kt9Anglqw6hJ4qoi0ZZYswuXQuSZqaorzbbrvcxiaKsdu89UO3oRHIw1pKhpHsCaRJBvsUW1o9tyT0fk1VMrxqths2mPckBooryBx_iKxM69tQu1GCDu6-l0hSnBPiQw4eZoZGpuGh8P3GZJw7aNEFGaAnWOo1Q0YwKcZn_gNSqlw-96-ZAOqfS?purpose=fullsize)

![Friendly k-means and PCA python tutorial | Step By Step Data Science](https://images.openai.com/static-rsc-4/kuIcV2x46BaykZS5i5FSXgz1Kn0SF8FgSV3DlgTx9sB5ppJq3kpDcv_YiFlEXL3K8gUcUaWkCGBivaRMvcUvXcXRUu-IshPK0iELqNu0EwkwkhewXbRhQCI4hpS7xpQdx69voCRaXWemXuuO2KxjTSSNRLeJ1CWDY9pF5lw_XUjnqQc1vTQB9elq_uiTCtt9?purpose=fullsize)

![What Kind of Car Do You Like? Principal Component Analysis (PCA) step by step | Medium](https://images.openai.com/static-rsc-4/GDSs_e1DKgkwis4M6qSFDeKXOI2y6am_npQRbvOLjqxGTRlC51o-9onCFdWYAKm3op0_T1ZCy_Ha1gJhHRYW-S-QqtxBLtt13GC9cTfD__sRej7jv9MtHHmzvTP6yxYc-RQK3vNSz47JVvGGx__u4FKi2WkAwfD-dJPrHnAc4kV6v-pKOzCIjF_ffv--0I6P?purpose=fullsize)

![ML From Scratch VI: Principal Component Analysis](https://images.openai.com/static-rsc-4/OdeFL8qlsKMuS_YhBcRY6L_X4HK3K7oJjIXFpcDA-2GMuyRO4DiCqSBTTtaVvJgjrZEkd5rqIXnPvhg1rc_x7ZdhSsT1FDqsbSfHfqqv6f0EEknqzR3K_-LtHBJ0MiBov1FTmkaS73YNBdnkmd3tSyz8gOo2LJEKWSFO3BJlvBOR27xmsd4wehZQDK4pOj_K?purpose=fullsize)

![PCA Explained with an Example. PCA is a dimensionality reduction… | by Nazrul Miya | Medium](https://images.openai.com/static-rsc-4/c6Ym-2pBUVwB2H3vZrPMqV5bJpSa4V8spYkV2zvklV13HBltawmrIQKT0tyX7zBDcoH58pIBYt34zZBZlSagxcxKiAWJhUoN5fVGu5OITup-Q_7Msld4vvoV9Ii6hNDGuPVo04IoARwf7NL9bRYwxceCtmQT_Zz5ARyz_Ucke5Cvf1PRoOOYwqcHHaOoFtDU?purpose=fullsize)

You look for the point where the curve begins to flatten.

This is called the elbow.

The idea:

* Before the elbow: components explain relatively substantial amounts of variation.

* After the elbow: additional components provide comparatively smaller gains.

The elbow isn't always obvious, and subjective judgment may be involved.

### Method 3: Kaiser criterion

When performing PCA on a correlation matrix, a common heuristic is:

\\lambda>1

Keep components whose eigenvalues exceed 1.

Why?

When standardized variables are used, each original variable has variance of approximately 1.

Therefore, a component with eigenvalue greater than 1 captures more variance than a single standardized original variable.

The Kaiser criterion is a heuristic and should not be used blindly.

### Method 4: Evaluate downstream performance

If PCA is used before machine learning, you can choose the number of components based on performance in cross-validation.

For example:

| Number of PCs | Cross-validation accuracy |
| ------------- | ------------------------- |
| 2             | 78%                       |
| 5             | 84%                       |
| 10            | 88%                       |
| 20            | 89%                       |
| 30            | 87%                       |

Perhaps 20 components are best.

This approach is often more useful than choosing solely based on variance explained when PCA is used for prediction.

The number of components must be selected without leaking information from the test set. PCA should be fitted within each training fold of a cross-validation pipeline.

### 8\. PCA in Python: a practical implementation

We'll use Python, NumPy, Pandas and Scikit-learn.

### Step 1: Import libraries

```python
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
```

### Step 2: Create a dataset

```python
data = pd.DataFrame({
    "height": [160, 165, 170, 175, 180],
    "weight": [55, 60, 65, 70, 75],
    "age": [20, 25, 30, 35, 40]
})

print(data)
```

### Step 3: Standardize the data

```python
scaler = StandardScaler()

X_scaled = scaler.fit_transform(data)
```

We standardize because the variables may have different scales and distributions.

### Step 4: Fit PCA

```python
pca = PCA()

X_pca = pca.fit_transform(X_scaled)
```

Here, PCA initially calculates all available components.

### Step 5: Examine explained variance

```python
print(pca.explained_variance_ratio_)
```

Example output:

```text
[0.95 0.04 0.01]
```

This means PC1 captures 95% of the variance, PC2 captures 4%, and PC3 captures 1%.

The exact results depend on the dataset.

### Step 6: Examine cumulative variance

```python
cumulative_variance = np.cumsum(
    pca.explained_variance_ratio_
)

print(cumulative_variance)
```

Example:

```text
[0.95 0.99 1.00]
```

### Step 7: Select the number of components

Suppose you want to retain 95% of the variance:

```python
pca = PCA(n_components=0.95)

X_reduced = pca.fit_transform(X_scaled)

print(X_reduced.shape)
```

Scikit-learn allows you to specify a fraction such as `0.95` for variance-based component selection with its PCA implementation.

### Step 8: Examine the component weights

```python
print(pca.components_)
```

The rows of `pca.components_` contain the component direction coefficients.

For example, if the output is:

```text
[[ 0.60  0.58  0.55]
 [ 0.70 -0.70  0.10]]
```

The first component combines all three features, while the second emphasizes a contrast between the first two features.

The interpretation depends on the actual coefficients and the variables involved.

### 9\. PCA visualization

PCA is commonly used to visualize datasets with many dimensions.

Suppose a dataset has 20 features. We can reduce it to 2 components:

```python
pca = PCA(n_components=2)

X_2d = pca.fit_transform(X_scaled)

plot_data = pd.DataFrame(
    X_2d,
    columns=["PC1", "PC2"]
)

print(plot_data.head())
```

Then visualize it:

```python
import matplotlib.pyplot as plt

plt.scatter(
    plot_data["PC1"],
    plot_data["PC2"]
)

plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA Projection")
plt.show()
```

![Example for Principal Component Analysis (PCA): Iris data](https://images.openai.com/static-rsc-4/00zPGHznBA8dttv_VyDVnewCCsxqs_4efKKk3rZI74qEcYyEikj2Rf13QKWYEz_Hklly-LoagT5GAv0FQwza17KfwoIw-J1w6gjg1NBYgzjtuco8Gk9MlvQlOGlEV7tRQ51anEHoYtI0MiC4_oR9f4gnk6DgenhCV9FmJ-KjJYJs4CUQnDSidUNux4Hhgo_V?purpose=fullsize)

Each point represents an observation projected into the new two-dimensional coordinate system.

This visualization may help reveal:

* Clusters

* Outliers

* Patterns

* Group separation

* Trends

* Overlapping populations

PCA does not guarantee that clusters or class separation will be visible.

It preserves variance, not necessarily the information that best separates classes.

### 10\. PCA and Singular Value Decomposition (SVD)

There is another way to compute PCA: Singular Value Decomposition (SVD).

Instead of explicitly calculating the covariance matrix and then finding its eigenvectors, we decompose the centered data matrix:

X_c=U\\Sigma V^T

Where:

* = left singular vectors

* = diagonal matrix of singular values

* = right singular vectors

The columns of correspond to principal component directions.

The relationship between singular values and covariance eigenvalues is:

\\lambda_i=\\frac{\\sigma_i^2}{n-1}

Where:

* = singular value

* = covariance matrix eigenvalue

SVD is often numerically more stable and is a common implementation approach for PCA.

### 11\. PCA versus other dimensionality-reduction techniques

| Technique         | Main purpose                     | Linear?           | Supervised?                       |
| ----------------- | -------------------------------- | ----------------- | --------------------------------- |
| PCA               | Preserve variance                | Yes               | No                                |
| LDA               | Separate classes                 | Yes               | Yes                               |
| t-SNE             | Visualize local structure        | No                | Usually no                        |
| UMAP              | Visualize/manifold structure     | No                | Usually no                        |
| Autoencoder       | Learn compressed representations | Usually nonlinear | Usually no, but can be supervised |
| Feature selection | Select original variables        | Not necessarily   | Depends                           |

### PCA vs. LDA

PCA maximizes variance.

LDA attempts to maximize separation between classes relative to within-class variation.

If your goal is classification, LDA or supervised methods may sometimes be more appropriate.

### PCA vs. t-SNE

PCA is a linear transformation and preserves large-scale variance patterns.

t-SNE is primarily a nonlinear visualization technique that emphasizes local neighborhood structure.

### PCA vs. UMAP

UMAP can capture nonlinear manifold structure and is often used to visualize high-dimensional datasets.

However, its axes generally aren't interpretable in the same way as PCA axes.

### 12\. Applications of PCA

### A. Machine learning preprocessing

PCA can reduce the number of features before training:

* Logistic regression

* Support vector machines

* K-nearest neighbors

* Clustering algorithms

* Neural networks

Potential benefits include reduced computational cost and less redundancy.

### B. Image compression

Images can be represented as matrices or collections of features.

PCA can compress image representations by retaining a subset of components.

This may reduce storage requirements, although PCA isn't necessarily the best compression method for every image.

### C. Financial analysis

PCA is used to study correlated financial variables.

For example, in fixed-income markets, PCA may be applied to interest-rate curves.

The first few components often correspond to broad patterns such as:

* Level

* Slope

* Curvature

These are common interpretations in yield-curve PCA, though the precise meaning depends on the data and model.

### D. Bioinformatics

PCA is frequently used in:

* Gene-expression analysis

* Genomics

* Population genetics

* Proteomics

It can help identify major patterns and population structure.

### E. Exploratory data analysis

PCA can help analysts:

* Detect unusual observations

* Discover structure

* Investigate correlations

* Understand data distributions

* Explore high-dimensional data

### F. Multicollinearity reduction

When many predictors are correlated, PCA can transform them into orthogonal components.

The components are uncorrelated within the PCA representation, making PCA-based regression useful in certain settings.

This approach is often referred to as principal component regression (PCR).

### 13\. PCA limitations and common mistakes

These are particularly important in practical applications.

### 1\. PCA is sensitive to scale

If features have different scales, large-scale variables may dominate.

Solution: Standardize when appropriate.

### 2\. PCA is sensitive to outliers

Extreme observations can significantly influence the covariance matrix and principal components.

Solution: Investigate outliers and consider robust PCA methods if appropriate.

### 3\. PCA only captures linear structure

Standard PCA finds linear combinations of variables.

It may not capture nonlinear relationships effectively.

### 4\. PCA may reduce interpretability

The original features may be easy to interpret.

Principal components may be combinations of dozens or hundreds of features.

### 5\. High variance doesn't necessarily mean predictive importance

A feature can have low variance but be highly predictive of a target.

PCA doesn't know about your target variable.

### 6\. PCA can discard useful information

A component with low variance might contain useful information for classification, regression or another downstream task.

### 7\. PCA can cause data leakage

Never fit PCA on your entire dataset before splitting into training and test sets in a predictive modeling workflow.

Instead:

* Split the data.

* Fit the scaler on the training set.

* Fit PCA on the training set.

* Transform training data.

* Transform test data using the fitted scaler and PCA.

* Evaluate your model.

A Scikit-learn pipeline is generally the safest approach.

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("pca", PCA(n_components=0.95)),
    ("model", LogisticRegression())
])
```

You can then use this pipeline within cross-validation.

### 14\. PCA cheat sheet

| Concept                  | Meaning                                                                          |
| ------------------------ | -------------------------------------------------------------------------------- |
| Variance                 | How much a variable changes                                                      |
| Covariance               | How two variables vary together                                                  |
| Eigenvector              | Direction of a principal component                                               |
| Eigenvalue               | Variance captured along that direction                                           |
| PC1                      | Direction of maximum variance                                                    |
| PC2                      | Next maximum variance, orthogonal to PC1                                         |
| Loading/weight           | Coefficient or contribution associated with a component, depending on definition |
| Score                    | Coordinates of observations in PCA space                                         |
| Explained variance ratio | Fraction of total variance captured by a component                               |
| Scree plot               | Plot used to inspect eigenvalues or variance                                     |
| SVD                      | A numerical method commonly used to compute PCA                                  |

### 15\. The PCA workflow to memorize

The complete PCA pipeline

Original dataset

Center the data

Scale if appropriate

Calculate covariance matrix / use SVD

Find eigenvectors and eigenvalues

Sort components by explained variance

Choose the number of components

Project the dataset onto selected components

### Remember these three equations

1\. Covariance matrix

C=\\frac{1}{n-1}X_c^TX_c

2\. Eigenvalue equation

Cv=\\lambda v

3\. PCA transformation

Z=X_cV_k

The big idea: PCA finds directions along which your data varies most, then expresses your observations along those new directions so you can represent complex datasets with fewer dimensions.

If you want to master PCA rather than just understand the theory, I can next teach you PCA from scratch with a hand-calculated example, visual intuition, Python code, and exercises, as a step-by-step mini-course.
