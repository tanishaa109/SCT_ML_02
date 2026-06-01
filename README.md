# SCT_ML_02 - Customer Segmentation using K-Means Clustering

## Overview

This project uses the K-Means Clustering algorithm to segment retail store customers based on their Annual Income and Spending Score.

The project also allows users to enter customer details and predict which customer segment (cluster) they belong to.

## Dataset

Mall Customers Dataset

Features Used:

* Annual Income (k$)
* Spending Score (1-100)

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Scikit-Learn

## Project Structure

SCT_ML_02/

├── data/

│   └── Mall_Customers.csv

├── images/

│   ├── elbow_method.png

│   └── customer_segments.png

├── customer_segmentation.py

├── requirements.txt

└── README.md

## Installation

Clone the repository:

```bash
git clone YOUR_REPOSITORY_URL
```

Move into the project directory:

```bash
cd SCT_ML_02
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the Project

```bash
python customer_segmentation.py
```

## User Input Example

When prompted, enter:

```text
Enter Annual Income (k$): 70
Enter Spending Score (1-100): 85
```

Example Output:

```text
Customer belongs to Cluster 1
```

## Workflow

1. Load customer dataset
2. Select Annual Income and Spending Score
3. Apply Elbow Method to find optimal clusters
4. Train K-Means Clustering model
5. Visualize customer segments
6. Accept user input
7. Predict customer cluster

## Generated Outputs

The project automatically generates:

* images/elbow_method.png
* images/customer_segments.png

## Learning Outcomes

* Data Preprocessing
* Unsupervised Machine Learning
* K-Means Clustering
* Elbow Method
* Data Visualization
* Customer Segmentation

## Author

Tanisha Sharma

SkillCraft Technology Internship - SCT_ML_02
