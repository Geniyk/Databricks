%sql
-- Describe Bronze table
DESCRIBE TABLE ecommerce.bronze.events;

%sql
-- Describe Silver table
DESCRIBE TABLE ecommerce.silver.daily_sales;

%sql
-- Describe Gold tables
DESCRIBE TABLE ecommerce.gold.products;

DESCRIBE TABLE ecommerce.gold.top_products;

## Load data
base_df = spark.table("ecommerce.gold.top_products").dropna()
base_df.printSchema()
display(base_df)

# MAGIC %pip install transformers torch

## Simple NLP Task

from transformers import pipeline

classifier = pipeline("sentiment-analysis")

reviews = [
    "This product is amazing!",
    "Terrible quality, waste of money",
    "Very satisfied with the purchase",
    "Not worth the price"
]

results = classifier(reviews)
results


## Log NLP Model with MLflow

import mlflow

with mlflow.start_run(run_name="sentiment_analysis_nlp"):
    mlflow.log_param("model", "distilbert-base-uncased-finetuned-sst-2-english")
    mlflow.log_param("task", "sentiment-analysis")
    
    # Example metric (for demo)
    mlflow.log_metric("sample_accuracy", 0.95)

