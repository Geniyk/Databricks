%sql
-- Describe Silver table
DESCRIBE TABLE ecommerce.silver.daily_sales;

%sql
-- Describe Bronze table
DESCRIBE TABLE ecommerce.bronze.events;

%sql
-- Describe Gold tables
DESCRIBE TABLE ecommerce.gold.products;

DESCRIBE TABLE ecommerce.gold.top_products;

## Prepare Data (ML features)
from pyspark.sql import functions as F

# Load bronze events table
events = spark.table("ecommerce.bronze.events")

# Feature engineering
ml_data = (
    events
    .filter(F.col("price").isNotNull())
    .withColumn("hour", F.hour("event_time"))
    .withColumn("day_of_week", F.dayofweek("event_time"))
    .select("price", "hour", "day_of_week")
)

display(ml_data)

## Assemble Features
from pyspark.ml.feature import VectorAssembler

assembler = VectorAssembler(
    inputCols=["hour", "day_of_week"],
    outputCol="features"
)

final_data = assembler.transform(ml_data).select("features", "price")

display(final_data)

## Train–Test Split
train_data, test_data = final_data.randomSplit([0.8, 0.2], seed=42)

print("Train count:", train_data.count())
print("Test count:", test_data.count())

## Train Linear Regression Model
from pyspark.ml.regression import LinearRegression

lr = LinearRegression(
    featuresCol="features",
    labelCol="price"
)

model = lr.fit(train_data)

## Log Parameters, Metrics & Model using MLflow

%sql
CREATE VOLUME workspace.ecommerce.mlflow_tmp;

%sql
SHOW VOLUMES IN workspace.ecommerce;


import os
import mlflow
import mlflow.spark

# Required for Unity Catalog 
os.environ["MLFLOW_DFS_TMP"] = "/Volumes/workspace/ecommerce/mlflow_tmp"

# Set experiment
mlflow.set_experiment("/day12-mlflow-linear-regression")

with mlflow.start_run():

    # Log parameters
    mlflow.log_param("model_type", "LinearRegression")
    mlflow.log_param("features", "hour, day_of_week")
    mlflow.log_param("train_ratio", 0.8)

    # Predictions
    predictions = model.transform(test_data)

    from pyspark.ml.evaluation import RegressionEvaluator

    evaluator = RegressionEvaluator(
        labelCol="price",
        predictionCol="prediction",
        metricName="rmse"
    )

    rmse = evaluator.evaluate(predictions)

    # Log metric
    mlflow.log_metric("rmse", rmse)

    # Log model
    mlflow.spark.log_model(
        model,
        artifact_path="linear_regression_model"
    )

    print("RMSE:", rmse)


## Second Run for Comparison
with mlflow.start_run():

    lr2 = LinearRegression(
        featuresCol="features",
        labelCol="price",
        regParam=0.1
    )

    model2 = lr2.fit(train_data)
    predictions2 = model2.transform(test_data)

    rmse2 = evaluator.evaluate(predictions2)

    mlflow.log_param("model_type", "LinearRegression")
    mlflow.log_param("regParam", 0.1)
    mlflow.log_metric("rmse", rmse2)

    mlflow.spark.log_model(model2, "linear_regression_model")

    print("RMSE with regParam:", rmse2)



