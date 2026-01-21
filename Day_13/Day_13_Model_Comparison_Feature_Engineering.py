## Train 3 Different Models

import os
import mlflow
import mlflow.spark
import pandas as pd

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import (
    LinearRegression,
    DecisionTreeRegressor,
    RandomForestRegressor
)
from pyspark.ml.evaluation import RegressionEvaluator

# Unity Catalog requirement
os.environ["MLFLOW_DFS_TMP"] = "/Volumes/workspace/ecommerce/mlflow_tmp"

# Set MLflow Experiment
mlflow.set_experiment("/day13-mlflow-model-comparison")

# Load data
data = (
    spark.table("ecommerce.silver.daily_sales")
    .select("total_events", "total_revenue")
    .dropna()
)

train_df, test_df = data.randomSplit([0.8, 0.2], seed=42)

# Feature Engineering
assembler = VectorAssembler(
    inputCols=["total_events"],
    outputCol="features"
)

train_vec = assembler.transform(train_df)
test_vec = assembler.transform(test_df)

# Models
models = {
    "LinearRegression": LinearRegression(
        featuresCol="features",
        labelCol="total_revenue"
    ),
    "DecisionTree": DecisionTreeRegressor(
        featuresCol="features",
        labelCol="total_revenue",
        maxDepth=5
    ),
    "RandomForest": RandomForestRegressor(
        featuresCol="features",
        labelCol="total_revenue",
        numTrees=50
    )
}

# Evaluator
evaluator = RegressionEvaluator(
    labelCol="total_revenue",
    predictionCol="prediction",
    metricName="rmse"
)

# Train & log
for name, model in models.items():

    with mlflow.start_run(run_name=name):

        mlflow.log_param("model_type", name)
        mlflow.log_param("features", "total_events")

        fitted_model = model.fit(train_vec)
        predictions = fitted_model.transform(test_vec)

        rmse = evaluator.evaluate(predictions)
        mlflow.log_metric("rmse", rmse)

        #  FIX: Convert DenseVector → list
        sample_rows = (
            train_vec
            .select("features")
            .limit(5)
            .collect()
        )

        input_example = pd.DataFrame({
            "features": [row["features"].toArray().tolist() for row in sample_rows]
        })

        mlflow.spark.log_model(
            spark_model=fitted_model,
            artifact_path="model",
            input_example=input_example
        )

        print(f"{name} | RMSE = {rmse:.2f}")


import mlflow

runs = mlflow.search_runs()

runs[[
    "tags.mlflow.runName",
    "metrics.rmse",
    "params.model_type"
]]

clean_runs = runs[[
    "tags.mlflow.runName",
    "metrics.rmse",
    "params.model_type"
]].rename(columns={
    "tags.mlflow.runName": "run_name",
    "metrics.rmse": "rmse",
    "params.model_type": "model"
})

clean_runs

## Build Spark ML Pipeline

import os
import mlflow
import mlflow.spark

from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator

# Unity Catalog requirement
os.environ["MLFLOW_DFS_TMP"] = "/Volumes/workspace/ecommerce/mlflow_tmp"

# Set MLflow Experiment
mlflow.set_experiment("/day13-mlflow-pipeline")

# Load data from Unity Catalog
data = (
    spark.table("ecommerce.silver.daily_sales")
    .select("total_events", "total_revenue")
    .dropna()
)

# Train-test split
train, test = data.randomSplit([0.8, 0.2], seed=42)

# Feature engineering
assembler = VectorAssembler(
    inputCols=["total_events"],
    outputCol="features"
)

# Model
rf = RandomForestRegressor(
    featuresCol="features",
    labelCol="total_revenue",
    numTrees=50
)

# Pipeline
pipeline = Pipeline(stages=[assembler, rf])

with mlflow.start_run(run_name="RandomForest_Pipeline"):

    pipeline_model = pipeline.fit(train)
    predictions = pipeline_model.transform(test)

    evaluator = RegressionEvaluator(
        labelCol="total_revenue",
        predictionCol="prediction",
        metricName="rmse"
    )

    rmse = evaluator.evaluate(predictions)

    # Log metadata
    mlflow.log_param("model_type", "RandomForestPipeline")
    mlflow.log_param("features", "total_events")
    mlflow.log_metric("rmse", rmse)

    # Log pipeline model (UC-safe)
    mlflow.spark.log_model(
        spark_model=pipeline_model,
        artifact_path="pipeline_model"
    )

    print("Pipeline RMSE:", rmse)

## Select Best Model
best_run = clean_runs.sort_values("rmse").iloc[0]

print("Best Model:", best_run["model"])
print("Best RMSE:", best_run["rmse"])
