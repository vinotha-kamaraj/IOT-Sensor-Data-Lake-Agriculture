# Databricks notebook source
# ============================================================
# IMPORT LIBRARIES
# ============================================================

from pyspark.sql import functions as F

from pyspark.ml.feature import VectorAssembler

from pyspark.ml.classification import RandomForestClassifier

from pyspark.ml.evaluation import (
    MulticlassClassificationEvaluator
)


# ============================================================
# FEATURE ENGINEERING SOURCE TABLE
# ============================================================

FEATURE_ENGINEERING_TABLE = (
    "agriculture_iot.feature_engineering.irrigation_features"
)


# ============================================================
# MODEL NAME
# ============================================================

MODEL_NAME = (
    "agriculture_iot_irrigation_random_forest"
)

# COMMAND ----------

# ============================================================
# READ FEATURE ENGINEERING TABLE
# ============================================================

ml_df = (
    spark.table(
        FEATURE_ENGINEERING_TABLE
    )
)


# ============================================================
# DISPLAY DATA
# ============================================================

display(
    ml_df
)


# ============================================================
# DISPLAY SCHEMA
# ============================================================

ml_df.printSchema()


# ============================================================
# DISPLAY RECORD COUNT
# ============================================================

print(
    "Total Records:",
    ml_df.count()
)

# COMMAND ----------

# ============================================================
# DATASET INSPECTION
# ============================================================

print(
    "============================================"
)

print(
    "MODEL TRAINING DATASET INSPECTION"
)

print(
    "============================================"
)


# ============================================================
# TOTAL RECORDS
# ============================================================

print(
    "Total Records:",
    ml_df.count()
)


# ============================================================
# TOTAL COLUMNS
# ============================================================

print(
    "Total Columns:",
    len(ml_df.columns)
)


# ============================================================
# COLUMN NAMES
# ============================================================

print(
    "Columns:"
)

for column_name in ml_df.columns:

    print(
        column_name
    )


# ============================================================
# TARGET DISTRIBUTION
# ============================================================

print(
    "============================================"
)

print(
    "IRRIGATION LABEL DISTRIBUTION"
)

print(
    "============================================"
)

display(
    ml_df
    .groupBy(
        "irrigation_label"
    )
    .count()
    .orderBy(
        "irrigation_label"
    )
)


# ============================================================
# NULL VALUE CHECK
# ============================================================

null_counts_df = (
    ml_df
    .select(
        [
            F.sum(
                F.when(
                    F.col(c).isNull(),
                    1
                )
                .otherwise(
                    0
                )
            )
            .alias(c)
            for c in ml_df.columns
        ]
    )
)


print(
    "============================================"
)

print(
    "NULL VALUE CHECK"
)

print(
    "============================================"
)

display(
    null_counts_df
)

# COMMAND ----------

# ============================================================
# DEFINE MODEL FEATURE COLUMNS
# ============================================================

feature_columns = [

    # -----------------------------------------
    # ORIGINAL SENSOR FEATURES
    # -----------------------------------------

    "temperature",

    "humidity",

    "soil_moisture",

    "rainfall",

    "battery",


    # -----------------------------------------
    # TIME FEATURES
    # -----------------------------------------

    "reading_hour",

    "day_of_week",

    "month",

    "day_of_month",


    # -----------------------------------------
    # INTERACTION FEATURES
    # -----------------------------------------

    "temperature_humidity_interaction",

    "soil_rainfall_interaction",

    "temperature_humidity_difference",

    "soil_moisture_deficit",


    # -----------------------------------------
    # TREND FEATURES
    # -----------------------------------------

    "temperature_change",

    "soil_moisture_change",


    # -----------------------------------------
    # ROLLING FEATURES
    # -----------------------------------------

    "temperature_rolling_avg",

    "humidity_rolling_avg",

    "soil_moisture_rolling_avg",

    "rainfall_rolling_avg"
]


# ============================================================
# TARGET COLUMN
# ============================================================

target_column = (
    "irrigation_label"
)


# ============================================================
# SELECT MODEL DATA
# ============================================================

model_df = (
    ml_df
    .select(
        *feature_columns,
        target_column
    )
)


# ============================================================
# DISPLAY MODEL DATA
# ============================================================

display(
    model_df
)


# ============================================================
# MODEL DATA SUMMARY
# ============================================================

print(
    "Total Features:",
    len(feature_columns)
)

print(
    "Target Column:",
    target_column
)

print(
    "Total Model Records:",
    model_df.count()
)

# COMMAND ----------

# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

train_df, test_df = (
    model_df
    .randomSplit(
        [0.8, 0.2],
        seed=42
    )
)


# ============================================================
# DISPLAY RECORD COUNTS
# ============================================================

print(
    "============================================"
)

print(
    "TRAIN / TEST SPLIT"
)

print(
    "============================================"
)

print(
    "Training Records:",
    train_df.count()
)

print(
    "Testing Records:",
    test_df.count()
)


# ============================================================
# DISPLAY TRAINING DATA
# ============================================================

display(
    train_df
)


# ============================================================
# DISPLAY TESTING DATA
# ============================================================

display(
    test_df
)

# COMMAND ----------

# ============================================================
# VECTOR ASSEMBLER
# ============================================================

vector_assembler = (
    VectorAssembler(
        inputCols=feature_columns,
        outputCol="features"
    )
)


# ============================================================
# CREATE FEATURE VECTORS
# ============================================================

train_vector_df = (
    vector_assembler
    .transform(
        train_df
    )
)


test_vector_df = (
    vector_assembler
    .transform(
        test_df
    )
)


# ============================================================
# SELECT MODEL-READY COLUMNS
# ============================================================

train_vector_df = (
    train_vector_df
    .select(
        "features",
        target_column
    )
)


test_vector_df = (
    test_vector_df
    .select(
        "features",
        target_column
    )
)


# ============================================================
# DISPLAY TRAINING FEATURE VECTORS
# ============================================================

display(
    train_vector_df
)


# ============================================================
# DISPLAY TESTING FEATURE VECTORS
# ============================================================

display(
    test_vector_df
)


# ============================================================
# RECORD COUNTS
# ============================================================

print(
    "Training Vector Records:",
    train_vector_df.count()
)

print(
    "Testing Vector Records:",
    test_vector_df.count()
)

# COMMAND ----------

# ============================================================
# RANDOM FOREST CLASSIFIER
# ============================================================

random_forest_model = (
    RandomForestClassifier(
        labelCol=target_column,
        featuresCol="features",
        numTrees=100,
        maxDepth=10,
        seed=42
    )
)


# ============================================================
# TRAIN MODEL
# ============================================================

rf_model = (
    random_forest_model
    .fit(
        train_vector_df
    )
)


# ============================================================
# MODEL TRAINING SUCCESS
# ============================================================

print(
    "Random Forest Model Trained Successfully"
)


# ============================================================
# DISPLAY MODEL INFORMATION
# ============================================================

print(
    "Number of Trees:",
    rf_model.getNumTrees
)


print(
    "Number of Features:",
    len(feature_columns)
)

# COMMAND ----------

# ============================================================
# MAKE PREDICTIONS ON TEST DATA
# ============================================================

predictions_df = (
    rf_model
    .transform(
        test_vector_df
    )
)


# ============================================================
# SELECT PREDICTION RESULTS
# ============================================================

predictions_df = (
    predictions_df
    .select(
        target_column,
        "prediction",
        "probability",
        "rawPrediction"
    )
)


# ============================================================
# DISPLAY PREDICTIONS
# ============================================================

display(
    predictions_df
)


# ============================================================
# TOTAL PREDICTIONS
# ============================================================

print(
    "Total Predictions:",
    predictions_df.count()
)

# COMMAND ----------

# ============================================================
# MODEL ACCURACY EVALUATION
# ============================================================

accuracy_evaluator = (
    MulticlassClassificationEvaluator(
        labelCol=target_column,
        predictionCol="prediction",
        metricName="accuracy"
    )
)


# ============================================================
# CALCULATE ACCURACY
# ============================================================

accuracy = (
    accuracy_evaluator
    .evaluate(
        predictions_df
    )
)


# ============================================================
# DISPLAY ACCURACY
# ============================================================

print(
    "============================================"
)

print(
    "RANDOM FOREST MODEL ACCURACY"
)

print(
    "============================================"
)

print(
    "Accuracy:",
    round(
        accuracy,
        4
    )
)


print(
    "Accuracy Percentage:",
    round(
        accuracy * 100,
        2
    ),
    "%"
)

print(
    "============================================"
)

# COMMAND ----------

# ============================================================
# PRECISION EVALUATION
# ============================================================

precision_evaluator = (
    MulticlassClassificationEvaluator(
        labelCol=target_column,
        predictionCol="prediction",
        metricName="weightedPrecision"
    )
)


precision = (
    precision_evaluator
    .evaluate(
        predictions_df
    )
)


# ============================================================
# RECALL EVALUATION
# ============================================================

recall_evaluator = (
    MulticlassClassificationEvaluator(
        labelCol=target_column,
        predictionCol="prediction",
        metricName="weightedRecall"
    )
)


recall = (
    recall_evaluator
    .evaluate(
        predictions_df
    )
)


# ============================================================
# F1 SCORE EVALUATION
# ============================================================

f1_evaluator = (
    MulticlassClassificationEvaluator(
        labelCol=target_column,
        predictionCol="prediction",
        metricName="f1"
    )
)


f1_score = (
    f1_evaluator
    .evaluate(
        predictions_df
    )
)


# ============================================================
# DISPLAY MODEL METRICS
# ============================================================

print(
    "============================================"
)

print(
    "RANDOM FOREST MODEL EVALUATION"
)

print(
    "============================================"
)

print(
    "Precision:",
    round(
        precision,
        4
    )
)


print(
    "Recall:",
    round(
        recall,
        4
    )
)


print(
    "F1 Score:",
    round(
        f1_score,
        4
    )
)


print(
    "============================================"
)

# COMMAND ----------

# ============================================================
# FEATURE IMPORTANCE ANALYSIS
# ============================================================

feature_importance_values = (
    rf_model
    .featureImportances
    .toArray()
)


# ============================================================
# CREATE FEATURE IMPORTANCE DATA
# ============================================================

feature_importance_data = [
    
    (
        str(feature_name),
        float(importance_score)
    )
    
    for feature_name, importance_score in zip(
        feature_columns,
        feature_importance_values
    )
]


# ============================================================
# CREATE FEATURE IMPORTANCE DATAFRAME
# ============================================================

feature_importance_df = (
    spark
    .createDataFrame(
        feature_importance_data,
        [
            "feature_name",
            "importance_score"
        ]
    )
)


# ============================================================
# SORT FEATURES BY IMPORTANCE
# ============================================================

feature_importance_df = (
    feature_importance_df
    .orderBy(
        F.col(
            "importance_score"
        ).desc()
    )
)


# ============================================================
# DISPLAY ALL FEATURE IMPORTANCE
# ============================================================

display(
    feature_importance_df
)


# ============================================================
# DISPLAY TOP 5 FEATURES
# ============================================================

print(
    "============================================"
)

print(
    "TOP 5 IMPORTANT FEATURES"
)

print(
    "============================================"
)

display(
    feature_importance_df
    .limit(5)
)

# COMMAND ----------

# ============================================================
# CREATE VOLUME FOR MODEL STORAGE
# ============================================================

spark.sql(
    """
    CREATE VOLUME IF NOT EXISTS
    agriculture_iot.feature_engineering.model_storage
    """
)


# ============================================================
# MODEL SAVE PATH
# ============================================================

MODEL_PATH = (
    "/Volumes/agriculture_iot/"
    "feature_engineering/"
    "model_storage/"
    + MODEL_NAME
)


# ============================================================
# SAVE TRAINED RANDOM FOREST MODEL
# ============================================================

(
    rf_model
    .write()
    .overwrite()
    .save(
        MODEL_PATH
    )
)


# ============================================================
# SUCCESS MESSAGE
# ============================================================

print(
    "============================================"
)

print(
    "RANDOM FOREST MODEL SAVED SUCCESSFULLY"
)

print(
    "============================================"
)

print(
    "Model Name:",
    MODEL_NAME
)

print(
    "Model Path:",
    MODEL_PATH
)

print(
    "============================================"
)

# COMMAND ----------

# ============================================================
# FINAL MODEL TRAINING SUMMARY
# ============================================================

print(
    "============================================"
)

print(
    "AGRICULTURE IOT MODEL TRAINING SUMMARY"
)

print(
    "============================================"
)


# ============================================================
# DATASET SUMMARY
# ============================================================

print(
    "Total Model Records:",
    model_df.count()
)

print(
    "Training Records:",
    train_df.count()
)

print(
    "Testing Records:",
    test_df.count()
)


# ============================================================
# MODEL INFORMATION
# ============================================================

print(
    "============================================"
)

print(
    "MODEL INFORMATION"
)

print(
    "============================================"
)

print(
    "Model Name:",
    MODEL_NAME
)

print(
    "Algorithm: Random Forest Classifier"
)

print(
    "Number of Features:",
    len(feature_columns)
)


# ============================================================
# MODEL PERFORMANCE
# ============================================================

print(
    "============================================"
)

print(
    "MODEL PERFORMANCE"
)

print(
    "============================================"
)

print(
    "Accuracy:",
    round(
        accuracy,
        4
    )
)

print(
    "Precision:",
    round(
        precision,
        4
    )
)

print(
    "Recall:",
    round(
        recall,
        4
    )
)

print(
    "F1 Score:",
    round(
        f1_score,
        4
    )
)


# ============================================================
# MODEL STORAGE
# ============================================================

print(
    "============================================"
)

print(
    "MODEL STORAGE"
)

print(
    "============================================"
)

print(
    "Model Path:",
    MODEL_PATH
)


# ============================================================
# FINAL STATUS
# ============================================================

print(
    "============================================"
)

print(
    "MODEL TRAINING COMPLETED SUCCESSFULLY"
)

print(
    "============================================"
)