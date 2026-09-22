# Databricks notebook source
from pyspark.sql import functions as F
from pyspark.sql.window import Window

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator

# COMMAND ----------

spark.sql("SHOW TABLES IN workspace.default").show(truncate=False)

# COMMAND ----------

# Load sensor data from silver layer for ML modeling
gold_df = (
    spark.table("agriculture_iot.silver.valid_sensor")
    .withColumn(
        "reading_hour",
        F.hour(F.col("timestamp"))
    )
)

required_columns = [
    "farm_id",
    "reading_hour",
    "temperature",
    "humidity",
    "soil_moisture",
    "rainfall",
    "battery"
]

missing_columns = [
    column
    for column in required_columns
    if column not in gold_df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

print("All required columns are available.")

# COMMAND ----------

ml_base_df = (
    gold_df
    .withColumn(
        "reading_hour",
        F.to_timestamp("reading_hour")
    )
    .withColumn(
        "temperature",
        F.col("temperature").cast("double")
    )
    .withColumn(
        "humidity",
        F.col("humidity").cast("double")
    )
    .withColumn(
        "soil_moisture",
        F.col("soil_moisture").cast("double")
    )
    .withColumn(
        "rainfall",
        F.col("rainfall").cast("double")
    )
    .withColumn(
        "battery",
        F.col("battery").cast("double")
    )
)

display(
    ml_base_df
    .orderBy("farm_id", "reading_hour")
    .limit(10)
)

# COMMAND ----------

time_window = (
    Window
    .partitionBy("farm_id")
    .orderBy("reading_hour")
)

ml_features_df = (
    ml_base_df
    .withColumn(
        "soil_moisture_lag_1",
        F.lag("soil_moisture", 1).over(time_window)
    )
    .withColumn(
        "soil_moisture_lag_2",
        F.lag("soil_moisture", 2).over(time_window)
    )
    .withColumn(
        "soil_moisture_lag_3",
        F.lag("soil_moisture", 3).over(time_window)
    )
)

display(
    ml_features_df.select(
        "farm_id",
        "reading_hour",
        "soil_moisture",
        "soil_moisture_lag_1",
        "soil_moisture_lag_2",
        "soil_moisture_lag_3"
    ).limit(10)
)

# COMMAND ----------

rolling_window = (
    Window
    .partitionBy("farm_id")
    .orderBy("reading_hour")
    .rowsBetween(-3, -1)
)

ml_features_df = (
    ml_features_df
    .withColumn(
        "soil_moisture_rolling_avg_3",
        F.avg("soil_moisture").over(rolling_window)
    )
    .withColumn(
        "temperature_rolling_avg_3",
        F.avg("temperature").over(rolling_window)
    )
    .withColumn(
        "humidity_rolling_avg_3",
        F.avg("humidity").over(rolling_window)
    )
)

# COMMAND ----------

ml_features_df = (
    ml_features_df
    .withColumn(
        "hour",
        F.hour("reading_hour")
    )
    .withColumn(
        "day_of_week",
        F.dayofweek("reading_hour")
    )
    .withColumn(
        "month",
        F.month("reading_hour")
    )
)

# COMMAND ----------

ml_features_df = (
    ml_features_df
    .withColumn(
        "next_hour_soil_moisture",
        F.lead("soil_moisture", 1).over(time_window)
    )
)

display(
    ml_features_df.select(
        "farm_id",
        "reading_hour",
        "soil_moisture",
        "next_hour_soil_moisture"
    ).limit(20)
)

# COMMAND ----------

feature_columns = [
    "temperature",
    "humidity",
    "rainfall",
    "battery",
    "soil_moisture_lag_1",
    "soil_moisture_lag_2",
    "soil_moisture_lag_3",
    "soil_moisture_rolling_avg_3",
    "temperature_rolling_avg_3",
    "humidity_rolling_avg_3",
    "hour",
    "day_of_week",
    "month"
]

target_column = "next_hour_soil_moisture"

ml_dataset = (
    ml_features_df
    .dropna(
        subset=feature_columns + [target_column]
    )
)

print("ML Dataset Records:", ml_dataset.count())

display(ml_dataset.limit(10))

# COMMAND ----------

assembler = VectorAssembler(
    inputCols=feature_columns,
    outputCol="features",
    handleInvalid="skip"
)

ml_vector_df = assembler.transform(ml_dataset)

display(
    ml_vector_df.select(
        "farm_id",
        "reading_hour",
        "features",
        "next_hour_soil_moisture"
    ).limit(10)
)

# COMMAND ----------

split_window = (
    Window
    .orderBy("reading_hour", "farm_id")
)

numbered_df = (
    ml_vector_df
    .withColumn(
        "row_number",
        F.row_number().over(split_window)
    )
)

total_records = numbered_df.count()

if total_records < 10:
    raise ValueError(
        f"Not enough records for ML training. Records available: {total_records}"
    )

split_point = int(total_records * 0.8)

train_df = (
    numbered_df
    .filter(
        F.col("row_number") <= split_point
    )
    .drop("row_number")
)

test_df = (
    numbered_df
    .filter(
        F.col("row_number") > split_point
    )
    .drop("row_number")
)

print("Total Records:", total_records)
print("Training Records:", train_df.count())
print("Testing Records:", test_df.count())

# COMMAND ----------

rf = RandomForestRegressor(
    featuresCol="features",
    labelCol="next_hour_soil_moisture",
    predictionCol="prediction",
    numTrees=100,
    maxDepth=10,
    seed=42
)

rf_model = rf.fit(train_df)

print("Random Forest Model Training Completed Successfully!")

# COMMAND ----------

predictions_df = rf_model.transform(test_df)

display(
    predictions_df.select(
        "farm_id",
        "reading_hour",
        "soil_moisture",
        "next_hour_soil_moisture",
        "prediction"
    )
    .orderBy("farm_id", "reading_hour")
    .limit(50)
)

# COMMAND ----------

mae_evaluator = RegressionEvaluator(
    labelCol="next_hour_soil_moisture",
    predictionCol="prediction",
    metricName="mae"
)

rmse_evaluator = RegressionEvaluator(
    labelCol="next_hour_soil_moisture",
    predictionCol="prediction",
    metricName="rmse"
)

r2_evaluator = RegressionEvaluator(
    labelCol="next_hour_soil_moisture",
    predictionCol="prediction",
    metricName="r2"
)

mae = mae_evaluator.evaluate(predictions_df)
rmse = rmse_evaluator.evaluate(predictions_df)
r2 = r2_evaluator.evaluate(predictions_df)

print("========== MODEL EVALUATION ==========")
print("MAE  :", round(mae, 4))
print("RMSE :", round(rmse, 4))
print("R²   :", round(r2, 4))
print("======================================")

# COMMAND ----------

final_prediction_df = (
    predictions_df
    .withColumn(
        "predicted_irrigation_required",
        F.when(
            F.col("prediction") < 30,
            F.lit("YES")
        ).otherwise(
            F.lit("NO")
        )
    )
)

display(
    final_prediction_df.select(
        "farm_id",
        "reading_hour",
        "soil_moisture",
        "next_hour_soil_moisture",
        F.round("prediction", 2).alias(
            "predicted_next_hour_soil_moisture"
        ),
        "predicted_irrigation_required"
    )
    .orderBy("farm_id", "reading_hour")
    .limit(50)
)

# COMMAND ----------

PREDICTION_TABLE = (
    "workspace.default.future_soil_moisture_predictions"
)

(
    final_prediction_df
    .select(
        "farm_id",
        "reading_hour",
        "temperature",
        "humidity",
        "soil_moisture",
        "rainfall",
        "battery",
        "next_hour_soil_moisture",
        F.round(
            "prediction",
            2
        ).alias("predicted_next_hour_soil_moisture"),
        "predicted_irrigation_required"
    )
    .write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(PREDICTION_TABLE)
)

print("Future prediction table created successfully!")

display(
    spark.table(PREDICTION_TABLE)
)