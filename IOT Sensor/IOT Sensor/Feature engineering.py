# Databricks notebook source
from pyspark.sql import functions as F
from pyspark.sql.window import Window


# ============================================================
# SOURCE TABLE
# ============================================================

SILVER_VALID_SENSOR = (
    "agriculture_iot.silver.valid_sensor"
)


# ============================================================
# FEATURE TABLE
# ============================================================

FEATURE_ENGINEERING_TABLE = (
    "agriculture_iot.feature_engineering.irrigation_features"
)

# COMMAND ----------

# ============================================================
# READ SILVER VALID SENSOR DATA
# ============================================================

feature_df = (
    spark.table(
        SILVER_VALID_SENSOR
    )
)


# ============================================================
# DISPLAY DATA
# ============================================================

display(
    feature_df
)

# COMMAND ----------

# ============================================================
# DATE AND TIME FEATURES
# ============================================================

feature_df = (
    feature_df

    .withColumn(
        "reading_date",
        F.to_date(
            F.col("timestamp")
        )
    )

    .withColumn(
        "reading_hour",
        F.hour(
            F.col("timestamp")
        )
    )

    .withColumn(
        "day_of_week",
        F.dayofweek(
            F.col("timestamp")
        )
    )

    .withColumn(
        "month",
        F.month(
            F.col("timestamp")
        )
    )

    .withColumn(
        "day_of_month",
        F.dayofmonth(
            F.col("timestamp")
        )
    )
)


display(
    feature_df
)

# COMMAND ----------

# ============================================================
# ENVIRONMENTAL INTERACTION FEATURES
# ============================================================

feature_df = (
    feature_df

    # -----------------------------------------
    # TEMPERATURE × HUMIDITY
    # -----------------------------------------

    .withColumn(
        "temperature_humidity_interaction",
        F.col("temperature")
        *
        F.col("humidity")
    )


    # -----------------------------------------
    # SOIL MOISTURE × RAINFALL
    # -----------------------------------------

    .withColumn(
        "soil_rainfall_interaction",
        F.col("soil_moisture")
        *
        F.col("rainfall")
    )


    # -----------------------------------------
    # TEMPERATURE DIFFERENCE
    # -----------------------------------------

    .withColumn(
        "temperature_humidity_difference",
        F.abs(
            F.col("temperature")
            -
            F.col("humidity")
        )
    )


    # -----------------------------------------
    # SOIL MOISTURE DEFICIT
    # -----------------------------------------

    .withColumn(
        "soil_moisture_deficit",
        F.when(
            F.col("soil_moisture") < 30,
            30 - F.col("soil_moisture")
        )
        .otherwise(
            F.lit(0)
        )
    )
)


display(
    feature_df
)

# COMMAND ----------

# ============================================================
# SENSOR TREND FEATURES
# ============================================================

sensor_window = (
    Window
    .partitionBy(
        "sensor_id"
    )
    .orderBy(
        "timestamp"
    )
)


feature_df = (
    feature_df

    # -----------------------------------------
    # PREVIOUS TEMPERATURE
    # -----------------------------------------

    .withColumn(
        "previous_temperature",
        F.lag(
            "temperature"
        ).over(
            sensor_window
        )
    )


    # -----------------------------------------
    # TEMPERATURE CHANGE
    # -----------------------------------------

    .withColumn(
        "temperature_change",
        F.col("temperature")
        -
        F.col("previous_temperature")
    )


    # -----------------------------------------
    # PREVIOUS SOIL MOISTURE
    # -----------------------------------------

    .withColumn(
        "previous_soil_moisture",
        F.lag(
            "soil_moisture"
        ).over(
            sensor_window
        )
    )


    # -----------------------------------------
    # SOIL MOISTURE CHANGE
    # -----------------------------------------

    .withColumn(
        "soil_moisture_change",
        F.col("soil_moisture")
        -
        F.col("previous_soil_moisture")
    )
)


display(
    feature_df
)

# COMMAND ----------

# ============================================================
# ROLLING WINDOW
# ============================================================

rolling_window = (
    Window
    .partitionBy(
        "sensor_id"
    )
    .orderBy(
        "timestamp"
    )
    .rowsBetween(
        -2,
        0
    )
)


# ============================================================
# ROLLING FEATURES
# ============================================================

feature_df = (
    feature_df

    # -----------------------------------------
    # TEMPERATURE ROLLING AVERAGE
    # -----------------------------------------

    .withColumn(
        "temperature_rolling_avg",
        F.avg(
            "temperature"
        ).over(
            rolling_window
        )
    )


    # -----------------------------------------
    # HUMIDITY ROLLING AVERAGE
    # -----------------------------------------

    .withColumn(
        "humidity_rolling_avg",
        F.avg(
            "humidity"
        ).over(
            rolling_window
        )
    )


    # -----------------------------------------
    # SOIL MOISTURE ROLLING AVERAGE
    # -----------------------------------------

    .withColumn(
        "soil_moisture_rolling_avg",
        F.avg(
            "soil_moisture"
        ).over(
            rolling_window
        )
    )


    # -----------------------------------------
    # RAINFALL ROLLING AVERAGE
    # -----------------------------------------

    .withColumn(
        "rainfall_rolling_avg",
        F.avg(
            "rainfall"
        ).over(
            rolling_window
        )
    )
)


display(
    feature_df
)

# COMMAND ----------

# ============================================================
# CREATE IRRIGATION TARGET LABEL
# ============================================================

feature_df = (
    feature_df

    .withColumn(
        "irrigation_label",

        F.when(
            (
                F.col("soil_moisture") < 30
            )
            &
            (
                F.col("rainfall") < 2
            ),
            F.lit(1)
        )

        .otherwise(
            F.lit(0)
        )
    )
)


# ============================================================
# CHECK TARGET DISTRIBUTION
# ============================================================

display(
    feature_df
    .groupBy(
        "irrigation_label"
    )
    .count()
)

# COMMAND ----------

# ============================================================
# HANDLE NULL VALUES
# ============================================================

feature_columns = [

    "temperature",

    "humidity",

    "soil_moisture",

    "rainfall",

    "battery",

    "reading_hour",

    "day_of_week",

    "month",

    "day_of_month",

    "temperature_humidity_interaction",

    "soil_rainfall_interaction",

    "temperature_humidity_difference",

    "soil_moisture_deficit",

    "temperature_change",

    "soil_moisture_change",

    "temperature_rolling_avg",

    "humidity_rolling_avg",

    "soil_moisture_rolling_avg",

    "rainfall_rolling_avg"
]


# ============================================================
# FILL NULL VALUES
# ============================================================

feature_df = (
    feature_df
    .fillna(
        0,
        subset=feature_columns
    )
)


display(
    feature_df
)

# COMMAND ----------

# ============================================================
# SELECT FINAL ML DATASET
# ============================================================

ml_feature_df = (
    feature_df
    .select(

        # -----------------------------------------
        # IDENTIFIERS
        # -----------------------------------------

        "reading_id",

        "sensor_id",

        "farm_id",

        "timestamp",


        # -----------------------------------------
        # ORIGINAL FEATURES
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

        "rainfall_rolling_avg",


        # -----------------------------------------
        # TARGET
        # -----------------------------------------

        "irrigation_label"
    )
)


display(
    ml_feature_df
)

# COMMAND ----------

# ============================================================
# CREATE FEATURE ENGINEERING SCHEMA
# ============================================================

spark.sql(
    """
    CREATE SCHEMA IF NOT EXISTS
    agriculture_iot.feature_engineering
    """
)


# ============================================================
# WRITE FEATURE ENGINEERING TABLE
# ============================================================

(
    ml_feature_df
    .write
    .format(
        "delta"
    )
    .mode(
        "overwrite"
    )
    .option(
        "overwriteSchema",
        "true"
    )
    .saveAsTable(
        FEATURE_ENGINEERING_TABLE
    )
)


# ============================================================
# SUCCESS MESSAGE
# ============================================================

print(
    "Feature Engineering Table Created Successfully"
)


print(
    "Table:",
    FEATURE_ENGINEERING_TABLE
)

# COMMAND ----------

# ============================================================
# READ FEATURE TABLE
# ============================================================

final_feature_df = (
    spark.table(
        FEATURE_ENGINEERING_TABLE
    )
)


# ============================================================
# RECORD COUNT
# ============================================================

print(
    "Total Feature Records:",
    final_feature_df.count()
)


# ============================================================
# TARGET DISTRIBUTION
# ============================================================

display(
    final_feature_df
    .groupBy(
        "irrigation_label"
    )
    .count()
)


# ============================================================
# DISPLAY FINAL DATA
# ============================================================

display(
    final_feature_df
)