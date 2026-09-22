# Databricks notebook source
from pyspark.sql import functions as F
from pyspark.sql.window import Window


# ============================================================
# SOURCE TABLE
# ============================================================

SILVER_VALID_SENSOR = "agriculture_iot.silver.valid_sensor"


# ============================================================
# REQUIRED GOLD TABLES
# ============================================================

GOLD_FARM_DAILY_METRICS = (
    "agriculture_iot.gold.gold_farm_daily_metrics"
)

GOLD_SENSOR_HEALTH = (
    "agriculture_iot.gold.gold_sensor_health"
)

GOLD_IRRIGATION_ALERTS = (
    "agriculture_iot.gold.gold_irrigation_alerts"
)

GOLD_CROP_CONDITIONS = (
    "agriculture_iot.gold.gold_crop_conditions"
)

GOLD_ENVIRONMENT_TREND = (
    "agriculture_iot.gold.gold_environment_trend"
)

# COMMAND ----------

silver_df = (
    spark.table(SILVER_VALID_SENSOR)
    
    .withColumn(
        "timestamp",
        F.to_timestamp("timestamp")
    )
    
    .withColumn(
        "reading_date",
        F.to_date("timestamp")
    )
    
    .withColumn(
        "reading_hour",
        F.date_trunc("hour", F.col("timestamp"))
    )
)

display(silver_df.limit(20))

# COMMAND ----------

enriched_df = (
    silver_df

    # ========================================================
    # TEMPERATURE STATUS
    # ========================================================

    .withColumn(
        "temperature_status",
        F.when(
            F.col("temperature") >= 40,
            F.lit("HIGH")
        )
        .when(
            F.col("temperature") <= 5,
            F.lit("LOW")
        )
        .otherwise(
            F.lit("NORMAL")
        )
    )

    # ========================================================
    # SOIL STATUS
    # ========================================================

    .withColumn(
        "soil_status",
        F.when(
            F.col("soil_moisture") < 30,
            F.lit("LOW")
        )
        .when(
            F.col("soil_moisture") > 80,
            F.lit("HIGH")
        )
        .otherwise(
            F.lit("NORMAL")
        )
    )

    # ========================================================
    # HUMIDITY STATUS
    # ========================================================

    .withColumn(
        "humidity_status",
        F.when(
            F.col("humidity") < 30,
            F.lit("LOW")
        )
        .when(
            F.col("humidity") > 80,
            F.lit("HIGH")
        )
        .otherwise(
            F.lit("NORMAL")
        )
    )

    # ========================================================
    # IRRIGATION REQUIREMENT
    # ========================================================

    .withColumn(
        "irrigation_required",
        F.when(
            (
                (F.col("soil_moisture") < 30)
                &
                (F.col("rainfall") < 2)
            ),
            F.lit("YES")
        )
        .otherwise(
            F.lit("NO")
        )
    )

    # ========================================================
    # SENSOR HEALTH
    # ========================================================

    .withColumn(
        "sensor_health",
        F.when(
            F.col("battery") < 20,
            F.lit("LOW_BATTERY")
        )
        .when(
            F.col("battery") < 50,
            F.lit("WARNING")
        )
        .otherwise(
            F.lit("HEALTHY")
        )
    )
)

display(enriched_df.limit(20))

# COMMAND ----------

sensor_window = (
    Window
    .partitionBy("farm_id", "sensor_id")
    .orderBy("timestamp")
)

daily_window = (
    Window
    .partitionBy("farm_id", "reading_date")
)


time_series_df = (
    enriched_df

    # ========================================================
    # PREVIOUS TEMPERATURE
    # ========================================================

    .withColumn(
        "previous_temperature",
        F.lag("temperature", 1).over(sensor_window)
    )

    # ========================================================
    # TEMPERATURE CHANGE
    # ========================================================

    .withColumn(
        "temperature_change",
        F.round(
            F.col("temperature")
            - F.col("previous_temperature"),
            2
        )
    )

    # ========================================================
    # 3 READING MOVING AVERAGE
    # ========================================================

    .withColumn(
        "3_reading_average",
        F.round(
            F.avg("temperature").over(
                sensor_window.rowsBetween(-2, 0)
            ),
            2
        )
    )

    # ========================================================
    # DAILY AVERAGE
    # ========================================================

    .withColumn(
        "daily_average",
        F.round(
            F.avg("temperature").over(daily_window),
            2
        )
    )

    # ========================================================
    # DAILY MINIMUM
    # ========================================================

    .withColumn(
        "daily_min",
        F.round(
            F.min("temperature").over(daily_window),
            2
        )
    )

    # ========================================================
    # DAILY MAXIMUM
    # ========================================================

    .withColumn(
        "daily_max",
        F.round(
            F.max("temperature").over(daily_window),
            2
        )
    )
)

display(time_series_df.limit(20))

# COMMAND ----------

gold_farm_daily_metrics_df = (
    time_series_df

    .groupBy(
        "farm_id",
        "reading_date"
    )

    .agg(
        F.count("reading_id").alias(
            "total_readings"
        ),

        F.countDistinct("sensor_id").alias(
            "active_sensors"
        ),

        F.round(
            F.avg("temperature"),
            2
        ).alias("avg_temperature"),

        F.round(
            F.min("temperature"),
            2
        ).alias("min_temperature"),

        F.round(
            F.max("temperature"),
            2
        ).alias("max_temperature"),

        F.round(
            F.avg("humidity"),
            2
        ).alias("avg_humidity"),

        F.round(
            F.avg("soil_moisture"),
            2
        ).alias("avg_soil_moisture"),

        F.round(
            F.sum("rainfall"),
            2
        ).alias("total_rainfall"),

        F.round(
            F.avg("battery"),
            2
        ).alias("avg_battery")
    )

    .withColumn(
        "temperature_status",
        F.when(
            F.col("avg_temperature") >= 40,
            "HIGH"
        )
        .when(
            F.col("avg_temperature") <= 5,
            "LOW"
        )
        .otherwise("NORMAL")
    )

    .withColumn(
        "soil_status",
        F.when(
            F.col("avg_soil_moisture") < 30,
            "LOW"
        )
        .when(
            F.col("avg_soil_moisture") > 80,
            "HIGH"
        )
        .otherwise("NORMAL")
    )

    .withColumn(
        "irrigation_required",
        F.when(
            (
                (F.col("avg_soil_moisture") < 30)
                &
                (F.col("total_rainfall") < 2)
            ),
            "YES"
        )
        .otherwise("NO")
    )

    .withColumn(
        "_gold_processed_timestamp",
        F.current_timestamp()
    )
)

# COMMAND ----------

gold_sensor_health_df = (
    time_series_df

    .groupBy(
        "farm_id",
        "sensor_id"
    )

    .agg(
        F.count("reading_id").alias(
            "total_readings"
        ),

        F.min("timestamp").alias(
            "first_reading_timestamp"
        ),

        F.max("timestamp").alias(
            "last_reading_timestamp"
        ),

        F.round(
            F.avg("battery"),
            2
        ).alias(
            "avg_battery"
        ),

        F.round(
            F.min("battery"),
            2
        ).alias(
            "min_battery"
        ),

        F.round(
            F.avg("temperature"),
            2
        ).alias(
            "avg_temperature"
        )
    )

    .withColumn(
        "sensor_health",

        F.when(
            F.col("min_battery") < 20,
            "LOW_BATTERY"
        )

        .when(
            F.col("avg_battery") < 50,
            "WARNING"
        )

        .otherwise(
            "HEALTHY"
        )
    )

    .withColumn(
        "_gold_processed_timestamp",
        F.current_timestamp()
    )
)

# COMMAND ----------

# DBTITLE 1,Reference note
# gold_farm_daily_metrics_df is defined in Cell 5
# This cell intentionally left for reference

# COMMAND ----------

(
    gold_sensor_health_df
    .write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(GOLD_SENSOR_HEALTH)
)

print("gold_sensor_health created successfully")

display(
    spark.table(GOLD_SENSOR_HEALTH)
)

# COMMAND ----------

(
    gold_farm_daily_metrics_df
    .write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(GOLD_FARM_DAILY_METRICS)
)

print("gold_farm_daily_metrics created successfully")

display(
    spark.table(GOLD_FARM_DAILY_METRICS)
)

# COMMAND ----------

gold_irrigation_alerts_df = (
    time_series_df

    .filter(
        F.col("irrigation_required") == "YES"
    )

    .select(
        "farm_id",
        "sensor_id",
        "timestamp",
        "reading_date",
        "reading_hour",
        "temperature",
        "humidity",
        "soil_moisture",
        "rainfall",
        "battery",
        "soil_status",
        "irrigation_required"
    )

    .withColumn(
        "alert_status",
        F.lit("IRRIGATION_ALERT")
    )

    .withColumn(
        "_gold_processed_timestamp",
        F.current_timestamp()
    )
)

# COMMAND ----------

(
    gold_irrigation_alerts_df
    .write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(GOLD_IRRIGATION_ALERTS)
)

print("gold_irrigation_alerts created successfully")

display(
    spark.table(GOLD_IRRIGATION_ALERTS)
)

# COMMAND ----------

gold_crop_conditions_df = (
    time_series_df

    .groupBy(
        "farm_id",
        "reading_date"
    )

    .agg(
        F.round(
            F.avg("temperature"),
            2
        ).alias("avg_temperature"),

        F.round(
            F.avg("humidity"),
            2
        ).alias("avg_humidity"),

        F.round(
            F.avg("soil_moisture"),
            2
        ).alias("avg_soil_moisture"),

        F.round(
            F.sum("rainfall"),
            2
        ).alias("total_rainfall")
    )

    .withColumn(
        "temperature_status",
        F.when(
            F.col("avg_temperature") >= 40,
            "HIGH"
        )
        .when(
            F.col("avg_temperature") <= 5,
            "LOW"
        )
        .otherwise("NORMAL")
    )

    .withColumn(
        "humidity_status",
        F.when(
            F.col("avg_humidity") < 30,
            "LOW"
        )
        .when(
            F.col("avg_humidity") > 80,
            "HIGH"
        )
        .otherwise("NORMAL")
    )

    .withColumn(
        "soil_status",
        F.when(
            F.col("avg_soil_moisture") < 30,
            "LOW"
        )
        .when(
            F.col("avg_soil_moisture") > 80,
            "HIGH"
        )
        .otherwise("NORMAL")
    )

    .withColumn(
        "_gold_processed_timestamp",
        F.current_timestamp()
    )
)

# COMMAND ----------

(
    gold_crop_conditions_df
    .write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(GOLD_CROP_CONDITIONS)
)

print("gold_crop_conditions created successfully")

display(
    spark.table(GOLD_CROP_CONDITIONS)
)

# COMMAND ----------

from pyspark.sql import functions as F

from pyspark.sql.types import (

    NumericType,

    StringType,

    DateType,

    TimestampType

)
 

numeric_columns = [

    "temperature",

    "humidity",

    "soil_moisture",

    "rainfall",

    "battery",

    "temperature_change",

    "3_reading_average",

    "daily_average",

    "daily_min",

    "daily_max"

]
 
time_series_clean_df = time_series_df.fillna(

    0,

    subset=numeric_columns

)
 

 
gold_environment_trend_df = (

    time_series_clean_df

    .groupBy(

        "farm_id",

        "sensor_id",

        "reading_hour"

    )

    .agg(

        F.coalesce(

            F.round(F.avg("temperature"), 2),

            F.lit(0)

        ).alias("avg_temperature"),
 
        F.coalesce(

            F.round(F.avg("humidity"), 2),

            F.lit(0)

        ).alias("avg_humidity"),
 
        F.coalesce(

            F.round(F.avg("soil_moisture"), 2),

            F.lit(0)

        ).alias("avg_soil_moisture"),
 
        F.coalesce(

            F.round(F.avg("rainfall"), 2),

            F.lit(0)

        ).alias("avg_rainfall"),
 
        F.coalesce(

            F.round(F.avg("battery"), 2),

            F.lit(0)

        ).alias("avg_battery"),
 
        F.coalesce(

            F.round(F.avg("temperature_change"), 2),

            F.lit(0)

        ).alias("avg_temperature_change"),
 
        F.coalesce(

            F.round(F.avg("3_reading_average"), 2),

            F.lit(0)

        ).alias("three_reading_temperature_avg"),
 
        F.coalesce(

            F.round(F.avg("daily_average"), 2),

            F.lit(0)

        ).alias("daily_temperature_avg"),
 
        F.coalesce(

            F.min("daily_min"),

            F.lit(0)

        ).alias("daily_temperature_min"),
 
        F.coalesce(

            F.max("daily_max"),

            F.lit(0)

        ).alias("daily_temperature_max")

    )

    .withColumn(

        "reading_date",

        F.coalesce(

            F.to_date("reading_hour"),

            F.lit("1900-01-01").cast("date")

        )

    )

    .withColumn(

        "_gold_processed_timestamp",

        F.current_timestamp()

    )

)
 


gold_environment_trend_df = gold_environment_trend_df.fillna(

    0,

    subset=[

        field.name

        for field in gold_environment_trend_df.schema.fields

        if isinstance(field.dataType, NumericType)

    ]

)
 


gold_environment_trend_df = gold_environment_trend_df.fillna(

    "UNKNOWN",

    subset=[

        field.name

        for field in gold_environment_trend_df.schema.fields

        if isinstance(field.dataType, StringType)

    ]

)
 

 
(

    gold_environment_trend_df

    .write

    .format("delta")

    .mode("overwrite")

    .option("overwriteSchema", "true")

    .saveAsTable(

        "agriculture_iot.gold.gold_environment_trend"

    )

)
 


# COMMAND ----------

(
    gold_environment_trend_df
    .write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(GOLD_ENVIRONMENT_TREND)
)

print("gold_environment_trend created successfully")

display(
    spark.table(GOLD_ENVIRONMENT_TREND)
)

# COMMAND ----------

gold_tables = [
    GOLD_FARM_DAILY_METRICS,
    GOLD_SENSOR_HEALTH,
    GOLD_IRRIGATION_ALERTS,
    GOLD_CROP_CONDITIONS,
    GOLD_ENVIRONMENT_TREND
]

for table_name in gold_tables:

    print("=" * 60)
    print("TABLE:", table_name)

    try:
        record_count = spark.table(table_name).count()
        print("RECORD COUNT:", record_count)
        display(spark.table(table_name))
    except Exception as e:
        print("ERROR:", str(e))

    print("=" * 60)

# COMMAND ----------

spark.sql("""
SHOW TABLES IN agriculture_iot.gold
""").show(truncate=False)