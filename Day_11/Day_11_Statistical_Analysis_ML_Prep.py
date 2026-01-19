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

-- ## Calculate statistical summaries
events = spark.table("ecommerce.bronze.events")

events.describe(["price"]).show()

-- ## Hypothesis Testing (Weekday vs Weekend)
from pyspark.sql import functions as F

weekday = events.withColumn(
    "event_date", F.to_date("event_time")
).withColumn(
    "is_weekend",
    F.dayofweek("event_date").isin([1, 7])
)

weekday.groupBy("is_weekend", "event_type").count()\
.orderBy("is_weekend", "event_type") \
.show()


-- ## Identify Correlations
events = events.withColumn(
    "conversion_rate",
    F.when(F.col("event_type") == "purchase", 1).otherwise(0)
)
events.stat.corr("price", "conversion_rate")


-- ## Feature Engineering for Machine Learning
from pyspark.sql import functions as F
from pyspark.sql.window import Window

features = (
    events
    # Create event_date FIRST
    .withColumn("event_date", F.to_date("event_time"))
    
    .withColumn("hour", F.hour("event_time"))
    .withColumn("day_of_week", F.dayofweek("event_date"))
    .withColumn("price_log", F.log(F.col("price") + 1))
    .withColumn(
        "time_since_first_view",
        F.unix_timestamp("event_time") -
        F.unix_timestamp(
            F.first("event_time").over(
                Window.partitionBy("user_id").orderBy("event_time")
            )
        )
    )
)

features.select(
    "user_id",
    "product_id",
    "brand",
    "hour",
    "day_of_week",
    "price_log",
    "time_since_first_view"
).show(10)
