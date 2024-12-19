import os
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
import logging
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder
from pyspark.ml.tuning import TrainValidationSplit
import cloudpickle
from pyspark.ml.feature import OneHotEncoder

# -------------------------
# Initialize Logging
# -------------------------
logging.basicConfig(
    level=logging.INFO,  # Set log level to INFO
    format="%(asctime)s - %(levelname)s - %(message)s",  # Log format
    datefmt="%Y-%m-%d %H:%M:%S"  # Date format
)
logger = logging.getLogger(__name__)  # Create a logger instance

# -------------------------
# Environment Variables
# -------------------------


os.environ["JAVA_HOME"] = r"X:\java\jdk-11"
os.environ["HADOOP_HOME"] = r"X:\hadoop"
os.environ["SPARK_LOCAL_DIRS"] = "X:\\tmp\\spark-temp"
# Ensure spark.local.dir exists (managed by YARN)
os.makedirs("X:/tmp/spark-temp", exist_ok=True)


# -------------------------
# Initialize Spark Session
# -------------------------
spark = SparkSession.builder \
    .appName("NFL_Predict_Ball_Handler_GPU") \
    .config("spark.executor.memory", "24g") \
    .config("spark.driver.memory", "8g") \
    .config("spark.executor.cores", "2") \
    .config("spark.executor.instances", "3") \
    .config("spark.task.cpus", "1") \
    .config("spark.local.dir", "X:\\tmp\\spark-temp") \
    .config("spark.sql.shuffle.partitions", "200") \
    .getOrCreate()


# -------------------------
# Load Datasets
# -------------------------
play_data = spark.read.csv("X:/Coding/Python/Projects/NFL Big Data Bowl 2025/plays.csv", header=True, inferSchema=True)
player_play_data = spark.read.csv("X:/Coding/Python/Projects/NFL Big Data Bowl 2025/player_play.csv", header=True, inferSchema=True)
player_data = spark.read.csv("X:/Coding/Python/Projects/NFL Big Data Bowl 2025/players.csv", header=True, inferSchema=True)
game_data = spark.read.csv("X:/Coding/Python/Projects/NFL Big Data Bowl 2025/games.csv", header=True, inferSchema=True)
tracking_data = spark.read.csv("X:/Coding/Python/Projects/NFL Big Data Bowl 2025/tracking_week_1.csv", header=True, inferSchema=True)
tracking_data2 = spark.read.csv("X:/Coding/Python/Projects/NFL Big Data Bowl 2025/tracking_week_2.csv", header=True, inferSchema=True)
tracking_data3 = spark.read.csv("X:/Coding/Python/Projects/NFL Big Data Bowl 2025/tracking_week_3.csv", header=True, inferSchema=True)
tracking_data4 = spark.read.csv("X:/Coding/Python/Projects/NFL Big Data Bowl 2025/tracking_week_4.csv", header=True, inferSchema=True)
tracking_data5 = spark.read.csv("X:/Coding/Python/Projects/NFL Big Data Bowl 2025/tracking_week_5.csv", header=True, inferSchema=True)
tracking_data6 = spark.read.csv("X:/Coding/Python/Projects/NFL Big Data Bowl 2025/tracking_week_6.csv", header=True, inferSchema=True)
tracking_data7 = spark.read.csv("X:/Coding/Python/Projects/NFL Big Data Bowl 2025/tracking_week_7.csv", header=True, inferSchema=True)
tracking_data8 = spark.read.csv("X:/Coding/Python/Projects/NFL Big Data Bowl 2025/tracking_week_8.csv", header=True, inferSchema=True)
tracking_data9 = spark.read.csv("X:/Coding/Python/Projects/NFL Big Data Bowl 2025/tracking_week_9.csv", header=True, inferSchema=True)

tracking_data = tracking_data.union(tracking_data2).union(tracking_data3).union(tracking_data4).union(tracking_data5).union(tracking_data6).union(tracking_data7).union(tracking_data8).union(tracking_data9)
# 1. Extract Date from Time Column
tracking_data = tracking_data.withColumn(
    "play_date", F.to_date(F.col("time"))  # Extract yyyy-mm-dd
)

play_dates = tracking_data.select("gameId", "playId", "play_date").distinct()

# Add play_date to play_data
play_data = play_data.join(play_dates, on=["gameId", "playId"], how="left")
player_play_data = player_play_data.join(play_dates, on=["gameId", "playId"], how="left")

# -------------------------
# Add frameType Column (before_snap)
# -------------------------
# Define the list of snap events
snap_events = ['autoevent_ballsnap', 'ball_snap']

snap_frames = tracking_data.filter(F.col("event").isin(snap_events)) \
    .groupBy("gameId", "playId", "play_date" ) \
    .agg(F.min("frameId").alias("snap_frame"))

tracking_data = tracking_data.join(snap_frames, on=["gameId", "playId", "play_date"], how="left")

tracking_data = tracking_data.withColumn(
    "frameType",
    F.when(F.col("frameId") < F.col("snap_frame"), "before_snap")
    .otherwise(F.lit(None))  # Leave null for other rows
)

tracking_data = tracking_data.drop("snap_frame")


# -------------------------
# Filter out defensive players
# -------------------------
player_data_filtered = player_data.select("nflId", "position")

play_data_alias = play_data.alias("play")
tracking_data_alias = tracking_data.alias("track")
player_data_filtered_alias = player_data_filtered.alias("player_filter")


# Filter offensive players only (exclude football and null nflId)
offensive_players = tracking_data_alias.join(
    play_data_alias,
    on=["gameId", "playId", "play_date"],  # Join keys
    how="inner"
).filter(
    (F.col("track.club") != "football") &                  # Exclude football entity
    (F.col("track.nflId").isNotNull())                     # Exclude null nflId rows
)



offensive_players = offensive_players.join(
    player_data_filtered_alias,
    on="nflId",  # Join on nflId
    how="inner"  # Only include matching nflId rows
)



offensive_players = offensive_players.filter(
    F.col("position").isin(["QB", "RB", "FB", "TE", "WR"])  # Include only offensive positions
)
#Higher count because multiple frames foreach play
#offensive_players.groupBy("position").count().show()


# -------------------------
# Join player_play_data with tracking_data
# -------------------------

# Filter pre-snap frames for offensive players only
presnap_data = offensive_players.filter(F.col("frameType") == "before_snap")

# Assign aliases
presnap_data_alias = presnap_data.alias("pd")
player_play_data_alias = player_play_data.alias("ppd")

# Perform the join with aliases
combined_data = presnap_data_alias.join(
    player_play_data_alias,
    on=["gameId", "playId", "nflId", "play_date"],  # Join keys
    how="left"
)

#Reduced position count because only last frame is used
#combined_data.groupBy("position").count().show()


 
# -------------------------
# Extract Features
# -------------------------

# 1. Final Positions (last x, y before snap)
final_positions = combined_data.groupBy("gameId", "playId", "nflId", "play_date") \
    .agg(F.max("frameId").alias("final_frame"))

# Join to get final x and y positions
final_positions_joined = final_positions.join(
    combined_data.alias("cd"),
    (final_positions["gameId"] == F.col("cd.gameId")) &
    (final_positions["playId"] == F.col("cd.playId")) &
    (final_positions["nflId"] == F.col("cd.nflId")) &
    (final_positions["play_date"] == F.col("cd.play_date")) &
    (final_positions["final_frame"] == F.col("cd.frameId")),
    "left"
).select(
    final_positions["gameId"],
    final_positions["playId"],
    final_positions["nflId"],
    final_positions["play_date"],
    F.col("cd.x").alias("final_x"),
    F.col("cd.y").alias("final_y"),
    F.col("cd.position")
)

# 2. Movement Trends
movement_trends = combined_data.groupBy("gameId", "playId", "nflId", "play_date") \
    .agg(
        F.avg("s").alias("mean_speed"),
        F.max("s").alias("max_speed"),
        F.avg("a").alias("mean_acceleration"),
        F.max("a").alias("max_acceleration"),
        F.sum("dis").alias("total_distance"),
        F.avg("dir").alias("mean_direction"),
        (F.max("dir") - F.min("dir")).alias("change_in_direction")
    )

# 3. Combine Features
# Join final positions with movement trends
features_joined = final_positions_joined.join(
    movement_trends.alias("mt"),
    on=["gameId", "playId", "nflId", "play_date"],
    how="inner"
)

# Select motion and shift indicators from player_play_data
motion_shift_data = player_play_data_alias.select(
    "gameId",
    "playId",
    "nflId",
    "play_date",
    "inMotionAtBallSnap",
    "shiftSinceLineset",
).distinct()



    
# Join with motion and shift indicators
final_features = features_joined.join(
    motion_shift_data.alias("msd"),
    on=["gameId", "playId", "nflId", "play_date"],
    how="left"
).select(
    "gameId",
    "playId",
    "nflId",
    "play_date",
    "final_x",
    "final_y",
    "mean_speed",
    "max_speed",
    "mean_acceleration",
    "max_acceleration",
    "total_distance",
    "mean_direction",
    "change_in_direction",
    "inMotionAtBallSnap",
    "shiftSinceLineset",
    "position"
)

# Add offenseFormation and receiverAlignment to final_features
final_features = final_features.join(
    play_data.select("gameId", "playId", "play_date", "offenseFormation", "receiverAlignment"),
    on=["gameId", "playId", "play_date"],
    how="left"
).select(
    "gameId",
    "playId",
    "nflId",
    "play_date",
    "final_x",
    "final_y",
    "mean_speed",
    "max_speed",
    "mean_acceleration",
    "max_acceleration",
    "total_distance",
    "mean_direction",
    "change_in_direction",
    "inMotionAtBallSnap",
    "shiftSinceLineset",
    "offenseFormation",
    "receiverAlignment",
    "position"
)



# -------------------------
# Show Results
# -------------------------
#logger.info("Final aggregated features including motion and shift indicators:")
#final_features.show(10)



 #================================================ ================================================ ================================================
 #=====  Aggregate Offensive Team Features  ====== =====  Aggregate Offensive Team Features  ======  =====  Aggregate Offensive Team Features  =====
 #================================================ ================================================ ================================================






''''

# Extract ball position with play_date included
ball_position = tracking_data.filter(
    (F.col("club") == "football") & (F.col("frameType") == "before_snap")
).select(
    "gameId", "playId", "play_date", "x", "y"
).withColumnRenamed("x", "ball_x").withColumnRenamed("y", "ball_y")

ball_position = ball_position.groupBy("gameId", "playId", "play_date") \
    .agg(F.first("ball_x").alias("ball_x"), F.first("ball_y").alias("ball_y"))

# Join ball position to offensive player data, including play_date
team_features = final_features.join(
    ball_position,
    on=["gameId", "playId", "play_date"],  # Include play_date in join keys
    how="left"
).withColumn(
    "x_relative", F.col("final_x") - F.col("ball_x")
).withColumn(
    "y_relative", F.col("final_y") - F.col("ball_y")
)

# 2. Add player positions for WR/RB identification, keeping play_date
team_features = team_features.join(
    player_data[["nflId", "position"]],
    on="nflId",
    how="left"
)

# Aggregate position-based features, grouped by play_date
team_summary = team_features.groupBy("gameId", "playId", "play_date").agg(
    # Count WRs to the left and right of the ball
    F.sum(F.when((F.col("position") == "WR") & (F.col("y_relative") < 0), 1).otherwise(0)).alias("num_WR_left"),
    F.sum(F.when((F.col("position") == "WR") & (F.col("y_relative") > 0), 1).otherwise(0)).alias("num_WR_right"),

    # Average RB depth (distance behind the ball)
    F.avg(F.when(F.col("position") == "RB", F.col("x_relative")).otherwise(None)).alias("rb_avg_depth"),

    # Number of players in motion
    F.sum(F.when(F.col("inMotionAtBallSnap") == True, 1).otherwise(0)).alias("num_players_in_motion"),

    # Offensive horizontal spread (range of y_relative positions)
    (F.max("y_relative") - F.min("y_relative")).alias("offense_horizontal_spread"),

    # Total distance traveled by all players
    F.sum("total_distance").alias("team_total_distance"),
        
    F.avg(F.when(F.col("position").isin("C","G","T"), F.col("x_relative"))).alias("avg_ol_x"),
    F.avg(F.when(F.col("position") == "TE", F.col("x_relative"))).alias("avg_te_x")
)

team_summary = team_summary.withColumn(
    "te_position_relative_to_ol", F.col("avg_te_x") - F.col("avg_ol_x")
)

# 3. Add `offenseFormation` and `receiverAlignment` from play data, grouped by play_date
team_summary = team_summary.join(
    play_data.select("gameId", "playId", "offenseFormation", "receiverAlignment"),
    on=["gameId", "playId"],  # Note: play_date is only in tracking_data
    how="left"
)


# -------------------------
# Show Aggregated Team Features
# -------------------------
#logger.info("Aggregated offensive team features with play_date included:")
#team_summary.show(10)

# Group by gameId and playId to calculate the number of TEs per play
two_te_plays = team_features.groupBy("gameId", "playId", "play_date") \
    .agg(F.sum(F.when(F.col("position") == "TE", 1).otherwise(0)).alias("num_TEs")) \
    .filter(F.col("num_TEs") == 2)

# Count the number of plays with exactly two TEs
two_te_count = two_te_plays.count()

# Log the result
#logger.info(f"Number of plays with exactly two TEs: {two_te_count}")
'''

 #================================================ ================================================ ================================================
 #=====  Training model  ====== =====  Training model  ======  =====  Training model  =====
 #================================================ ================================================ ================================================

# Select relevant columns for labeling
label_data = player_play_data.select(
    "gameId",
    "playId",
    "nflId",
    "play_date",
    "wasTargettedReceiver",  # Corrected the column name
    "hadRushAttempt"
)

# Create 'gotBall' label
label_data = label_data.withColumn(
    "gotBall", 
    (F.col("wasTargettedReceiver") == True) | (F.col("hadRushAttempt") == True)
)

# Join with feature data
labeled_data = final_features.join(
    label_data,
    on=["gameId", "playId", "nflId", "play_date"],
    how="left"
).fillna({"gotBall": False})

# Convert boolean inMotionAtBallSnap to numeric (0/1)
labeled_data = labeled_data.withColumn("inMotionAtBallSnap_numeric", F.when(F.col("inMotionAtBallSnap") == True, 1.0).otherwise(0.0))
# Convert boolean column to numeric
labeled_data = labeled_data.withColumn("shiftSinceLineset_numeric",F.when(F.col("shiftSinceLineset") == True, 1.0).otherwise(0.0))

# Handle categorical features with StringIndexer
offenseFormationIndexer = StringIndexer(inputCol="offenseFormation", outputCol="offenseFormation_indexed", handleInvalid="keep")
receiverAlignmentIndexer = StringIndexer(inputCol="receiverAlignment", outputCol="receiverAlignment_indexed", handleInvalid="keep")
positionIndexer = StringIndexer(inputCol="position", outputCol="position_indexed", handleInvalid="keep")

# OneHotEncode the indexed categorical features
offenseFormationEncoder = OneHotEncoder(inputCols=["offenseFormation_indexed"], outputCols=["offenseFormation_vec"])
receiverAlignmentEncoder = OneHotEncoder(inputCols=["receiverAlignment_indexed"], outputCols=["receiverAlignment_vec"])
positionEncoder = OneHotEncoder(inputCols=["position_indexed"], outputCols=["position_vec"]) 

# Define the feature columns
numeric_features = [
    "final_x", "final_y", "mean_speed", "max_speed", "mean_acceleration", "max_acceleration",
    "total_distance", "mean_direction", "change_in_direction", "shiftSinceLineset_numeric", "inMotionAtBallSnap_numeric"
]

# Feature assembler to combine all features into a single vector
assembler = VectorAssembler(
    inputCols=numeric_features + ["offenseFormation_vec", "receiverAlignment_vec", "position_vec"],
    outputCol="features"
)

# Convert gotBall to a numeric label (0/1)
labeled_data = labeled_data.withColumn("label", F.when(F.col("gotBall") == True, 1.0).otherwise(0.0))

# Set the checkpoint directory




labeled_data.cache()

# Split into train and test
train_data, test_data = labeled_data.randomSplit([0.8, 0.2], seed=42)

positive_count = train_data.filter(F.col("label") == 1).count()
negative_count = train_data.filter(F.col("label") == 0).count()
total_count = positive_count + negative_count
weight_for_positive = total_count / (2 * positive_count)
weight_for_negative = total_count / (2 * negative_count)

# Add weight column to training data
train_data = train_data.withColumn(
    "classWeight",
    F.when(F.col("label") == 1, weight_for_positive).otherwise(weight_for_negative)
)

print(f"Positive Samples: {positive_count} ({positive_count/total_count*100:.2f}%)")
print(f"Negative Samples: {negative_count} ({negative_count/total_count*100:.2f}%)")

# Define the classifier (example: RandomForest)
rf = RandomForestClassifier(featuresCol="features", labelCol="label", weightCol="classWeight", seed=42)

# Build the pipeline
pipeline = Pipeline(stages=[
    offenseFormationIndexer,
    receiverAlignmentIndexer,
    positionIndexer,
    offenseFormationEncoder,
    receiverAlignmentEncoder,
    positionEncoder,
    assembler,
    rf
])

## Define an evaluator (for binary classification)
evaluator = BinaryClassificationEvaluator(labelCol="label", metricName="areaUnderROC") #change to areaUnderPr later on 

# Create a parameter grid to tune RandomForest hyperparameters
paramGrid = (ParamGridBuilder()
             .addGrid(rf.numTrees, [200])          # Number of trees in the forest
             .addGrid(rf.maxDepth, [10])           # Maximum depth of each tree
             .addGrid(rf.maxBins, [128])
             .addGrid(rf.featureSubsetStrategy, ["sqrt"]) 
             .addGrid(rf.minInstancesPerNode, [50])        
             .build())

# Use TrainValidationSplit for hyperparameter tuning
tvs = TrainValidationSplit(
    estimator=pipeline,              # The pipeline to tune
    estimatorParamMaps=paramGrid,    # Parameter grid
    evaluator=evaluator,             # Evaluator for binary classification
    trainRatio=0.8,                  # 80% training data, 20% test data
    parallelism=2                    # Number of parallel tasks
)

# Fit TrainValidationSplit on the training data
tvModel = tvs.fit(train_data)

# The best model is available as tvModel.bestModel
# Evaluate on test data
# For test data
test_predictions = tvModel.transform(test_data)
test_auc = evaluator.evaluate(test_predictions)
print(f"Test AUC: {test_auc}")

# For training data
train_predictions = tvModel.transform(train_data)
train_auc = evaluator.evaluate(train_predictions)
print(f"Train AUC: {train_auc}")


''''
# Adjust the threshold to favor recall
threshold = 0.3
test_predictions_with_threshold = test_predictions.withColumn(
    "customPrediction",
    F.when(F.col("probability")[1] >= threshold, 1.0).otherwise(0.0)  # Custom threshold for positive class
)
'''

# Inspect the best model parameters
bestModel = tvModel.bestModel
best_rf_model = tvModel.bestModel.stages[-1]  # Assuming RandomForestClassifier is the last stage in the pipeline

# Extract the number of trees and maximum depth
best_num_trees = best_rf_model.getNumTrees # Access as an attribute, not a method
best_max_depth = best_rf_model.getMaxDepth()
best_min_instances_per_node = best_rf_model.getMinInstancesPerNode() # Access as a method
best_max_bin = best_rf_model.getMaxBins()

# Print the parameters
print(f"Best number of trees: {best_num_trees}")
print(f"Best maximum depth: {best_max_depth}")
print(f"Best min instance per node: {best_min_instances_per_node}")
print(f"Best max bins: {best_max_bin}")

tp = test_predictions.filter((F.col("label") == 1) & (F.col("prediction") == 1)).count()
fp = test_predictions.filter((F.col("label") == 0) & (F.col("prediction") == 1)).count()
tn = test_predictions.filter((F.col("label") == 0) & (F.col("prediction") == 0)).count()
fn = test_predictions.filter((F.col("label") == 1) & (F.col("prediction") == 0)).count()

# Precision, Recall, F1-Score
precision = tp / (tp + fp) if (tp + fp) != 0 else 0
recall = tp / (tp + fn) if (tp + fn) != 0 else 0
f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

# Print metrics
print(f"Precision (Test Data): {precision}")
print(f"Recall (Test Data): {recall}")
print(f"F1-Score (Test Data): {f1_score}")

# Calculate ROC-AUC for test data
test_roc_auc = evaluator.evaluate(test_predictions)
print(f"ROC-AUC (Test Data): {test_roc_auc}")

# Repeat metrics calculation for training data if needed
tp_train = train_predictions.filter((F.col("label") == 1) & (F.col("prediction") == 1)).count()
fp_train = train_predictions.filter((F.col("label") == 0) & (F.col("prediction") == 1)).count()
tn_train = train_predictions.filter((F.col("label") == 0) & (F.col("prediction") == 0)).count()
fn_train = train_predictions.filter((F.col("label") == 1) & (F.col("prediction") == 0)).count()

precision_train = tp_train / (tp_train + fp_train) if (tp_train + fp_train) != 0 else 0
recall_train = tp_train / (tp_train + fn_train) if (tp_train + fn_train) != 0 else 0
f1_score_train = 2 * (precision_train * recall_train) / (precision_train + recall_train) if (precision_train + recall_train) != 0 else 0

print(f"Precision (Train Data): {precision_train}")
print(f"Recall (Train Data): {recall_train}")
print(f"F1-Score (Train Data): {f1_score_train}")

# Calculate ROC-AUC for training data
train_roc_auc = evaluator.evaluate(train_predictions)
print(f"ROC-AUC (Train Data): {train_roc_auc}")

# then add all the rest of theweeks
# may need to cross validate the results for best hyperparmater settings then add the rest of the weeks 