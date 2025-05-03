import matplotlib.pyplot as plt
from pandas import DataFrame
import seaborn as sns

def ratingToAge(df_fifa_csv : DataFrame):
    sns.set_theme(style="whitegrid")
    
    plt.figure(figsize=(10,5))
    sns.scatterplot(x=df_fifa_csv["age"], y=df_fifa_csv["overall_rating"], alpha=0.5)
    plt.xlabel("Age")
    plt.ylabel("Overall Rating")
    plt.title("Age vs. Overall Rating")
    plt.show()


def featureCorrelation(df_fifa_csv : DataFrame):
    
    plt.figure(figsize=(10,5))
    sns.histplot(df_fifa_csv["overall_rating"], bins=30, kde=True, color='blue')
    plt.title("Distribution of Player Overall Ratings")
    plt.xlabel("Overall Rating")
    plt.ylabel("Count")
    plt.show()

    plt.figure(figsize=(12,8))
    important_features = ["overall_rating", "potential", "age", "height_cm", "weight_kgs",
                        "crossing", "finishing", "short_passing", "dribbling", "ball_control",
                        "acceleration", "sprint_speed", "agility", "reactions", "balance",
                        "shot_power", "stamina", "strength", "vision", "composure"]
    sns.heatmap(df_fifa_csv[important_features].corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Feature Correlation Heatmap")
    plt.show()