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
    plt.show(block=False)


def featureCorrelation(df_fifa_csv : DataFrame):
    
    plt.figure(figsize=(10,5))
    sns.histplot(df_fifa_csv["overall_rating"], bins=30, kde=True, color='blue')
    plt.title("Distribution of Player Overall Ratings")
    plt.xlabel("Overall Rating")
    plt.ylabel("Count")
    plt.show(block=False)

    plt.figure(figsize=(12,8))
    important_features = ["overall_rating", "potential", "age", "height_cm", "weight_kgs",
                        "crossing", "finishing", "short_passing", "dribbling", "ball_control",
                        "acceleration", "sprint_speed", "agility", "reactions", "balance",
                        "shot_power", "stamina", "strength", "vision", "composure"]
    sns.heatmap(df_fifa_csv[important_features].corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Feature Correlation Heatmap")
    plt.show(block=False)

def avgRatingByNationality(df_fifa_csv: DataFrame, top_n: int = 10):
    top_nations = df_fifa_csv['nationality'].value_counts().head(top_n).index
    avg_ratings = df_fifa_csv[df_fifa_csv['nationality'].isin(top_nations)] \
        .groupby('nationality')['overall_rating'].mean().sort_values(ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=avg_ratings.values, y=avg_ratings.index, palette="viridis")
    plt.xlabel("Average Rating")
    plt.ylabel("Nationality")
    plt.title(f"Top {top_n} Nationalities by Average Overall Rating")
    plt.show(block=True)
