import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load cleaned dataset
df = pd.read_csv("cleaned_sentiment_dataset.csv")

# Set a clean style
sns.set(style="whitegrid")

# Bar chart_sentiment distribution
plt.figure(figsize=(6,5))
sns.countplot(data=df, x="Sentiment")
plt.title("Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Count")

plt.savefig("sentiment_bar_chart.png")
plt.show()

# Bar chart_platform distribution
plt.figure(figsize=(7,5))
sns.countplot(data=df, x="Platform")
plt.title("Platform Distribution")
plt.xlabel("Platform")
plt.ylabel("Number of Posts")
plt.xticks(rotation=45)

plt.savefig("platform_bar_chart.png")
plt.show()

# Line chart_average likes by month
monthly = df.groupby("Month")["Likes"].mean()

plt.figure(figsize=(8,5))
plt.plot(monthly.index, monthly.values, marker='o')

plt.title("Average Likes by Month")
plt.xlabel("Month")
plt.ylabel("Average Likes")

plt.savefig("likes_line_chart.png")
plt.show()

# scatter plot_likes vs retweets
plt.figure(figsize=(7,5))

plt.scatter(df["Likes"], df["Retweets"])

plt.title("Likes vs Retweets")
plt.xlabel("Likes")
plt.ylabel("Retweets")

plt.savefig("likes_retweets_scatter.png")
plt.show()
