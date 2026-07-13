import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the cleaned dataset
df = pd.read_csv("cleaned_sentiment_dataset.csv")

# display basic information
print(df.head())

print("\nDataset Information")
print(df.info())

print("\nShape of Dataset")
print(df.shape)

# Summary statistics
print("\nSummary Statistics")
print(df.describe())

print("\nMedian")
print(df.median(numeric_only=True))

print("\nMode")
print(df.mode().iloc[0])

print("\nStandard Deviation")
print(df.std(numeric_only=True))

# Correlation between numerical features
numeric_df = df.select_dtypes(include=['number'])

correlation = numeric_df.corr()

print(correlation)

# Correlation heatmap
plt.figure(figsize=(8,6))
sns.heatmap(correlation, annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

#Histogram
plt.figure(figsize=(7,5))
plt.hist(df["Likes"], bins=20)
plt.title("Distribution of Likes")
plt.xlabel("Likes")
plt.ylabel("Frequency")
plt.show()

# Boxplot
plt.figure(figsize=(7,5))
sns.boxplot(y=df["Likes"])
plt.title("Box Plot of Likes")
plt.show()

# sactter plot
plt.figure(figsize=(7,5))
plt.scatter(df["Likes"], df["Retweets"])
plt.xlabel("Likes")
plt.ylabel("Retweets")
plt.title("Likes vs Retweets")
plt.show()

#Sentiment distribution
plt.figure(figsize=(6,4))
sns.countplot(x="Sentiment", data=df)
plt.title("Sentiment Distribution")
plt.show()

#platform distribution
plt.figure(figsize=(6,4))
sns.countplot(x="Platform", data=df)
plt.title("Platform Distribution")
plt.xticks(rotation=45)
plt.show()

#EDA results
summary = df.describe()

summary.to_csv("summary_statistics.csv")
