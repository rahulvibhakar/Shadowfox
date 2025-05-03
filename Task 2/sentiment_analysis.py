import pandas as pd
import numpy as np
import re
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from datetime import datetime, timedelta
import random

# Download necessary NLTK resources
nltk.download('vader_lexicon', quiet=True)
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords

class TweetSentimentAnalyzer:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.sid = SentimentIntensityAnalyzer()
    
    def load_or_generate_data(self, filepath="C:/Users/Lenovo/Desktop/6th sem/shadowfox/Intermediate task/X data.csv"):
        """Load tweets from CSV file or generate sample data"""
        if filepath:
            try:
                df = pd.read_csv(filepath)
                required_columns = ['text', 'created_at']
                
                # Check if required columns exist
                if not all(col in df.columns for col in required_columns):
                    print("CSV file must contain 'text' and 'created_at' columns.")
                    return self.generate_sample_data()
                
                return df
            except Exception as e:
                print(f"Error loading CSV file: {e}")
                return self.generate_sample_data()
        else:
            return self.generate_sample_data()
    
    def generate_sample_data(self, num_samples=200):
        """Generate sample tweet data for demonstration"""
        print("Generating sample Twitter data for demonstration...")
        
        # Create sample dates spanning 10 days
        end_date = datetime.now()
        dates = [end_date - timedelta(days=i) for i in range(10)]
        
        # Topics to include in sample data
        topics = [
            "climate change", 
            "politics", 
            "technology", 
            "healthcare", 
            "education"
        ]
        
        # Sample tweets with different sentiments for each topic
        sample_tweets = {
            "climate change": [
                # Positive
                "Excited about new breakthroughs in renewable energy! #ClimateAction",
                "Trees planted today will help our planet breathe tomorrow. #ClimateAction",
                "So proud of youth activists fighting for climate justice!",
                # Neutral
                "Scientists continue to monitor sea level changes due to climate change.",
                "Climate conference scheduled for next month in Geneva. #ClimateChange",
                "Studies show varying impacts of climate change across different regions.",
                # Negative
                "Politicians continue to ignore the climate crisis. Shameful!",
                "Another devastating hurricane. How much more evidence do we need? #ClimateEmergency",
                "Frustrating to see so little progress on reducing carbon emissions."
            ],
            "politics": [
                # Mix of sentiments
                "Impressed by the bipartisan effort on the new infrastructure bill!",
                "Election results announced today. No major surprises.",
                "Disgusted by the corruption in government. We deserve better!"
            ],
            "technology": [
                # Mix of sentiments
                "This new AI assistant is amazing! Saving me hours of work.",
                "The latest phone has same features but costs more. Not worth upgrading.",
                "Interesting developments in quantum computing announced today."
            ],
            "healthcare": [
                # Mix of sentiments
                "Grateful for the healthcare workers during these challenging times!",
                "New study shows mixed results on treatment effectiveness.",
                "Medical bills are crushing families. System needs reform now!"
            ],
            "education": [
                # Mix of sentiments
                "So happy with the new school curriculum focusing on critical thinking!",
                "University enrollment numbers remain stable compared to last year.",
                "Budget cuts to education are hurting our children's future. Unacceptable!"
            ]
        }
        
        # Create sample data
        data = []
        for i in range(num_samples):
            # Select random topic
            topic = random.choice(topics)
            
            # Get tweets for that topic
            topic_tweets = sample_tweets[topic]
            
            # Select random tweet from that topic
            tweet = random.choice(topic_tweets)
            
            # Generate random engagement metrics
            likes = int(random.expovariate(1/50))  # Exponential distribution for realistic skew
            retweets = int(likes * random.uniform(0.1, 0.5))  # Retweets usually less than likes
            
            # Create sample tweet data
            data.append({
                'id': i + 1000,
                'created_at': random.choice(dates).strftime('%Y-%m-%d %H:%M:%S'),
                'username': f"user_{random.randint(1, 100)}",
                'text': tweet,
                'topic': topic,
                'likes': likes,
                'retweets': retweets
            })
        
        return pd.DataFrame(data)
    
    def preprocess_text(self, text):
        """Clean and preprocess tweet text"""
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove user mentions and hashtag symbols
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'#', '', text)
        
        # Remove special characters and convert to lowercase
        text = re.sub(r'[^\w\s]', '', text)
        text = text.lower().strip()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        return text
    
    def analyze_sentiment(self, text):
        """Analyze sentiment using VADER"""
        scores = self.sid.polarity_scores(text)
        
        # Determine sentiment category based on compound score
        if scores['compound'] >= 0.05:
            sentiment = 'Positive'
        elif scores['compound'] <= -0.05:
            sentiment = 'Negative'
        else:
            sentiment = 'Neutral'
        
        return {
            'compound': scores['compound'],
            'pos': scores['pos'],
            'neu': scores['neu'],
            'neg': scores['neg'],
            'sentiment': sentiment
        }
    
    def analyze_data(self, df):
        """Process and analyze the tweet data"""
        # Ensure created_at is in datetime format
        df['created_at'] = pd.to_datetime(df['created_at'])
        df['date'] = df['created_at'].dt.date
        
        # Preprocess text
        print("Preprocessing tweet text...")
        df['processed_text'] = df['text'].apply(self.preprocess_text)
        
        # Analyze sentiment
        print("Analyzing sentiment...")
        sentiment_results = df['processed_text'].apply(self.analyze_sentiment)
        sentiment_df = pd.DataFrame(sentiment_results.tolist())
        
        # Combine DataFrames
        df = pd.concat([df, sentiment_df], axis=1)
        
        # Add topic if not present (extract from text)
        if 'topic' not in df.columns:
            # Simple topic extraction based on keywords
            def extract_topic(text):
                text = text.lower()
                if any(word in text for word in ['climate', 'warming', 'carbon']):
                    return 'climate change'
                elif any(word in text for word in ['election', 'government', 'president', 'vote']):
                    return 'politics'
                elif any(word in text for word in ['tech', 'digital', 'ai', 'robot']):
                    return 'technology'
                elif any(word in text for word in ['health', 'doctor', 'hospital', 'medical']):
                    return 'healthcare'
                elif any(word in text for word in ['school', 'education', 'student', 'learn']):
                    return 'education'
                else:
                    return 'other'
            
            df['topic'] = df['text'].apply(extract_topic)
        
        return df
    
    def generate_insights(self, df):
        """Generate statistical insights from the analyzed data"""
        # Overall sentiment distribution
        sentiment_counts = df['sentiment'].value_counts()
        total_tweets = len(df)
        
        # Sentiment by topic
        topic_sentiment = df.groupby(['topic', 'sentiment']).size().unstack(fill_value=0)
        
        # Time trends
        df['date'] = pd.to_datetime(df['created_at']).dt.date
        time_sentiment = df.groupby(['date', 'sentiment']).size().unstack(fill_value=0)
        
        # Engagement metrics
        engagement_by_sentiment = df.groupby('sentiment')[['likes', 'retweets']].mean()
        
        # Most common words by sentiment
        def get_common_words(texts, n=10):
            all_words = ' '.join(texts).split()
            words = [word for word in all_words if word not in self.stop_words and len(word) > 2]
            word_counts = pd.Series(words).value_counts().head(n)
            return word_counts.to_dict()
        
        word_insights = {}
        for sentiment in ['Positive', 'Neutral', 'Negative']:
            texts = df[df['sentiment'] == sentiment]['processed_text']
            word_insights[sentiment] = get_common_words(texts)
        
        # Combine all insights
        insights = {
            'total_tweets': total_tweets,
            'sentiment_distribution': sentiment_counts.to_dict(),
            'sentiment_percentages': (sentiment_counts / total_tweets * 100).round(1).to_dict(),
            'sentiment_by_topic': topic_sentiment.to_dict(),
            'sentiment_over_time': time_sentiment.to_dict(),
            'engagement_metrics': engagement_by_sentiment.to_dict(),
            'common_words_by_sentiment': word_insights
        }
        
        return insights
    
    def plot_results(self, df, research_topic="Twitter Data", save_plots=True):
        """Generate visualizations for the sentiment analysis"""
        print("Generating visualizations...")
        
        # 1. Overall Sentiment Distribution (Pie Chart)
        plt.figure(figsize=(10, 6))
        sentiment_counts = df['sentiment'].value_counts()
        plt.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%',
                colors=['green', 'gray', 'red'], startangle=90)
        plt.title(f'Sentiment Distribution for "{research_topic}"', fontsize=14)
        if save_plots:
            plt.savefig('sentiment_distribution.png')
            print("- Saved sentiment distribution pie chart")
        plt.close()
        
        # 2. Sentiment Over Time (Line Plot)
        time_sentiment = df.groupby(['date', 'sentiment']).size().unstack(fill_value=0)
        
        plt.figure(figsize=(12, 6))
        time_sentiment.plot(kind='line', marker='o', ax=plt.gca())
        plt.title(f'Sentiment Trends Over Time for "{research_topic}"', fontsize=14)
        plt.xlabel('Date')
        plt.ylabel('Number of Tweets')
        plt.grid(True, alpha=0.3)
        plt.legend(title='Sentiment')
        plt.xticks(rotation=45)
        if save_plots:
            plt.savefig('sentiment_over_time.png')
            print("- Saved sentiment time trends chart")
        plt.close()
        
        # 3. Sentiment by Topic (Stacked Bar Chart)
        topic_sentiment = pd.crosstab(df['topic'], df['sentiment'])
        topic_sentiment_pct = topic_sentiment.div(topic_sentiment.sum(axis=1), axis=0)
        
        plt.figure(figsize=(12, 6))
        topic_sentiment_pct.plot(kind='barh', stacked=True, 
                               color=['red', 'gray', 'green'])
        plt.title('Sentiment Distribution by Topic', fontsize=14)
        plt.xlabel('Proportion')
        plt.ylabel('Topic')
        plt.xlim(0, 1)
        plt.grid(True, alpha=0.3)
        plt.legend(title='Sentiment')
        if save_plots:
            plt.savefig('sentiment_by_topic.png')
            print("- Saved sentiment by topic chart")
        plt.close()
        
        # 4. Word Clouds by Sentiment
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        sentiments = ['Positive', 'Neutral', 'Negative']
        
        for i, sentiment in enumerate(sentiments):
            text = ' '.join(df[df['sentiment'] == sentiment]['processed_text'])
            
            if text.strip():
                wordcloud = WordCloud(width=800, height=400,
                                    background_color='white',
                                    stopwords=self.stop_words,
                                    max_words=50,
                                    collocations=False).generate(text)
                
                axes[i].imshow(wordcloud, interpolation='bilinear')
                axes[i].set_title(f'{sentiment} Sentiment', fontsize=14)
                axes[i].axis('off')
            else:
                axes[i].text(0.5, 0.5, f"No {sentiment} tweets",
                           horizontalalignment='center', fontsize=14)
                axes[i].axis('off')
        
        plt.tight_layout()
        if save_plots:
            plt.savefig('sentiment_wordclouds.png')
            print("- Saved sentiment wordclouds")
        plt.close()
        
        # 5. Engagement Analysis (Bar Chart)
        plt.figure(figsize=(10, 6))
        engagement = df.groupby('sentiment')[['likes', 'retweets']].mean()
        engagement.plot(kind='bar', ax=plt.gca())
        plt.title('Average Engagement by Sentiment', fontsize=14)
        plt.ylabel('Average Count')
        plt.grid(True, alpha=0.3)
        if save_plots:
            plt.savefig('engagement_by_sentiment.png')
            print("- Saved engagement analysis chart")
        plt.close()
        
        return self.get_sample_tweets_by_sentiment(df)
    
    def get_sample_tweets_by_sentiment(self, df, n=3):
        """Get sample tweets for each sentiment category"""
        samples = {}
        for sentiment in ['Positive', 'Neutral', 'Negative']:
            sentiment_df = df[df['sentiment'] == sentiment]
            if len(sentiment_df) > 0:
                # Sort by engagement (likes + retweets) and get top n
                sentiment_df['engagement'] = sentiment_df['likes'] + sentiment_df['retweets']
                top_tweets = sentiment_df.sort_values('engagement', ascending=False).head(n)
                samples[sentiment] = top_tweets[['text', 'likes', 'retweets']].to_dict('records')
            else:
                samples[sentiment] = []
        return samples
    
    def save_results(self, df, filename='sentiment_analysis_results.csv'):
        """Save the analyzed data to CSV"""
        df.to_csv(filename, index=False)
        print(f"Results saved to {filename}")

def main():
    # Initialize analyzer
    analyzer = TweetSentimentAnalyzer()
    
    # Get input - either file path or use sample data
    print("\nTwitter/X Sentiment Analysis")
    print("-" * 30)
    
    choice = input("Would you like to use a CSV file with tweets? (y/n): ").strip().lower()
    
    if choice == 'y':
        filepath = input("Enter the path to your CSV file: ").strip()
        tweets_df = analyzer.load_or_generate_data(filepath)
    else:
        tweet_count = input("How many sample tweets would you like to generate? (default: 200): ").strip()
        try:
            tweet_count = int(tweet_count) if tweet_count else 200
        except ValueError:
            tweet_count = 200
        tweets_df = analyzer.generate_sample_data(tweet_count)
    
    print(f"Working with {len(tweets_df)} tweets.")
    
    # Research topic detection
    if 'topic' in tweets_df.columns and len(tweets_df['topic'].unique()) == 1:
        research_topic = tweets_df['topic'].iloc[0]
    else:
        research_topic = "Multiple Topics"
    
    # Analyze data
    analysis_df = analyzer.analyze_data(tweets_df)
    
    # Generate insights
    insights = analyzer.generate_insights(analysis_df)
    
    # Generate visualizations and get sample tweets
    sample_tweets = analyzer.plot_results(analysis_df, research_topic)
    
    # Save results to CSV
    save_choice = input("Would you like to save the analyzed data to CSV? (y/n): ").strip().lower()
    if save_choice == 'y':
        analyzer.save_results(analysis_df)
    
    # Print summary report
    print("\n" + "=" * 50)
    print(f"SENTIMENT ANALYSIS SUMMARY: {research_topic}")
    print("=" * 50)
    
    print(f"\nTotal Tweets Analyzed: {insights['total_tweets']}")
    
    print("\nSentiment Distribution:")
    for sentiment, percentage in insights['sentiment_percentages'].items():
        count = insights['sentiment_distribution'][sentiment]
        print(f"  - {sentiment}: {count} tweets ({percentage}%)")
    
    print("\nTop Topics by Sentiment:")
    for sentiment in ['Positive', 'Negative', 'Neutral']:
        if sentiment in sample_tweets:
            print(f"\n{sentiment} Tweet Examples:")
            for i, tweet in enumerate(sample_tweets[sentiment][:2], 1):
                text = tweet['text']
                text = text if len(text) <= 100 else f"{text[:97]}..."
                print(f"  {i}. {text}")
                print(f"     Likes: {tweet['likes']}, Retweets: {tweet['retweets']}")
    
    print("\nCommon Words by Sentiment:")
    for sentiment, words in insights['common_words_by_sentiment'].items():
        top_words = list(words.keys())[:5]
        print(f"  - {sentiment}: {', '.join(top_words)}")
    
    print("\nAnalysis complete! Visualizations saved as PNG files.")

if __name__ == "__main__":
    main()