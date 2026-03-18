import requests
import re
import json
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import time
import random
import os
from collections import Counter
import csv
from urllib.parse import quote_plus, urlparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from calendar import day_name
import yt_dlp
import matplotlib.dates as mdates
import shutil  # Add this import at the top of your file

class YouTubeTrendingHashtagsScraper:
    def __init__(self, output_dir="../../Downloads/trending_hashtags"):
        """Initialize the scraper with output directory."""
        self.output_dir = output_dir
        # Delete the output directory if it exists
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)  # Remove the directory and its contents
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Referer': 'https://www.youtube.com/'
        }
        self.search_terms = [
            "trending", 
            "viral", "popular", "trending now", "trending videos", 
            "trending music", "trending games", "trending entertainment",
            "trending technology", "trending sports", "trending fashion","popular trends this week",
            "latest celebrity news","hollywood trending now","bollywood trending today","best Netflix shows 2025","must-watch movies this month","viral celebrity gossip","music videos trending","top 10 songs this week","Grammy award winners 2025","most popular music right now","latest smartphone reviews","best tech gadgets 2025"
        ]
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
    def _add_date_filter(self, url, days=7):
        """Add date filter to the YouTube search URL."""
        filter_param = f"&sp=EgIIAw%253D%253D"  # Filter for last 7 days
        if "?" in url:
            return f"{url}{filter_param}"
        else:
            return f"{url}?{filter_param}"
            
    def _extract_hashtags_from_text(self, text):
        """Extract hashtags from text using regex."""
        hashtag_pattern = r'#\w+'  # Pattern to match hashtags
        hashtags = re.findall(hashtag_pattern, text)
        return [tag.lower() for tag in hashtags]  # Convert to lowercase for consistency
        
    def _extract_hashtags_from_video_data(self, json_data):
        """Extract hashtags from YouTube initial data JSON."""
        hashtags = []
        
        # Extract from video renderer objects (search results)
        try:
            if 'contents' in json_data:
                items = json_data.get('contents', {}).get('twoColumnSearchResultsRenderer', {}).get('primaryContents', {}).get('sectionListRenderer', {}).get('contents', [])
                
                for item in items:
                    if 'itemSectionRenderer' in item:
                        contents = item.get('itemSectionRenderer', {}).get('contents', [])
                        
                        for content in contents:
                            # Extract from video titles and descriptions
                            if 'videoRenderer' in content:
                                video = content.get('videoRenderer', {})
                                
                                # Title
                                title = video.get('title', {}).get('runs', [{}])[0].get('text', '')
                                hashtags.extend(self._extract_hashtags_from_text(title))
                                
                                # Description snippet
                                if 'descriptionSnippet' in video:
                                    desc = ''.join([run.get('text', '') for run in video.get('descriptionSnippet', {}).get('runs', [])])
                                    hashtags.extend(self._extract_hashtags_from_text(desc))
                                    
                                # Published time text might contain hashtags
                                if 'publishedTimeText' in video:
                                    published_text = video.get('publishedTimeText', {}).get('simpleText', '')
                                    hashtags.extend(self._extract_hashtags_from_text(published_text))
        except Exception as e:
            print(f"Error extracting hashtags from video data: {str(e)}")
        
        return hashtags
        
    def _sleep_random(self, min_seconds=2, max_seconds=5):
        """Sleep for a random amount of time to avoid rate limiting."""
        time.sleep(random.uniform(min_seconds, max_seconds))
        
    def fetch_trending_hashtags(self, days=7):
        """Fetch trending hashtags from YouTube for the last specified days."""
        all_hashtags = []
        
        for search_term in self.search_terms:
            print(f"Searching for '{search_term}'...")
            encoded_search = quote_plus(search_term)
            url = f"https://www.youtube.com/results?search_query={encoded_search}"
            url = self._add_date_filter(url, days)
            
            try:
                response = requests.get(url, headers=self.headers)
                if response.status_code == 200:
                    # Extract initial data from the page
                    data_match = re.search(r'var ytInitialData\s*=\s*({.+?});</script>', response.text)
                    if data_match:
                        json_str = data_match.group(1)
                        try:
                            json_data = json.loads(json_str)
                            hashtags = self._extract_hashtags_from_video_data(json_data)
                            all_hashtags.extend(hashtags)
                            print(f"  Found {len(hashtags)} hashtags")
                        except json.JSONDecodeError:
                            print(f"  Failed to parse JSON data for '{search_term}'")
                    else:
                        print(f"  Could not find initial data for '{search_term}'")
                else:
                    print(f"  Failed to get search results for '{search_term}': {response.status_code}")
            except Exception as e:
                print(f"  Error fetching data for '{search_term}': {str(e)}")
            
            # Add a random delay between requests
            self._sleep_random()
        
        # Count and sort hashtags
        hashtag_counter = Counter(all_hashtags)
        sorted_hashtags = hashtag_counter.most_common()
        
        # Save results
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        csv_path = os.path.join(self.output_dir, f"trending_hashtags_{timestamp}.csv")
        
        with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Hashtag', 'Count'])
            writer.writerows(sorted_hashtags)
        
        print(f"\nResults saved to {csv_path}")
        return sorted_hashtags, csv_path


class YouTubeChannelAnalyzer:
    def __init__(self, channel_url, output_dir="../../Downloads/channel_analysis"):
        """Initialize the channel analyzer with a YouTube channel URL."""
        self.channel_url = channel_url
        self.output_dir = output_dir
        self.channel_id = self._extract_channel_id(channel_url)
        self.videos_data = []
        self.channel_stats = {}
        # Delete the output directory if it exists
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)  # Remove the directory and its contents
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
    
    def _extract_channel_id(self, url):
        """Extract the channel ID from the YouTube channel URL."""
        parsed_url = urlparse(url)
        path_parts = parsed_url.path.strip('/').split('/')
        
        if 'youtube.com' in parsed_url.netloc:
            if 'channel' in path_parts:
                # URL format: youtube.com/channel/CHANNEL_ID
                return path_parts[path_parts.index('channel') + 1]
            elif 'c' in path_parts:
                # URL format: youtube.com/c/CHANNEL_NAME
                return path_parts[path_parts.index('c') + 1]
            elif 'user' in path_parts:
                # URL format: youtube.com/user/USERNAME
                return path_parts[path_parts.index('user') + 1]
            elif '@' in path_parts[0]:
                # URL format: youtube.com/@USERNAME
                return path_parts[0]
        
        # If we can't determine the channel ID, use the URL as is
        return url
    
    def _fetch_channel_videos(self, limit=150):
        """Fetch video data for the channel using yt-dlp."""
        print(f"Fetching data for channel: {self.channel_url}")
        
        ydl_opts = {
            'ignoreerrors': True,
            'extract_flat': True,
            'quiet': True,
            'skip_download': True,
            'playlistend': limit,
        }
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                # Get channel info
                channel_info = ydl.extract_info(self.channel_url, download=False)
                
                if not channel_info or 'entries' not in channel_info or not channel_info['entries']:
                    print("No videos found for this channel.")
                    return
                
                # Store channel stats
                self.channel_stats = {
                    'name': channel_info.get('title', 'Unknown'),
                    'channel_id': channel_info.get('id', 'Unknown'),
                    'description': channel_info.get('description', ''),
                    'subscriber_count': channel_info.get('subscriber_count', 0),
                    'view_count': channel_info.get('view_count', 0),
                    'video_count': channel_info.get('n_entries', 0)
                }
                
                # Get videos
                entries = list(channel_info['entries'])
                
                # Fetch detailed information for each video
                for entry in entries:
                    video_url = f"https://www.youtube.com/watch?v={entry['id']}"
                    video_info = ydl.extract_info(video_url, download=False)
                    
                    if video_info:
                        publish_date = None
                        if 'upload_date' in video_info:
                            # Format is YYYYMMDD
                            date_str = video_info['upload_date']
                            publish_date = datetime(
                                int(date_str[0:4]),
                                int(date_str[4:6]),
                                int(date_str[6:8])
                            )
                        
                        # Print only the tags from the video_info
                        tags = video_info.get('tags', [])
                        print(f"Tags for video ID {video_info.get('id', 'Unknown')}: {tags}")  # Print only tags
                        
                        self.videos_data.append({
                            'id': video_info.get('id', ''),
                            'title': video_info.get('title', ''),
                            'description': video_info.get('description', ''),
                            'publish_date': publish_date,
                            'view_count': video_info.get('view_count', 0),
                            'like_count': video_info.get('like_count', 0),
                            'comment_count': video_info.get('comment_count', 0),
                            'duration': video_info.get('duration', 0),
                            'tags': tags,  # Ensure tags are collected
                            'categories': video_info.get('categories', [])
                        })
                        
                        print(f"Fetched data for video: {video_info.get('title', 'Unknown')}")
                    else:
                        print(f"Could not fetch data for video ID: {entry['id']}")
                    
                    # Add a short delay between requests
                    time.sleep(1)
                
                print(f"Fetched data for {len(self.videos_data)} videos")
        
        except Exception as e:
            print(f"Error fetching channel videos: {str(e)}")
    
    def analyze_best_upload_times(self):
        """Analyze the best times to upload videos based on performance metrics."""
        if not self.videos_data:
            print("No video data available for analysis")
            return {}
        
        # Create a DataFrame for easier analysis
        df = pd.DataFrame(self.videos_data)
        
        # Filter out videos with missing publish dates
        df = df[df['publish_date'].notna()]
        
        if df.empty:
            print("No videos with valid publish dates")
            return {}
        
        # Add day of week and hour columns
        df['day_of_week'] = df['publish_date'].dt.day_name()
        df['hour'] = df['publish_date'].dt.hour
        
        # Calculate engagement metrics
        df['engagement_score'] = df['view_count'] * 1.0 + df['like_count'] * 5.0 + df['comment_count'] * 10.0
        df['views_per_day'] = df['view_count'] / ((datetime.now() - df['publish_date']).dt.total_seconds() / 86400)
        
        # Filter out outliers
        df = df[df['views_per_day'] < df['views_per_day'].quantile(0.95)]
        
        # Analyze best day of week
        day_performance = df.groupby('day_of_week')['engagement_score'].mean().sort_values(ascending=False)
        
        # Analyze best hour
        hour_performance = df.groupby('hour')['engagement_score'].mean().sort_values(ascending=False)
        
        # Analyze best day-hour combination
        day_hour_performance = df.groupby(['day_of_week', 'hour'])['engagement_score'].mean().sort_values(ascending=False)
        
        # Convert to more readable format
        best_days = day_performance.index.tolist()
        best_hours = hour_performance.index.tolist()
        best_combinations = [(day, hour) for day, hour in day_hour_performance.index[:5]]
        
        # Format hour ranges for readability
        hour_ranges = []
        for hour in best_hours[:3]:
            hour_ranges.append(f"{hour}:00-{(hour+1)%24}:00")
        
        combination_ranges = []
        for day, hour in best_combinations:
            combination_ranges.append(f"{day} {hour}:00-{(hour+1)%24}:00")
        
        results = {
            'best_days': best_days[:3],
            'best_hours': hour_ranges,
            'best_combinations': combination_ranges
        }
        
        return results
    
    def analyze_content_performance(self):
        """Analyze what type of content performs best."""
        if not self.videos_data:
            print("No video data available for analysis")
            return {}
        
        df = pd.DataFrame(self.videos_data)
        
        # Filter out videos with missing data
        df = df[df['view_count'].notna()]
        
        if df.empty:
            print("No videos with valid view counts")
            return {}
        
        # Analyze video duration
        df['duration_minutes'] = df['duration'] / 60
        
        # Group by duration ranges
        duration_bins = [0, 5, 10, 15, 20, 30, 60, float('inf')]
        duration_labels = ['0-5 min', '5-10 min', '10-15 min', '15-20 min', '20-30 min', '30-60 min', '60+ min']
        df['duration_range'] = pd.cut(df['duration_minutes'], bins=duration_bins, labels=duration_labels)
        
        duration_performance = df.groupby('duration_range')['view_count'].mean().sort_values(ascending=False)
        
        # Analyze tags
        all_tags = []
        for tags in df['tags']:
            if isinstance(tags, list):
                all_tags.extend(tags)  # Ensure tags are being collected correctly
        
        tag_counter = Counter(all_tags)
        top_tags = tag_counter.most_common(10)  # Get the top 10 tags
        
        # Analyze title length
        df['title_length'] = df['title'].str.len()
        title_length_corr = df['title_length'].corr(df['view_count'])
        
        # Analyze common title words
        title_words = []
        for title in df['title']:
            if isinstance(title, str):
                words = re.findall(r'\w+', title.lower())
                title_words.extend(words)
        
        word_counter = Counter(title_words)
        top_words = word_counter.most_common(10)
        
        results = {
            'best_duration_ranges': duration_performance.index.tolist(),
            'top_tags': top_tags,
            'title_length_correlation': title_length_corr,
            'top_title_words': top_words
        }
        
        return results
    
    def generate_growth_recommendations(self):
        """Generate recommendations for channel growth based on analysis."""
        upload_times = self.analyze_best_upload_times()
        content_performance = self.analyze_content_performance()
        
        recommendations = []
        
        # Upload timing recommendations
        if upload_times:
            recommendations.append(f"1. Upload Schedule: Based on your channel's performance, consider uploading videos on {', '.join(upload_times['best_days'])} between {' or '.join(upload_times['best_hours'])}.")
            recommendations.append(f"2. Optimal Posting Time: The best specific time slots for your channel are {', '.join(upload_times['best_combinations'])}.")
        
        # Content recommendations
        if content_performance:
            if 'best_duration_ranges' in content_performance and content_performance['best_duration_ranges']:
                recommendations.append(f"3. Video Length: Your audience engages best with videos of {content_performance['best_duration_ranges'][0]} length.")
            
            if 'top_tags' in content_performance and content_performance['top_tags']:
                tag_recommendations = ', '.join([f"#{tag}" for tag, _ in content_performance['top_tags'][:5]])
                recommendations.append(f"4. Effective Tags: Consider using these tags which perform well: {tag_recommendations}")
            
            if 'top_title_words' in content_performance and content_performance['top_title_words']:
                word_recommendations = ', '.join([word for word, _ in content_performance['top_title_words'][:5]])
                recommendations.append(f"5. Title Keywords: Include these high-performing keywords in your titles: {word_recommendations}")
        
        # General recommendations
        recommendations.append("6. Consistency: Maintain a regular posting schedule based on the recommended times.")
        recommendations.append("7. Audience Engagement: Respond to comments within the first 24 hours to boost engagement signals.")
        recommendations.append("8. Call-to-Actions: Include clear CTAs in your videos to increase like and comment rates.")
        recommendations.append("9. Thumbnails: Ensure your thumbnails have high contrast, clear text, and emotional appeal.")
        recommendations.append("10. Cross-Promotion: Leverage your other social media channels to promote new uploads.")
        
        return recommendations
    
    def generate_channel_report(self):
        """Generate a comprehensive channel analysis report."""
        # Fetch channel videos if not already done
        if not self.videos_data:
            self._fetch_channel_videos()
        
        # Generate analysis
        upload_times = self.analyze_best_upload_times()
        content_performance = self.analyze_content_performance()
        recommendations = self.generate_growth_recommendations()
        
        # Create report
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        report_path = os.path.join(self.output_dir, f"channel_analysis_{timestamp}.txt")
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"YouTube Channel Analysis Report\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Channel Stats
            f.write("Channel Information:\n")
            f.write(f"Channel Name: {self.channel_stats.get('name', 'Unknown')}\n")
            f.write(f"Subscriber Count: {self.channel_stats.get('subscriber_count', 'Unknown')}\n")
            f.write(f"Total Views: {self.channel_stats.get('view_count', 'Unknown')}\n")
            f.write(f"Video Count: {self.channel_stats.get('video_count', 'Unknown')}\n\n")
            
            # Upload Time Analysis
            f.write("Best Upload Times:\n")
            if upload_times:
                f.write(f"Best Days: {', '.join(upload_times['best_days'])}\n")
                f.write(f"Best Hours: {', '.join(upload_times['best_hours'])}\n")
                f.write(f"Best Day-Hour Combinations: {', '.join(upload_times['best_combinations'])}\n\n")
            else:
                f.write("Insufficient data for upload time analysis.\n\n")
            
            # Content Performance
            f.write("Content Performance:\n")
            if content_performance:
                if 'best_duration_ranges' in content_performance:
                    f.write(f"Best Video Durations: {', '.join(str(x) for x in content_performance['best_duration_ranges'][:3])}\n")
                
                if 'top_tags' in content_performance:
                    f.write("Top Performing Tags:\n")
                    for tag, count in content_performance['top_tags'][:10]:
                        f.write(f"  - {tag}: {count}\n")
                
                if 'title_length_correlation' in content_performance:
                    f.write(f"Title Length Correlation with Views: {content_performance['title_length_correlation']:.3f}\n")
                
                if 'top_title_words' in content_performance:
                    f.write("Most Common Title Words:\n")
                    for word, count in content_performance['top_title_words'][:10]:
                        f.write(f"  - {word}: {count}\n")
                f.write("\n")
            else:
                f.write("Insufficient data for content performance analysis.\n\n")
            
            # Growth Recommendations
            f.write("Growth Recommendations:\n")
            for i, recommendation in enumerate(recommendations, 1):
                f.write(f"{recommendation}\n")
        
        print(f"\nChannel analysis report saved to {report_path}")
        return report_path
    
    def generate_visualization(self):
        """Generate visualizations for channel performance."""
        if not self.videos_data:
            print("No video data available for visualization")
            return
        
        df = pd.DataFrame(self.videos_data)
        
        # Filter out videos with missing data
        df = df[df['publish_date'].notna() & df['view_count'].notna()]
        
        if df.empty:
            print("No videos with valid data for visualization")
            return
        
        # Create output directory for visualizations
        viz_dir = os.path.join(self.output_dir, "visualizations")
        os.makedirs(viz_dir, exist_ok=True)
        
        # 1. View count vs. publish date
        plt.figure(figsize=(12, 6))
        plt.scatter(df['publish_date'], df['view_count'], alpha=0.7)
        plt.title('View Count vs. Publish Date')
        plt.xlabel('Publish Date')
        plt.ylabel('View Count')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, 'views_vs_date.png'))
        plt.close()
        
        # 2. Day of week performance
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        df['day_of_week'] = pd.Categorical(df['publish_date'].dt.day_name(), categories=day_order, ordered=True)
        
        day_performance = df.groupby('day_of_week')['view_count'].mean().reindex(day_order)
        
        plt.figure(figsize=(10, 6))
        ax = day_performance.plot(kind='bar', color='skyblue')
        plt.title('Average Views by Day of Week')
        plt.xlabel('Day of Week')
        plt.ylabel('Average View Count')
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, 'day_performance.png'))
        plt.close()
        
        # 3. Hour of day performance
        df['hour'] = df['publish_date'].dt.hour
        hour_performance = df.groupby('hour')['view_count'].mean()
        
        plt.figure(figsize=(12, 6))
        ax = hour_performance.plot(kind='bar', color='lightgreen')
        plt.title('Average Views by Hour of Day')
        plt.xlabel('Hour of Day')
        plt.ylabel('Average View Count')
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, 'hour_performance.png'))
        plt.close()
        
        # 4. Video duration vs. view count
        df['duration_minutes'] = df['duration'] / 60
        
        plt.figure(figsize=(10, 6))
        plt.scatter(df['duration_minutes'], df['view_count'], alpha=0.7)
        plt.title('View Count vs. Video Duration')
        plt.xlabel('Duration (minutes)')
        plt.ylabel('View Count')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, 'duration_vs_views.png'))
        plt.close()
        
        # 5. Engagement metrics over time
        df['engagement_score'] = df['view_count'] * 1.0 + df['like_count'] * 5.0 + df['comment_count'] * 10.0
        df = df.sort_values('publish_date')
        
        plt.figure(figsize=(12, 6))
        plt.plot(df['publish_date'], df['engagement_score'], marker='o', linestyle='-', alpha=0.7)
        plt.title('Engagement Score Over Time')
        plt.xlabel('Publish Date')
        plt.ylabel('Engagement Score')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, 'engagement_over_time.png'))
        plt.close()
        
        print(f"Visualizations saved to {viz_dir}")
        return viz_dir


if __name__ == "__main__":
    # Example usage
    
    output_dir = "../../Downloads/youtube_analysis"
    # Delete the output directory if it exists
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)  # Remove the directory and its contents
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Fetch trending hashtags
    print("="*50)
    print("Step 1: Fetching trending hashtags from YouTube")
    hashtag_scraper = YouTubeTrendingHashtagsScraper(output_dir=output_dir)
    trending_hashtags, hashtags_file = hashtag_scraper.fetch_trending_hashtags(days=7)
    
    print("\nTop 10 Trending Hashtags:")
    for hashtag, count in trending_hashtags[:10]:
        print(f"  {hashtag}: {count}")
    
    # 2. Analyze channel performance
    print("\n" + "="*50)
    print("Step 2: Analyzing YouTube channel performance")
    channel_url = input("Enter your YouTube channel URL: ")
    
    channel_analyzer = YouTubeChannelAnalyzer(channel_url, output_dir=output_dir)
    report_path = channel_analyzer.generate_channel_report()
    
    print("\nChannel analysis complete!")
    print(f"Report saved to: {report_path}")
    
    # 3. Generate visualizations
    print("\n" + "="*50)
    print("Step 3: Generating performance visualizations")
    viz_dir = channel_analyzer.generate_visualization()
    
    print("\nAnalysis complete! Use the trending hashtags and channel insights to optimize your content strategy.")