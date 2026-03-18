import requests
import json
import re
import pandas as pd
from datetime import datetime, timedelta
import time
import random
import os
from collections import Counter
from urllib.parse import quote_plus
import aiohttp
import asyncio

class YouTubeTrendingHashtagsScraper:
    def __init__(self, output_dir="./trending_hashtags"):
        """Initialize the scraper with output directory."""
        self.output_dir = output_dir
        self.session = requests.Session()
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Connection': 'keep-alive',
            'Cache-Control': 'max-age=0'
        }
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
    def _get_random_delay(self):
        """Get a random delay between requests to avoid detection."""
        return random.uniform(1, 3)
    
    async def search_youtube(self, query):
        """Asynchronously search YouTube with the given query."""
        search_url = f"https://www.youtube.com/results?search_query={quote_plus(query)}"
        
        async with aiohttp.ClientSession() as session:
            try:
                print(f"Searching YouTube for: {query}")
                async with session.get(search_url, headers=self.headers) as response:
                    response.raise_for_status()
                    response_text = await response.text()
                    
                    # Extract initial data from the response
                    initial_data_match = re.search(r'ytInitialData\s*=\s*({.+?});</script>', response_text)
                    if not initial_data_match:
                        print("Could not find YouTube initial data")
                        return []
                        
                    initial_data = json.loads(initial_data_match.group(1))
                    
                    # Extract video information
                    video_renderer_path = [
                        'contents', 'twoColumnSearchResultsRenderer', 
                        'primaryContents', 'sectionListRenderer', 
                        'contents', 0, 'itemSectionRenderer', 'contents'
                    ]
                    
                    # Navigate through the nested structure
                    contents = initial_data
                    for key in video_renderer_path:
                        if isinstance(key, int):
                            if key < len(contents):
                                contents = contents[key]
                            else:
                                print(f"Index {key} out of range")
                                return []
                        else:
                            if key in contents:
                                contents = contents[key]
                            else:
                                print(f"Key {key} not found")
                                return []
                    
                    # Extract hashtags from video titles and descriptions
                    hashtags = set()  # Use a set to avoid duplicates
                    for item in contents:
                        if 'videoRenderer' in item:
                            video = item['videoRenderer']
                            
                            # Extract title and description
                            title = self._extract_text(video.get('title', {}))
                            description = self._extract_text(video.get('descriptionSnippet', {}))
                            
                            # Extract hashtags from title and description
                            hashtags.update(self._extract_hashtags(title))
                            hashtags.update(self._extract_hashtags(description))
                    
                    return list(hashtags)  # Convert back to list for consistency
            
            except Exception as e:
                print(f"Error searching YouTube: {str(e)}")
                return []
    
    def _extract_text(self, text_obj):
        """Extract text from YouTube's nested text objects."""
        if not text_obj:
            return ""
        
        if 'runs' in text_obj:
            return " ".join(run.get('text', '') for run in text_obj['runs'])
        elif 'simpleText' in text_obj:
            return text_obj['simpleText']
        
        return ""
    
    def _extract_hashtags(self, text):
        """Extract hashtags from text."""
        if not text:
            return []
            
        # Match hashtags that start with # and contain letters, numbers, or underscores
        return re.findall(r'#(\w+)', text)
    
    async def fetch_hashtags(self, session, query):
        """Asynchronously fetch hashtags for a given query."""
        return await self.search_youtube(query)
    
    async def get_trending_hashtags(self, search_queries, days=7):
        """Get trending hashtags from YouTube searches over the last X days."""
        all_hashtags = []
        
        # Get dates for the last X days
        today = datetime.now()
        date_ranges = [(today - timedelta(days=i)).strftime('%Y-%m-%d') for i in range(days)]
        
        print(f"Scraping trending hashtags for the last {days} days")
        
        async with aiohttp.ClientSession() as session:
            tasks = []
            
            # For each search query
            for query in search_queries:
                print(f"\nSearching for query: {query}")
                
                # For each date range
                for date_str in date_ranges:
                    print(f"  Searching date: {date_str}")
                    
                    # Create a task for fetching hashtags
                    tasks.append(self.fetch_hashtags(session, query))
            
            # Gather all tasks and wait for them to complete
            results = await asyncio.gather(*tasks)
            
            # Process results
            for query, hashtags in zip(search_queries, results):
                for hashtag in hashtags:
                    all_hashtags.append({
                        'hashtag': hashtag,
                        'date': date_str,
                        'query': query
                    })
        
        # Convert to DataFrame
        df = pd.DataFrame(all_hashtags)
        
        # Calculate frequencies
        hashtag_counts = Counter(df['hashtag'])
        
        # Create frequency dataframe
        freq_df = pd.DataFrame({
            'hashtag': list(hashtag_counts.keys()),
            'frequency': list(hashtag_counts.values())
        })
        
        # Sort by frequency
        freq_df = freq_df.sort_values('frequency', ascending=False)
        
        # Get top hashtags
        top_hashtags = freq_df.head(50)
        
        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save detailed results
        df.to_csv(f"{self.output_dir}/hashtags_detailed_{timestamp}.csv", index=False)
        
        # Save frequency results
        freq_df.to_csv(f"{self.output_dir}/hashtags_frequency_{timestamp}.csv", index=False)
        
        # Save top hashtags to text file
        top_hashtags_file = f"{self.output_dir}/trending_hashtags_{timestamp}.txt"
        with open(top_hashtags_file, 'w', encoding='utf-8') as f:
            f.write(f"Top Trending YouTube Hashtags (Last {days} Days)\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            for i, row in top_hashtags.iterrows():
                f.write(f"#{row['hashtag']} ({row['frequency']} occurrences)\n")
        
        print(f"\nAnalysis complete! Results saved to {self.output_dir}")
        print(f"Top hashtags file: {top_hashtags_file}")
        
        return top_hashtags_file

def main():
    # Define search queries to use
    search_queries = [
    "trending",
    "viral videos",
    "popular youtube",
    "trending now",
    "latest trends",
    "trending reels",
    "popular content",
    "trending shorts",
    "viral challenges",
    "trending challenge",
    "trending now worldwide",
    "viral moments 2025",
    "hottest trends today",
    "must-watch videos",
    "top trending content",
    "latest viral sensations",
    "best trending videos",
    "social media trends",
    "daily viral clips",
    "popular trends this week",
    "latest celebrity news",
    "hollywood trending now",
    "bollywood trending today",
    "best Netflix shows 2025",
    "must-watch movies this month",
    "viral celebrity gossip",
    "music videos trending",
    "top 10 songs this week",
    "Grammy award winners 2025",
    "most popular music right now",
    "latest smartphone reviews",
    "best tech gadgets 2025",
    "trending AI news",
    "must-have apps this year",
    "Apple vs Samsung latest",
    "top gaming laptops 2025",
    "VR and AR trends",
    "next-gen console wars",
    "Elon Musk latest updates",
    "best budget smartphones",
    "top trending games",
    "must-watch gaming streams",
    "most popular Twitch games",
    "new game releases this month",
    "esports tournaments 2025",
    "best mobile games today",
    "GTA 6 latest news",
    "Fortnite viral clips",
    "Minecraft trending builds",
    "Call of Duty best plays",
    "best home workouts",
    "fitness trends this year",
    "must-try weight loss diets",
    "gym motivation videos",
    "viral yoga routines",
    "intermittent fasting benefits",
    "bodybuilding tips 2025",
    "mental health awareness trends",
    "best superfoods right now",
    "nutrition tips for longevity",
    "NASA latest discoveries",
    "SpaceX new mission",
    "trending science breakthroughs",
    "AI and robotics advancements",
    "future of space travel",
    "black hole latest research",
    "time travel theories trending",
    "medical innovations 2025",
    "most fascinating science facts",
    "climate change updates",
    "stock market trends today",
    "top investment strategies",
    "crypto latest news",
    "Bitcoin price prediction 2025",
    "best side hustle ideas",
    "how to make money online",
    "top startups this year",
    "passive income strategies",
    "business trends 2025",
    "real estate investment tips",
    "best travel destinations 2025",
    "top budget travel hacks",
    "solo travel tips trending",
    "best road trips this year",
    "Airbnb vs hotels debate",
    "digital nomad lifestyle",
    "latest fashion trends",
    "skincare routine trending",
    "best perfumes for men & women",
    "most popular luxury brands",
    "best coding languages to learn",
    "AI in education",
    "online courses trending now",
    "top productivity hacks",
    "best books to read 2025",
    "memory enhancement techniques",
    "free coding resources",
    "viral TED Talks this year",
    "best study techniques",
    "time management tips",
    "trending recipes this week",
    "must-try street food",
    "viral cooking hacks",
    "best air fryer recipes",
    "top food trends 2025",
    "healthy meal prep ideas",
    "coffee brewing techniques",
    "most delicious desserts",
    "Michelin star restaurants trending",
    "plant-based diet benefits"
    ]

    
    # Create scraper
    output_dir = os.path.expanduser("~/Downloads/youtube_trending_hashtags")
    scraper = YouTubeTrendingHashtagsScraper(output_dir)
    
    # Get trending hashtags for the last 7 days
    results_file = asyncio.run(scraper.get_trending_hashtags(search_queries, days=10))
    
    print(f"\nTrending hashtags have been saved to: {results_file}")
    
    # Optional: Display top 10 hashtags
    try:
        with open(results_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            print("\nTop 10 Trending Hashtags:")
            for line in lines[3:13]:  # Skip header lines and get top 10
                print(line.strip())
    except Exception as e:
        print(f"Error displaying top hashtags: {str(e)}")

if __name__ == "__main__":
    main()