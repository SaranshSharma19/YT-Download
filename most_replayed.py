import json
import requests
import re
from typing import List, Tuple, Optional
def get_video_id(url: str) -> Optional[str]:
    """Extract video ID from YouTube URL."""
    if 'youtu.be' in url:
        return url.split('/')[-1]
    elif 'youtube.com' in url:
        if 'v=' in url:
            return url.split('v=')[1].split('&')[0]
    return None
def get_most_replayed_segments(video_url: str) -> List[Tuple[int, int, float]]:
    """
    Get all segments with their replay intensity scores.
    Returns list of tuples: (start_time_seconds, end_time_seconds, intensity_score)
    """
    video_id = get_video_id(video_url)
    if not video_id:
        print("Invalid YouTube URL")
        return []
    # YouTube API endpoint for getting video info
    url = f"https://www.youtube.com/watch?v={video_id}"
    try:
        # Add headers to mimic a browser request
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers)
        response_text = response.text
        # Extract the ytInitialData from the response
        data_match = re.search(r'ytInitialData\s*=\s*({.+?});', response_text)
        if not data_match:
            print("Could not find YouTube initial data")
            return []
        data = json.loads(data_match.group(1))
        mutations = data.get('frameworkUpdates', {}).get('entityBatchUpdate', {}).get('mutations', [])
        for mutation in mutations:
            if 'payload' in mutation and 'macroMarkersListEntity' in mutation['payload']:
                markers = mutation['payload']['macroMarkersListEntity']['markersList']['markers']
                segments = []
                for marker in markers:
                    start_ms = int(marker['startMillis'])
                    duration_ms = int(marker['durationMillis'])
                    intensity = float(marker['intensityScoreNormalized'])
                    # Convert to seconds
                    start_sec = start_ms / 1000
                    end_sec = (start_ms + duration_ms) / 1000
                    segments.append((start_sec, end_sec, intensity))
                # Find segments with high replay intensity (e.g., top 10% of intensities)
                threshold = sorted([s[2] for s in segments], reverse=True)[len(segments)//10]
                most_replayed = [s for s in segments if s[2] >= threshold]
                merged_segments = []
                current_segment = None
                for segment in most_replayed:
                    # print("current_segment", current_segment, "segment", segment)
                    if current_segment is None:
                        current_segment = list(segment)
                    elif segment[0] <= current_segment[1]:
                        # Merge segments
                        # print( "segment[0], current_segment[1]",segment[0], current_segment[1])
                        current_segment[1] = segment[1]
                        current_segment[2] = max(current_segment[2], segment[2])
                    else:
                        merged_segments.append(tuple(current_segment))
                        # print('merged_segments', merged_segments)
                        current_segment = list(segment)
                if current_segment:
                    merged_segments.append(tuple(current_segment))
                return merged_segments
        print("Could not find heatmap data in response")
        return []
    except Exception as e:
        print(f"Error: {str(e)}")
        return []
if __name__ == "__main__":
    video_url = 'https://youtu.be/auWx4Knoap4?si=zGi0xhWsbMIEYNNt'
    segments = get_most_replayed_segments(video_url)
    if segments:
        print("\nMost replayed segments:")
        for start, end, intensity in segments:
            minutes_start = int(start) // 60
            seconds_start = int(start) % 60
            minutes_end = int(end) // 60
            seconds_end = int(end) % 60
            print(f"Time: {minutes_start}:{seconds_start:02d} to {minutes_end}:{seconds_end:02d} (Intensity: {intensity:.2f})")
    else:
        print("Failed to get most replayed segments")