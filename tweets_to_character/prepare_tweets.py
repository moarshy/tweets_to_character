from pathlib import Path
import json
from typing import Dict, List, Set, Optional, Any
import zipfile
import asyncio
from datetime import datetime
import os
import logging

logger = logging.getLogger(__name__)


def unzip_file(zip_file: str, output_folder: str) -> None:
    # unzip the file into a folder
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(output_folder)

    return output_folder


def get_account_data(output_folder: Path) -> dict:
    """Extract and parse account data from Twitter archive folder."""
    try:
        # Construct path to account.js
        account_data_js = output_folder / 'data' / 'account.js'
        
        # Read and parse the file
        with open(account_data_js, 'r') as f:
            account_data = json.loads(f.read().replace('window.YTD.account.part0 = ', ''))
            
        return account_data
    except Exception as e:
        logger.error(f"Error parsing account data: {str(e)}")
        raise


def get_tweets(output_folder: Path) -> list:
    """Extract and parse tweets from Twitter archive folder."""
    try:
        # Construct path to tweets.js
        tweets_js = output_folder / 'data' / 'tweets.js'
        
        # Read and parse the file
        with open(tweets_js, 'r') as f:
            tweet_data = json.loads(f.read().replace('window.YTD.tweets.part0 = ', ''))
            
        # Extract just the tweet objects and filter out retweets
        tweets = [
            item['tweet'] for item in tweet_data
            if not item['tweet'].get('retweeted', False)
        ]
        
        return tweets
    except Exception as e:
        logger.error(f"Error parsing tweets: {str(e)}")
        raise


async def build_conversation_thread(tweet: dict, tweets: list, account_data: dict) -> str:
    """Build a conversation thread from a tweet and its replies."""
    thread = []
    visited = set()

    async def process_thread(current_tweet: dict) -> None:
        if not current_tweet:
            return
            
        if current_tweet['id_str'] in visited:
            return
            
        visited.add(current_tweet['id_str'])
        thread.insert(0, current_tweet)  # equivalent to unshift
        
        if current_tweet.get('in_reply_to_status_id_str'):
            reply_to_tweet = next(
                (t for t in tweets 
                 if t['id_str'] == current_tweet['in_reply_to_status_id_str']),
                None
            )
            await process_thread(reply_to_tweet)

    await process_thread(tweet)
    
    # Remove duplicates while preserving order using id_str as key
    seen = set()
    thread = [t for t in thread if not (t['id_str'] in seen or seen.add(t['id_str']))]
    
    # Sort by timestamp
    # Changed format to match Twitter's format
    thread.sort(
        key=lambda t: datetime.strptime(t['created_at'], '%a %b %d %H:%M:%S +0000 %Y').timestamp()
    )
    
    # Build conversation text
    conversation_text = []
    for t in thread:
        post = []
        post.append(f"From: {account_data[0]['account']['accountDisplayName']} (@{account_data[0]['account']['username']})")
        post.append(f"Tweet ID: {t['id_str']}")
        
        if t.get('in_reply_to_status_id_str'):
            post.append(f"In Reply To: {t['in_reply_to_status_id_str']}")
            
        # Changed format to match Twitter's format
        timestamp = datetime.strptime(t['created_at'], '%a %b %d %H:%M:%S +0000 %Y')
        post.append(f"Timestamp: {timestamp.strftime('%c')}")
        post.append("Content:")
        post.append(t['full_text'])
        post.append("---")
        
        conversation_text.append('\n'.join(post))
    
    return '\n\n'.join(conversation_text)


async def chunk_text(tweets: list, account_data: dict, archive_path: str) -> list:
    """Split tweets into manageable chunks for processing."""
    chunks = []
    CHUNK_SIZE = 60000  # 50k tokens approx
    
    if isinstance(tweets, list):
        for i in range(0, len(tweets), 1000):
            tweet_chunk = tweets[i:i + 1000]
            
            # Process batch of tweets
            conversation_threads = await asyncio.gather(
                *[build_conversation_thread(tweet, tweets, account_data) 
                  for tweet in tweet_chunk]
            )
            
            current_chunk = ""
            
            # Process each thread in the batch
            for thread in conversation_threads:
                # If single thread is larger than chunk size, add as separate chunk
                if len(thread) > CHUNK_SIZE:
                    chunks.append(thread)
                    continue
                
                # If adding thread would exceed chunk size, save current and start new
                if len(current_chunk) + len(thread) > CHUNK_SIZE:
                    chunks.append(current_chunk)
                    current_chunk = ""
                
                current_chunk += thread
            
            # Add any remaining content
            if current_chunk:
                chunks.append(current_chunk)

    return chunks