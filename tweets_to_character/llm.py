from tqdm import tqdm
import instructor
from openai import AsyncOpenAI
from typing import List
from tweets_to_character.prompts import prompt, FormattedBio, Style, Message, MessageContent


async def run_chat_completion(messages: list, model: str = None) -> dict:
    """Run chat completion using specified AI model."""
    
    model_name = 'gpt-4o'
    client = instructor.from_openai(AsyncOpenAI())
    
    response = await client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=0,
        response_model=FormattedBio
    )

    return response
    

async def extract_info_for_chunk(chunk: str, chunk_index, account_data: dict, model: str) -> dict:
    name = account_data[0]['account']['accountDisplayName']
    username = account_data[0]['account']['username']

    result = await run_chat_completion(
        [{'role': 'user', 'content': prompt(name, username, "", chunk)}],
        model
    )

    return result


def combine_and_deduplicate(results: List[FormattedBio]) -> FormattedBio:
    """
    Combine and deduplicate results from multiple chunks.
    """
    if not results:
        return FormattedBio(
            bio='',
            lore=[],
            adjectives=[],
            topics=[],
            style=Style(
                all=[],
                chat=[],
                post=[]
            ),
            messageExamples=[],
            postExamples=[]
        )
    
    def flat_map_and_dedupe(items: List[FormattedBio], attr: str) -> List[str]:
        """Helper function to flatten and deduplicate list attributes."""
        flattened = []
        for item in items:
            if hasattr(item, attr):
                value = getattr(item, attr)
                if isinstance(value, list):
                    flattened.extend(value)
        # Only deduplicate if elements are hashable (strings)
        if all(isinstance(x, str) for x in flattened):
            return list(dict.fromkeys(flattened))
        return flattened
    
    def get_style_attr(items: List[FormattedBio], attr: str) -> List[str]:
        """Helper function to get style attributes."""
        flattened = []
        for item in items:
            if hasattr(item, 'style'):
                style = item.style
                if hasattr(style, attr):
                    value = getattr(style, attr)
                    if isinstance(value, list):
                        flattened.extend(value)
        return list(dict.fromkeys(flattened))

    def combine_message_examples(items: List[FormattedBio]) -> List[List[Message]]:
        """Helper function to combine message examples without deduplication."""
        combined = []
        for item in items:
            if hasattr(item, 'messageExamples'):
                value = item.messageExamples
                if isinstance(value, list):
                    combined.extend(value)
        return combined

    # Combine bios
    combined_bio = ' '.join([result.bio for result in results if result.bio])
    
    combined = FormattedBio(
        bio=combined_bio,
        lore=flat_map_and_dedupe(results, 'lore'),
        adjectives=flat_map_and_dedupe(results, 'adjectives'),
        topics=flat_map_and_dedupe(results, 'topics'),
        style=Style(
            all=get_style_attr(results, 'all'),
            chat=get_style_attr(results, 'chat'),
            post=get_style_attr(results, 'post')
        ),
        messageExamples=combine_message_examples(results),
        postExamples=flat_map_and_dedupe(results, 'postExamples')
    )
    
    return combined