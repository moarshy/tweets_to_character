import logging
import os
from tqdm import tqdm
from pathlib import Path
from tweets_to_character.prepare_tweets import unzip_file, get_account_data, get_tweets, chunk_text
from tweets_to_character.schemas import InputSchema
from tweets_to_character.llm import extract_info_for_chunk, combine_and_deduplicate
from naptha_sdk.schemas import AgentRunInput


logger = logging.getLogger(__name__)

async def run(agent_run: AgentRunInput, *args, **kwargs):
    logger.info(f"Running with inputs: {agent_run.inputs}")

    folder_id = Path(agent_run.inputs.input_dir)

    # get account data
    account_data = get_account_data(folder_id)
    logger.info(f"Account data: {account_data}")

    # get tweets
    tweets = get_tweets(folder_id)
    logger.info(f"Tweets: {len(tweets)}")

    # chunk tweets
    chunks = await chunk_text(tweets, account_data, folder_id)
    logger.info(f"Chunks: {len(chunks)}")

    # run llm
    results = []

    for i, chunk in tqdm(enumerate(chunks), total=len(chunks)):
        result = await extract_info_for_chunk(
            account_data=account_data,
            chunk=chunk,
            chunk_index=i,
            model='openai'
        )
        results.append(result)

    # combine results
    combined = combine_and_deduplicate(results)


    return combined.model_dump()