import logging
import os
from tqdm import tqdm
from pathlib import Path
from tweets_to_character.prepare_tweets import unzip_file, get_account_data, get_tweets, chunk_text
from tweets_to_character.schemas import InputSchema
from tweets_to_character.llm import extract_info_for_chunk, combine_and_deduplicate


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler())
logger.formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

BASE_OUTPUT_DIR = os.getenv('BASE_OUTPUT_DIR', './output')

async def run(inputs: InputSchema, worker_node_urls: list, *args, **kwargs):
    logger.info(f"Running with inputs: {inputs}")

    folder_id = inputs.folder_id

    # if tweeter_folder_id is .zip, unzip it
    if folder_id.endswith('.zip'):
        folder_id = unzip_file(folder_id, BASE_OUTPUT_DIR)
    else:
        folder_id = Path(BASE_OUTPUT_DIR) / folder_id
    
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


if __name__ == "__main__":
    import os
    import asyncio
    from dotenv import load_dotenv
    from tweets_to_character.schemas import InputSchema

    os.environ['BASE_OUTPUT_DIR'] = '/Users/arshath/play/tweets_to_character/output'
    
    if not os.environ.get('OPENAI_API_KEY'):
        raise ValueError('OPENAI_API_KEY is not set in the environment')
    
    if not os.path.exists(os.environ['BASE_OUTPUT_DIR']):
        os.makedirs(os.environ['BASE_OUTPUT_DIR'])

    load_dotenv()

    archive_path = '/Users/arshath/Downloads/twitter-2024-11-14-ebb9578b384ebab9a263b7621eb86794462f7f5fa47d51d2e33a7607ed0d8f70.zip'

    input_schema = InputSchema(folder_id=archive_path)
    worker_node_urls = []

    result = asyncio.run(run(inputs=input_schema, worker_node_urls=worker_node_urls))
    print(result)