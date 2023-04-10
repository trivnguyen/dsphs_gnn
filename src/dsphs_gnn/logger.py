
import logging
import sys

# create logger
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

# format output and add stream handler
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setLevel(logging.DEBUG)
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)
