import logging

"""
I'm not really sure how this all is supposed to work.
"""


logger = logging.getLogger('semlm')
logger.setLevel(logging.INFO)

# Create console handler and set level to debug
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
# Create formatter
formatter = logging.Formatter('%(levelname)s: %(message)s')
# Add formatter to console handler
ch.setFormatter(formatter)
# Add console handler to logger
logger.addHandler(ch)
