from unidecode import unidecode
import re


def pre_process_string(value):
    """
    Do a little bit of data cleaning with the help of Unidecode and Regex.
    Things like casing, extra spaces, quotes and new lines can be ignored.
    """

    if type(value) in (float, int):
        value = ""

    value = unidecode(value)
    value = re.sub('\n', ' ', value)
    value = re.sub('-', '', value)
    value = re.sub('/', ' ', value)
    value = re.sub("'", '', value)
    value = re.sub(",", '', value)
    value = re.sub(":", ' ', value)
    value = re.sub('  +', ' ', value)
    value = value.strip().strip('"').strip("'").lower().strip()
    if not value:
        value = None
    return value
