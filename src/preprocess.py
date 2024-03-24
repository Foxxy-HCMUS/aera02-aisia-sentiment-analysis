import pandas as pd

import spacy
from spacy_cld import LanguageDetector

from pandarallel import pandarallel

pandarallel.initialize(progress_bar=True)