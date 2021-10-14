import json
import urllib3.contrib.pyopenssl

from skills_ml.algorithms.skill_extractors import FuzzyMatchSkillExtractor
from skills_ml.ontologies.esco import Esco
from skills_ml.job_postings.common_schema import JobPostingCollectionSample
from skills_ml.storage import FSStore

print('building esco...')
ESCO = Esco(manual_build=True)
ESCO.print_summary_stats()
ESCO.name = 'esco_de'

storage = FSStore(path = 'resources/esco/')
ESCO.save(storage)