from convokit import Corpus, download

# List available datasets in ConvoKit
from convokit import corpora
for corpus_name in corpora.list():
    print(corpus_name)
