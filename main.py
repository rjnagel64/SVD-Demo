

from collections import Counter
import os
import os.path

#  import numpy as np
#  from scipy import linalg

import requests


TEXTS_DIR = "texts"


def main():
    if not os.path.exists(TEXTS_DIR):
        os.mkdir(TEXTS_DIR)
        download_files()
    
    pass

def download_files():
    """Download a selection of books and other texts from Project Gutenberg."""
    with open("sources.txt", "r") as f:
        sources = f.read()

    # Take only the nonempty, noncomment lines of the file.
    lines = list(filter(lambda l: l != "" and l[0] != "#", map(str.strip, sources.splitlines())))
    # This turns ['a', 'b', 'c', 'd', ...] into [('a', 'b'), ('c', 'd'), ...].
    files = list(zip(lines[::2], lines[1::2]))

    for (url, local_name) in files:
        print(f"Downloading {local_name} from {url}...")
        r = requests.get(url)
        if r.status_code != requests.codes.ok:
            print(f"\tError: {r.status_code} {r.reason}")
            # Continue trying to download the other files.
            continue

        destination_file = os.path.join(TEXTS_DIR, local_name)
        with open(destination_file, "w") as f:
            f.write(r.text)

def scrub_inputs():
    """Remove the Project Gutenberg front matter from our inputs.
    
    Additionally, works like Marlowe's Doctor Faustus have footnotes. Those need
    to be removed, as well as the footer they reference. (Both [73] and [Footnote
    73]: ...

    Hamlet: Ok.
    """
    pass

def build_matrix():
    """Build the term-document matrix.

    returns: a 3-tuple `(mat, words, docs)` where:
    * `mat` is the term-document matrix, as a numpy 'ndarray'.
    * `words` is a list such that `words[i]` corresponds to row `i` of `mat`.
    * `docs` is a list such that `docs[j]` corresponds to column `j` of `mat`. 
    """
    counters = {}
    for document in processed_texts:
        counters[document.name] = Counter(words(document))

if __name__ == "__main__":
    main()
