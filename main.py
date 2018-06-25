

from collections import Counter
import os
import os.path
import re

import numpy as np
from scipy import linalg
from matplotlib import rcParams
import matplotlib.pyplot as plt

import requests


TEXTS_DIR = "texts"
CLEAN_TEXTS_DIR = "clean-texts"

SEPARATORS_PAT = re.compile(r'(\[\d+\])|[:?!\(\),."“”;_]')

def main():
    if not os.path.exists(TEXTS_DIR):
        os.mkdir(TEXTS_DIR)
        file_info = download_files()
        scrub_inputs(file_info)
    
    (mat, words, docs) = build_matrix2()
    print(len(words))
    print(len(docs))
    print(mat)
    print(mat.shape)

    # Rank-2 approx:
    (W, ss, Vh) = linalg.svd(mat)
    (rows, _) = W.shape
    (_, cols) = Vh.shape

    S = np.zeros((rows, cols))
    for (i, s) in enumerate(ss):
        S[i, i] = s

    #  print(W @ S @ Vh)

    new_basis = Vh[0:2]
    print(new_basis)

    # TODO: Create a better image, get better clusters, add better labels.
    # TODO: Label points of note.

    vh1_label = "V1 = <{0}>".format(Vh[0:1])
    vh2_label = "V2 = <{0}>".format(Vh[1:2])


    words_blacklist = [
        "feel",
        "feelings",
        "A",
        "What",
        "passed",
        #  "",
        #  "",
        #  "",
        #  "",
    ]

    words_whitelist = [
        "Hamlet",
        "OPHELIA",
        "BARNARDO",
        "MARCELLUS",
        "ROSENCRANTZ",
        "HORATIO",
        "Horatio", # I should really fold cases...
        "LAERTES",
        "Polonius",
        "GHOST",
        "Elsinore",

        "FAUSTUS",
        "CORNELIUS",
        "MEPHISTOPHILIS",
        "MEPHIST",
    ]

    xs = []
    ys = []
    annotations = []
    num_words = mat.shape[0]
    for i in range(num_words):
        point = np.transpose(mat[i:i+1])
        (x, y) = new_basis @ point

        if x < -100 or y > 150:
            continue
        else:
            xs.append(x)
            ys.append(y)

        if x < -40 or y > 20:
            if words[i] in words_whitelist or abs(y/x) > 2:
                annotations.append((x, y, words[i]))
                #  print(f"Word: {words[i]} at {x}, {y}")
            #  if words[i] not in words_blacklist:
                #  annotations.append((x, y, words[i]))

    (fig, ax) = plt.subplots()
    ax.set_xlabel(vh1_label)
    ax.set_ylabel(vh2_label)

    ax.scatter(
            xs, # X-Coordinates
            ys, # Y-Coordinates
            s=2, # Marker size (default is 6)
    )

    for (x, y, text) in annotations:
        ax.annotate(xy=(x, y), s=text)

    plt.savefig("test.pdf", format="pdf")

    # V1 and V2 form a basis for the space.
    #  V1 = Vh[]
    #  V2 = Vh[]

def download_files():
    """Download a selection of books and other texts from Project Gutenberg.
    
    returns:
    a list of tuples `(url, filename, range)` parsed from 'sources.txt'.
    """
    with open("sources.txt", "r") as f:
        sources = f.read()

    # Take only the nonempty, noncomment lines of the file.
    lines = list(filter(lambda l: l != "" and l[0] != "#", map(str.strip, sources.splitlines())))
    files = list(zip(lines[::3], lines[1::3], lines[2::3]))

    for (url, local_name, _) in files:
        print(f"Downloading {local_name} from {url}...")
        r = requests.get(url)
        if r.status_code != requests.codes.ok:
            print(f"\tError: {r.status_code} {r.reason}")
            # Continue trying to download the other files.
            continue

        destination_file = os.path.join(TEXTS_DIR, local_name)
        with open(destination_file, "w") as f:
            f.write(r.text)

    return files

def scrub_inputs(file_info):
    """Clean up the sources so that they are suitable for processing.

    This involves deleting front/end matter, and replacing punctutation and
    footnotes with spaces.
    The cleaned copies are placed in `CLEAN_TEXTS_DIR`.
    
    Faustus: delete (1..25) (2152..) (Footnotes of the form `\[\d+\]`
    Hamlet: delete (1..30) (4710..)
    Frankenstein: delete (1..29) (7457..)

    Punctuation to remove/treat as word separators: _",.()!?;: U+201C U+201D
    Valid word characters: U+2019 U+2014

    arguments:
    - file_info: a list of 3-tuples `(url, filename, range)`. `range` is a string
        consisting of two whitespace-separated numbers. It specifies the portion of
        the file between the front and end matter.
    """
    if not os.path.exists(CLEAN_TEXTS_DIR):
        os.mkdir(CLEAN_TEXTS_DIR)

    
    for (_, filename, ran) in file_info:
        print(f"Cleaning {filename}...")
        (front_end, end_start) = ran.split()
        front_end = int(front_end)
        end_start = int(end_start)
        with open(os.path.join(TEXTS_DIR, filename), "r") as orig:
            with open(os.path.join(CLEAN_TEXTS_DIR, filename), "w") as new:
                for (i, line) in enumerate(orig):
                    # Ignore front matter.
                    if i < front_end:
                        continue

                    # Ignore end matter.
                    if i > end_start:
                        continue

                    # Strip out footnotes and other separators.
                    line = SEPARATORS_PAT.sub(" ", line)

                    new.write(line)

# TODO: Consider using a sparse matrix for this? Nah. Not important enough yet.
# TODO: Look into normalization, and ways of dealing with words like 'the',
#       'and', 'I', 'of', 'to', etc., etc. (Zipf's law? Divide count by rank in
#       frequency table?)
def build_matrix():
    """Build the term-document matrix.

    returns: a 3-tuple `(mat, words, docs)` where:
    * `mat` is the term-document matrix, as a numpy 'ndarray'.
    * `words` is a list such that `words[i]` corresponds to row `i` of `mat`.
    * `docs` is a list such that `docs[j]` corresponds to column `j` of `mat`. 
    """
    counters = {}
    for filename in os.listdir(CLEAN_TEXTS_DIR):
        with open(os.path.join(CLEAN_TEXTS_DIR, filename), "r") as f:
            counters[filename] = Counter(iter_words(f))
            print(counters[filename].most_common(10))

    all_words = set()
    for (filename, counter) in counters.items():
        for word in counter:
            all_words.add(word)

    words = list(all_words)
    words.sort()

    docs = list(counters.keys())
    docs.sort()

    mat = np.zeros((len(words), len(docs)))

    for (filename, counter) in counters.items():
        column = docs.index(filename)
        for (word, count) in counter.items():
            row = words.index(word)
            mat[row, column] = count

    return (mat, words, docs)

def build_matrix2():
    # `counters` associates each filename with a `Counter` of all the proteins
    # in that file.
    counters = {}
    for filename in os.listdir(CLEAN_TEXTS_DIR):
        with open(os.path.join(CLEAN_TEXTS_DIR, filename), "r") as f:
            counters[filename] = Counter(iter_words(f))
            print(counters[filename].most_common(10))

    # Now, get a set of all the proteins (from all the files)
    all_words = set()
    for (filename, counter) in counters.items():
        for word in counter.keys():
            all_words.add(word)

    words = []
    documents = list(counters.keys())
    mat = np.zeros((len(all_words), len(counters.keys())))

    # Put the documents on the outside loop so we have a few long iterations
    # instead of many short iterations
    for (j, (document, counter)) in enumerate(counters.items()):
        for (i, word) in enumerate(all_words):
            # If a key is not found in a counter, it defaults to zero.
            mat[i, j] = counter[word]

            # Only build the list of words once.
            if j == 0:
                words.append(word)

    return (mat, words, documents)

    

def iter_words(f):
    """Iterate over the words of a file.
    
    arguments:
    - f: An iterator yielding lines of text.

    returns:
    An iterator over each word in `f`.
    """
    for line in f:
        for word in line.split():
            yield word

if __name__ == "__main__":
    print(rcParams["lines.markersize"])
    main()
