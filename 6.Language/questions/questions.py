import nltk
import sys
import os
import string
import math

FILE_MATCHES = 1
SENTENCE_MATCHES = 1


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python questions.py corpus")

    # Calculate IDF values across files
    files = load_files(sys.argv[1])
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = compute_idfs(file_words)

    # Prompt user for query
    query = set(tokenize(input("Query: ")))

    # Determine top file matches according to TF-IDF
    filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)

    # Extract sentences from top files
    sentences = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens

    # Compute IDF values across sentences
    idfs = compute_idfs(sentences)

    # Determine top sentence matches
    matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
    for match in matches:
        print(match)


def load_files(directory):
    """
    Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """
    files = dict()
    for file in os.listdir(directory):
        with open(os.path.join(directory, file), "r", encoding = "utf-8") as f:
            files[file] = f.read()

    return files

def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """
    punctuation = string.punctuation
    stop_words = nltk.corpus.stopwords.words("english")

    tokens = nltk.word_tokenize(document.lower())
    words = [word for word in tokens if word not in punctuation and word not in stop_words]
    
    return words


def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """
    words_idfs = dict()
    num_docs = len(documents)

    unique_words = set(word for doc in documents.values() for word in doc)

    for word in unique_words:
        count = 0
        for doc in documents:
            if word in documents[doc]:
                count += 1
        
        words_idfs[word] = math.log(num_docs / count)

    return words_idfs


def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """
    file_scores = dict()
    
    for file, words in files.items():
        tf_idf = 0
        for word in query:
            tf_idf += words.count(word) * idfs[word]
        file_scores[file] = tf_idf
    
    top_files = sorted(file_scores.items(), key=lambda x: x[1], reverse=True)
    top_files = [x[0] for x in top_files]

    return top_files[:n]


def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """
    sentence_scores = dict()

    for s, words in sentences.items():
        words_in_query = query.intersection(words)

        idf = 0
        for word in words_in_query:
            idf += idfs[word]
        
        num_words_in_query = sum(map(lambda x: x in words_in_query, words))
        qtd = num_words_in_query / len(words)

        sentence_scores[s] = {"idf": idf, "qtd": qtd}
    
    top_sentences = sorted(sentence_scores.items(), key=lambda x: (x[1]["idf"], x[1]["qtd"]), reverse=True)
    top_sentences = [x[0] for x in top_sentences]

    return top_sentences[:n]

if __name__ == "__main__":
    main()
