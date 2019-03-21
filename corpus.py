# Cooccurrence matrix construction tools
# for fitting the GloVe model.
from libraries import *

logger = logging.getLogger("glove")

def read_corpus(filepath):
  fin = open(filepath, "r")
  lines = fin.readlines()
  fin.close()
  return sent_tokenize(str(lines).lower())

def get_vocab(corpus):
    vocab = Counter()
    delchars = [chr(c) for c in range(256)]
    delchars = [x for x in delchars if not x.isalnum()]
    delchars.remove(' ')
    delchars = ''.join(delchars)
    transTable = str.maketrans(dict.fromkeys(delchars))

    for line in corpus:
        line = line.lower()#.translate(transTable)
        tokens = line.split()
        # tokens = nltk.word_tokenize(line)
        vocab.update(tokens)

    return {word: (i,freq) for i, (word, freq) in enumerate(vocab.items())}

def preprocess_lines(lines):
    lines_without_stopwords = []  # stop words contain the set of stop words for line in lines
    stop_words = set(stopwords.words("english"))

    for line in lines:
        words = line.split(" ")
        temp_line = []
        for word in words:
            if word not in stop_words:
                temp_line.append(word)
        string=" "
        lines_without_stopwords.append(string.join(temp_line))
    lines=lines_without_stopwords

    # print(lines)

    wordnet_lemmatizer = WordNetLemmatizer()
    lines_with_lemmas=[] #stop words contain the set of stop words for line in lines:

    for line in lines:
        words = line.split(" ")
        temp_line = []
        for word in words:
            temp_line.append(wordnet_lemmatizer.lemmatize(word))
        string=" "
        lines_with_lemmas.append(string.join(temp_line))

    lines= [line.split(" ") for line in lines_with_lemmas]
    return lines

def construct_coocurrence_matrix(vocab, corpus, windowSize, minCount):
    # m = np.zeros((len(vocab), len(vocab)))  # n is the count of all words
    #
    # for i, word in enumerate(words):
    #     for j in range(max(i - windowSize, 0), min(i + windowSize, len(vocab))):
    #         m[vocab[word], vocab[words[j]]] += 1
    vocabSize = len(vocab)
    cooccurrences = sparse.lil_matrix((vocabSize, vocabSize), dtype=np.float64)

    id2word = dict((i, word) for word, (i, _) in vocab.items())

    for i, line in enumerate(corpus):
        if i % 1000 == 0:
            logger.info("Building cooccurrence matrix: on line %i", i)

        tokens = line.strip().split()
        token_ids = [vocab[word][0] for word in tokens]

        for centerIndex, center_id in enumerate(token_ids):
            # Collect all word IDs in left window of center word
            context_ids = token_ids[max(0, centerIndex - windowSize): centerIndex]
            contexts_len = len(context_ids)
            for left_i, left_id in enumerate(context_ids):
                # Distance from center word
                distance = contexts_len - left_i

                # Weight by inverse of distance between words
                increment = 1.0 / float(distance)

                # Build co-occurrence matrix symmetrically (pretend we
                # are calculating right contexts as well)
                cooccurrences[center_id, left_id] += increment
                cooccurrences[left_id, center_id] += increment

    # Now yield our tuple sequence (dig into the LiL-matrix internals to
    # quickly iterate through all nonzero cells)

    for i, (row, data) in enumerate(zip(cooccurrences.rows,
                                                   cooccurrences.data)):
        if minCount is not None and vocab[id2word[i]][1] < minCount:
            continue

        for data_idx, j in enumerate(row):
            if minCount is not None and vocab[id2word[j]][1] < minCount:
                continue
            yield i, j, data[data_idx]
    # return m

def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

def load_data():
    vocab = load_obj("vocab")
    wordsToIds = load_obj("wordsToIds")
    return vocab, wordsToIds

def save_coocurrences(vocab, corpus):
    cooccurrences = list(construct_coocurrence_matrix(vocab, corpus, windowSize,10))
    logger.info("Cooccurrence list fetch complete (%i pairs).\n",
                len(cooccurrences))
    print(len(cooccurrences))
    save_obj(cooccurrences, "coocurrenceMatrix")