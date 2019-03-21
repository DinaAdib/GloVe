from libraries import *
from corpus import *
pyximport.install()


FILEPATH = "/home/dina/.local/lib/python3.5/site-packages/gensim/models/text8"
shortStoryPath = "/home/dina/PycharmProjects/Glove/short_story.txt"
trialPath = "/home/dina/PycharmProjects/Glove/Trial_Sentences.txt"


def main():
    corpus = read_corpus(shortStoryPath)
    # vocab = get_vocab(corpus)
    # print(len(corpus), len(vocab))
    # print((corpus), (vocab))
    # save_obj(vocab, "shortVocabIdFreq")
    # cooccurrences = list(construct_coocurrence_matrix(vocab,corpus,windowSize,0))
    # save_obj(cooccurrences, "shortcooccurrences")
    vocab = load_obj("vocabIdFreq")
    cooccurrences = load_obj("coocurrenceMatrix")

    model = Glove(cooccurrences = cooccurrences, d=vectorDimension, alpha=alpha, x_max=xmax, vocabSize=len(vocab))

    for epoch in range(25):
        err = model.train()
        print("epoch %d, error %.3f" % (epoch, err), flush=True)

    # print(coocurrences)
    # print(len(vocab))

    # vocab, wordsToIds = load_data()

    # print(wordsToIds)

    # fout = open("tokens.txt", "w")
    # for token in vocab:
    #     fout.write(token)
    # fout.close()
def read_clean_data(filename):
    file = open(filename, 'rt')
    text = file.read()
    file.close()
    # split into words
    from nltk.tokenize import word_tokenize
    tokens = word_tokenize(text)
    # convert to lower case
    tokens = [w.lower() for w in tokens]
    # remove punctuation from each word
    import string
    table = str.maketrans('', '', string.punctuation)
    stripped = [w.translate(table) for w in tokens]
    # remove remaining tokens that are not alphanumeric
    words = [word for word in stripped if word.isalnum()]

if __name__ == "__main__":
    main()


