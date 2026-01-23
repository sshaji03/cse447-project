import pandas as pd

N_GRAM_FOLDER = "n_gram_counts/"
BIGRAM_DEV = N_GRAM_FOLDER + "bigrams_dev.csv"
BIGRAM_TRAIN = N_GRAM_FOLDER + "bigrams_train.csv"
TRIGRAM_DEV = N_GRAM_FOLDER + "trigrams_dev.csv"
TRIGRAM_TRAIN = N_GRAM_FOLDER + "trigrams_train.csv"
UNIGRAM_DEV = N_GRAM_FOLDER + "unigrams_dev.csv"
UNIGRAM_TRAIN = N_GRAM_FOLDER + "unigrams_train.csv"

class NgramConstructor:
    """
    This constructor creates hash tables out of the n grams for train and dev
    """

    def __init__(self):
        self.unigram_train = dict()
        self.unigram_dev = dict()
        self.unigram_total = 0
        self.bigrams_train = dict()
        self.bigrams_dev = dict()
        self.trigrams_train = dict()
        self.trigrams_dev = dict()

        # construct a unigram hashtable
        unigram_df_train = pd.read_csv(UNIGRAM_TRAIN)
        unigram_df_dev = pd.read_csv(UNIGRAM_DEV)
        bigram_df_train = pd.read_csv(BIGRAM_TRAIN)
        bigram_df_dev = pd.read_csv(BIGRAM_DEV)
        trigram_df_train = pd.read_csv(TRIGRAM_TRAIN)
        trigram_df_dev = pd.read_csv(TRIGRAM_DEV)

        for row in unigram_df_train.itertuples(index=False):
            if row[0] in self.unigram_train.keys():
                self.unigram_train[row[0]] += row[2]
            else:
                self.unigram_train[row[0]] = row[2]

        for row in unigram_df_dev.itertuples(index=False):
            if row[0] in self.unigram_dev.keys():
                self.unigram_dev[row[0]] += row[2]
            else:
                self.unigram_dev[row[0]] = row[2]

        for row in bigram_df_train.itertuples(index=False):
            if row[0] in self.bigrams_train.keys():
                self.bigrams_train[row[0]] += row[2]
            else:
                self.bigrams_train[row[0]] = row[2]

        for row in bigram_df_dev.itertuples(index=False):
            if row[0] in self.bigrams_dev.keys():
                self.bigrams_dev[row[0]] += row[2]
            else:
                self.bigrams_dev[row[0]] = row[2]

        for row in trigram_df_train.itertuples(index=False):
            if row[0] in self.trigrams_train.keys():
                self.trigrams_train[row[0]] += row[2]
            else:
                self.trigrams_train[row[0]] = row[2]

        for row in trigram_df_dev.itertuples(index=False):
            if row[0] in self.trigrams_dev.keys():
                self.trigrams_dev[row[0]] += row[2]
            else:
                self.trigrams_dev[row[0]] = row[2]

    def get_unigram_train(self):
        return self.unigram_train

    def get_unigram_dev(self):
        return self.unigram_dev

    def get_bigram_train(self):
        return self.bigrams_train

    def get_bigram_dev(self):
        return self.bigrams_dev

    def get_trigram_train(self):
        return self.trigrams_train

    def get_trigram_dev(self):
        return self.trigrams_dev

if __name__ == '__main__':
    ngram = NgramConstructor()
    print(ngram.get_unigram_train())
