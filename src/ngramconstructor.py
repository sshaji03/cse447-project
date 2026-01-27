import pandas as pd
import math

N_GRAM_FOLDER = "work/hashtable_counts/"
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

        self.vocab = set()

        # counts how many individual tokens there are
        self.unigram_token_total_train = 0
        self.unigram_token_total_dev = 0
        # counts how many bigram tokens start with a specific character
        self.bigrams_token_counts_train = dict()
        self.bigrams_token_counts_dev = dict()
        # counts how many trigram tokens start with a specific character
        self.trigram_token_counts_train = dict()
        self.trigram_token_counts_dev = dict()

        self.unigram_prob_train = dict()
        self.unigram_prob_dev = dict()
        self.bigram_prob_train = dict()
        self.bigram_prob_dev = dict()
        self.trigram_prob_train = dict()
        self.trigram_prob_dev = dict()

        unigram_df_train = pd.read_csv(UNIGRAM_TRAIN, dtype={'char_ngram': str})
        unigram_df_dev = pd.read_csv(UNIGRAM_DEV, dtype={'char_ngram': str})
        bigram_df_train = pd.read_csv(BIGRAM_TRAIN, dtype={'char_ngram': str})
        bigram_df_dev = pd.read_csv(BIGRAM_DEV, dtype={'char_ngram': str})
        trigram_df_train = pd.read_csv(TRIGRAM_TRAIN, dtype={'char_ngram': str})
        trigram_df_dev = pd.read_csv(TRIGRAM_DEV, dtype={'char_ngram': str})

        for _, row in unigram_df_train.iterrows():
            unigram = str(row['char_ngram'])
            count = row['count']
            self.unigram_token_total_train += count
            if unigram in self.unigram_train.keys():
                self.unigram_train[unigram] += count
            else:
                self.vocab.add(unigram)
                self.unigram_train[unigram] = count

        for  _, row in unigram_df_dev.iterrows():
            unigram = str(row['char_ngram'])
            count = row['count']
            self.unigram_token_total_dev += count
            if unigram in self.unigram_dev.keys():
                self.unigram_dev[unigram] += count
            else:
                self.unigram_dev[unigram] = count

        for _, row in bigram_df_train.iterrows():
            bigram = str(row['char_ngram'])
            count = row['count']
            self.bigrams_train[bigram] = self.bigrams_train.get(bigram, 0) + count

            c1, c2 = bigram
            self.bigrams_token_counts_train[c1] = self.bigrams_token_counts_train.get(c1, 0) + count

        for _, row in bigram_df_dev.iterrows():
            bigram = str(row['char_ngram'])
            count = row['count']
            self.bigrams_dev[bigram] = self.bigrams_dev.get(bigram, 0) + count

            c1, c2 = bigram
            self.bigrams_token_counts_dev[c1] = self.bigrams_token_counts_dev.get(c1, 0) + count

        for _, row in trigram_df_train.iterrows():
            trigram = str(row['char_ngram'])
            count = row['count']

            # one special case that isn't being counted as 3 because of how computer is interpreting
            # frequency of (1) so okay to skip
            if (len(trigram) < 3):
                continue

            self.trigrams_train[trigram] = self.trigrams_train.get(trigram, 0) + count
            c1, c2, c3 = trigram

            context = c1 + c2
            self.trigram_token_counts_train[context] = self.trigram_token_counts_train.get(context, 0) + count

        for _, row in trigram_df_dev.iterrows():
            trigram = str(row['char_ngram'])
            count = row['count']

            # one special case that isn't being counted as 3 because of how computer is interpreting
            # frequency of (1) so okay to skip
            if (len(trigram) < 3):
                continue

            self.trigrams_dev[trigram] = self.trigrams_dev.get(trigram, 0) + count

            c1, c2, c3 = trigram
            context = c1 + c2
            self.trigram_token_counts_dev[context] = self.trigram_token_counts_dev.get(context, 0) + count

        for unigram in self.unigram_train.keys():
            self.unigram_prob_train[unigram] = self.unigram_train[unigram] / self.unigram_token_total_train

        for unigram in self.unigram_dev.keys():
            self.unigram_prob_dev[unigram] = self.unigram_dev[unigram] / self.unigram_token_total_dev

        for bigram in self.bigrams_train.keys():
            c1, _ = bigram
            self.bigram_prob_train[bigram] = self.bigrams_train[bigram] / self.bigrams_token_counts_train[c1]

        for bigram in self.bigrams_dev.keys():
            c1, _ = bigram
            self.bigram_prob_dev[bigram] = self.bigrams_dev[bigram] / self.bigrams_token_counts_dev[c1]

        for trigram in self.trigrams_train.keys():
            c1, c2, _ = trigram
            context = c1 + c2
            self.trigram_prob_train[trigram] = self.trigrams_train[trigram] / self.trigram_token_counts_train[context]

        for trigram in self.trigrams_dev.keys():
            c1, c2, _ = trigram
            context = c1 + c2
            self.trigram_prob_dev[trigram] = self.trigrams_dev[trigram] / self.trigram_token_counts_dev[context]

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

    def get_vocab(self):
        return self.vocab

    def calculate_train_perplexity(self, lambda1, lambda2, lambda3):
        log_prob_sum = 0.0

        for trigram, count in self.trigrams_train.items():
            unigram, c2, c3 = trigram
            bigram = unigram+c2

            uni = lambda1 * self.unigram_prob_train[unigram]
            bi = lambda2 * self.bigram_prob_train[bigram]
            tri = lambda3 * self.trigram_prob_train[trigram]

            log_prob_sum += count * math.log(uni + bi + tri)

        n = sum(self.trigrams_train.values())

        return math.exp(-log_prob_sum / n)

    def calculate_dev_perplexity(self, lambda1, lambda2, lambda3):
        log_prob_sum = 0.0

        for trigram, count in self.trigrams_dev.items():
            unigram, c2, c3 = trigram
            bigram = unigram+c2

            uni = lambda1 * self.unigram_prob_dev[unigram]
            bi = lambda2 * self.bigram_prob_dev[bigram]
            tri = lambda3 * self.trigram_prob_dev[trigram]

            log_prob_sum += count * math.log((uni + bi + tri))

        n = sum(self.trigrams_dev.values())

        return math.exp(-log_prob_sum / n)


if __name__ == '__main__':
    ngram = NgramConstructor()
    print(ngram.get_unigram_train())
    print(ngram.get_vocab())
    print(ngram.get_unigram_train_prob())
