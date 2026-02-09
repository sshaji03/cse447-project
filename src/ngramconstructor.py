import pandas as pd
import math
import os

N_GRAM_FOLDER = "work/hashtable_counts/"
FIVEGRAM_DEV = N_GRAM_FOLDER + "5grams_dev.csv"
FIVEGRAM_TRAIN = N_GRAM_FOLDER + "/split_5grams_train/"
TRIGRAM_DEV = N_GRAM_FOLDER + "trigrams_dev.csv"
TRIGRAM_TRAIN = N_GRAM_FOLDER + "trigrams_train.csv"
UNIGRAM_DEV = N_GRAM_FOLDER + "unigrams_dev.csv"
UNIGRAM_TRAIN = N_GRAM_FOLDER + "unigrams_train.csv"
UNK_CHAR = "\uE000"

class NgramConstructor:
    """
    This constructor creates hash tables out of the n grams for train and dev
    """

    def __init__(self):
        self.unigram_train = dict()
        self.unigram_dev = dict()
        self.unigram_total = 0
        self.fivegrams_train = dict()
        self.fivegrams_dev = dict()
        self.trigrams_train = dict()
        self.trigrams_dev = dict()

        self.vocab = set()

        # counts how many individual tokens there are
        self.unigram_token_total_train = 0
        self.unigram_token_total_dev = 0
        # counts how many bigram tokens start with a specific character
        self.fivegrams_token_counts_train = dict()
        self.fivegrams_token_counts_dev = dict()
        # counts how many trigram tokens start with a specific character
        self.trigram_token_counts_train = dict()
        self.trigram_token_counts_dev = dict()

        self.unigram_prob_train = dict()
        self.unigram_prob_dev = dict()
        self.fivegram_prob_train = dict()
        self.fivegram_prob_dev = dict()
        self.trigram_prob_train = dict()
        self.trigram_prob_dev = dict()

        unigram_df_train = pd.read_csv(UNIGRAM_TRAIN, dtype={'char_ngram': str})
        unigram_df_dev = pd.read_csv(UNIGRAM_DEV, dtype={'char_ngram': str})
        fivegram_df_dev = pd.read_csv(FIVEGRAM_DEV, dtype={'char_ngram': str})
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

        files = os.listdir(FIVEGRAM_TRAIN)
        for file in files:
            path = os.path.join(FIVEGRAM_TRAIN, file)
            csv = pd.read_csv(path, dtype={'char_ngram': str})
            for _, row in csv.iterrows():
                fivegram = str(row['char_ngram'])
                if len(fivegram) < 5:
                    continue
                count = row['count']
                self.fivegrams_train[fivegram] = self.fivegrams_train.get(fivegram, 0) + count

                c1, c2, c3, c4, c5 = fivegram
                context = c1 + c2 + c3 + c4
                self.fivegrams_token_counts_train[context] = self.fivegrams_token_counts_train.get(context, 0) + count

        for _, row in fivegram_df_dev.iterrows():
            fivegram = str(row['char_ngram'])
            if len(fivegram) < 5:
                continue

            count = row['count']
            self.fivegrams_dev[fivegram] = self.fivegrams_dev.get(fivegram, 0) + count

            c1, c2, c3, c4, c5 = fivegram
            context = c1 + c2 + c3 + c4
            self.fivegrams_token_counts_dev[context] = self.fivegrams_token_counts_dev.get(context, 0) + count

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

        for fivegram in self.fivegrams_train.keys():
            c1, c2, c3, c4, _ = fivegram
            context = c1 + c2 + c3 + c4
            self.fivegram_prob_train[fivegram] = self.fivegrams_train[fivegram] / self.fivegrams_token_counts_train[context]

        for fivegram in self.fivegrams_dev.keys():
            c1, c2, c3, c4, _ = fivegram
            context = c1 + c2 + c3 + c4
            self.fivegram_prob_dev[fivegram] = self.fivegrams_dev[fivegram] / self.fivegrams_token_counts_dev[context]

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

    def get_fivegram_train(self):
        return self.fivegrams_train

    def get_fivegram_dev(self):
        return self.fivegrams_dev

    def get_trigram_train(self):
        return self.trigrams_train

    def get_trigram_dev(self):
        return self.trigrams_dev

    def get_vocab(self):
        return self.vocab

    def calculate_train_perplexity(self, lambda1, lambda2, lambda3):
        log_prob_sum = 0.0

        for fivegram, count in self.fivegrams_train.items():
            c1, c2, c3, c4, unigram = fivegram
            trigram = c3 + c4 + unigram
            unigram = unigram if unigram in self.vocab else UNK_CHAR

            uni = lambda1 * self.unigram_prob_train[unigram]
            tri = lambda2 * self.trigram_prob_train[trigram]
            five = lambda3 * self.fivegram_prob_train[fivegram]

            log_prob_sum += count * math.log(uni + tri + five)

        n = sum(self.fivegrams_train.values())

        return math.exp(-log_prob_sum / n)

    def calculate_dev_perplexity(self, lambda1, lambda2, lambda3):
        log_prob_sum = 0.0

        for fivegram, count in self.fivegrams_dev.items():
            c1, c2, c3, c4, unigram = fivegram
            trigram = c3 + c4 + unigram
            unigram = unigram if unigram in self.vocab else UNK_CHAR

            uni = lambda1 * self.unigram_prob_dev[unigram]
            tri = lambda2 * self.trigram_prob_dev[trigram]
            five = lambda3 * self.fivegram_prob_dev[fivegram]

            log_prob_sum += count * math.log(uni + tri + five)

        n = sum(self.fivegrams_dev.values())

        return math.exp(-log_prob_sum / n)


if __name__ == '__main__':
    ngram = NgramConstructor()
    print(ngram.fivegrams_train)
