from sets import Set
import math
import itertools
import json
from os import walk
import os

class LM:
    """
    ngram language model
    """

    #max_n_gram_size.
    max_n_gram_size = 7
    train_path = "./650_a3_train/"
    save_path = "./probs/"


    def robust_decode(self, bs):
        """
        Takes a byte string as param and convert it into a unicode one.
        First tries UTF8, and fallback to Latin1 if it fails
        """
        cr = None
        try:
            cr = bs.decode('utf8')
        except UnicodeDecodeError:
            cr = bs.decode('latin1')
        return cr

    def read_and_count(self):
        """
        This function counts all ngrams up to max_n_gram_size and adds them in two dictionaries.
        For example, ngram_list_dictionary has ngram 'abc' whereas ngram_list_dictionary_tuple has the tuple ('c', 'ab') which represents P('c'|'ab').
        """
        self.ngram_list_dictionary = {}
        self.ngram_list_dictionary_tuple = {}
        ngram = 1

        while (ngram <= self.max_n_gram_size):
            fd = open(self.train_path+self.fileName,"r")
            number_of_words = 0  # to capture number of words in the file.
            lines = fd.readlines()
            lines = [line.strip() for line in lines]
            for line in lines:
                words = line.split()
                number_of_words += len(words)
                for word in words:
                    word = self.robust_decode(word)
                    padded_word = self.pad_word(word, ngram)
                    padded_word = list(padded_word)
                    for i,_ in enumerate(padded_word):
                        if(i>=ngram - 1):
                            post = padded_word[i]
                            prior = padded_word[i - (ngram - 1) : i]
                            temp = ''.join(prior) + post
                            tuple_temp = (post, ''.join(prior))
                            if temp in self.ngram_list_dictionary:
                                self.ngram_list_dictionary[temp] += 1
                                self.ngram_list_dictionary_tuple[tuple_temp] += 1
                            else:
                                self.ngram_list_dictionary[temp] = 1
                                self.ngram_list_dictionary_tuple[tuple_temp] = 1
            ngram += 1

        # add count of starter symbols.
        ngram = 1
        while(ngram <= self.max_n_gram_size - 1):
            temp_list = ['<'] * ngram
            self.ngram_list_dictionary[''.join(temp_list)] = number_of_words
            ngram += 1


        # find the total number of all unigram characters.
        self.unigram_total = 0.0
        self.alphabet_set = Set()
        for (post, prior) in self.ngram_list_dictionary_tuple.keys():
            if prior == '':
                self.alphabet_set.add(post)
                count_of_prior_and_post = float(self.ngram_list_dictionary[prior + post])
                self.unigram_total += count_of_prior_and_post

        return

    def calculate_probabilities(self):
        """
        This function computes unsmoothed probabilities for ngrams.
        ngram_probability has these probabilities.
        """
        self.ngram_probability = {}
        for (post, prior) in self.ngram_list_dictionary_tuple.keys():
            count_of_prior_and_post = float(self.ngram_list_dictionary[prior + post])
            if prior in self.ngram_list_dictionary:
                count_of_prior = float(self.ngram_list_dictionary[prior])
                self.ngram_probability[prior+post] = float( (count_of_prior_and_post/count_of_prior) )
            else:
                if prior == '':
                    self.ngram_probability[prior+post] = float( count_of_prior_and_post/self.unigram_total )

        return

    def count_all_ngrams(self, ngram):
        """
        Count all possible ngrams that can be generated from alphabet_set with respect to ngram size.
        """

        #The is the number of all possible ngrams of size ngram from alphabet considering start and end symbols.
        set_size = len(list(self.alphabet_set))
        num1 = math.pow(set_size, ngram)
        num2 = math.pow(set_size, ngram-1)
        num3 = (float(num1-1)/float(set_size-1)) - 1
        return num1 + num2 +num3

    def count_unseen_ngrams(self):
        """
        Counts the ngrams that have not been seen in the training data and adds that count for each ngram size in self.N0 dictionary.
        """
        self.N0= {}
        ngram = 2
        while(ngram <= self.max_n_gram_size):
            num_all_ngrams = self.count_all_ngrams(ngram)
            num_seen_ngrams = 0
            for each in self.ngram_list_dictionary.keys():
                if(len(each)==ngram):
                    num_seen_ngrams += 1

            self.N0[ngram] = num_all_ngrams - num_seen_ngrams
            ngram += 1

        return

    def count_seen_ngrams(self, r, ngram):
        "Count seen ngrams of size ngram which appeares r times in the corpus"
        total = 0
        for each in self.ngram_list_dictionary.keys():
            if (len(each)==ngram) and (self.ngram_list_dictionary[each]==r):
                total += 1

        return total

    def good_turing_smoothing(self):
        """
        Good-Turing Smoothing, smoothed probabilities of each seen ngram is in self.smoothed_probabilities, and probabilities for unseen ngrams are in unseen_probabilities.
        """
        ngram = 2
        self.smoothed_probabilities = {}
        self.unseen_probabilities = {}

        for each in self.alphabet_set:
            if (len(each)==1):
                self.smoothed_probabilities[each] = self.ngram_probability[each]

        while (ngram <= self.max_n_gram_size):

            max_r = 0
            for each in self.ngram_list_dictionary.keys():
                if(len(each)==ngram):
                    if(self.ngram_list_dictionary[each] > max_r):
                        max_r = self.ngram_list_dictionary[each]

            probs = []
            Ns = []
            for r in range(max_r + 1):
                if(r==0):
                    Ns.append(self.N0[ngram])
                else:
                    Ns.append(self.count_seen_ngrams(r, ngram))

            total_N = 0
            for r in range(max_r + 1):
                total_N += Ns[r]

            for r in range(max_r):
                probs.append( float( ( float(r+1) * (float(Ns[r+1])+0.01) ) / ( float(total_N) * (float(Ns[r])+0.01) ) ) )

            probs.append( float( float(max_r) / float(total_N) ) )

            sum_probs = 0.0
            for each in probs:
                sum_probs += each

            for r in range(max_r + 1):
                probs[r] = float(probs[r]) / float(sum_probs)

            for each , count in self.ngram_list_dictionary.iteritems():
                if (len(each)==ngram):
                    self.smoothed_probabilities[each] = probs[count]

            self.unseen_probabilities[ngram] = probs[0]

            ngram += 1

        return

    def save_probabilities(self):
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        with open(self.save_path+self.fileName[:-8]+'.no.smoothing.json', 'w') as fp:
            json.dump(self.ngram_probability,fp)

        with open(self.save_path+self.fileName[:-8]+'.good.turing.seen.json', 'w') as fp:
            json.dump(self.smoothed_probabilities,fp)

        with open(self.save_path+self.fileName[:-8]+'.good.turing.unseen.json', 'w') as fp:
            json.dump(self.unseen_probabilities,fp)

        return


    def run(self, fileName):
        self.fileName = fileName
        self.read_and_count()
        self.calculate_probabilities()
        self.count_unseen_ngrams()
        self.good_turing_smoothing()
        self.save_probabilities()
        return

    #Utility function
    def pad_word(self, word, ngramSize):
        """
        adds padding to start and end of each word with '<' (start) and '>' (end) symbol.
        """
        if ngramSize > 1:
            start_chars = ['<'] * (ngramSize-1)
            new_word = start_chars + list(word)
            end_char = ['>']
            new_word = new_word + end_char
            return new_word
        else:
            return word

    #utility function
    def log_probability_of_word(self, word, ngram_size, language):
        """
        This function loads probabilites of a language ('udhr-eng').
        It returns sum of log probabilities of ngrams inside the word 'word' with respect to ngram size of 'ngram_size'.
        """

        """
        ngram_size cannot be greater than the maximum ngram size that has been used to trian the language model.
        """

        with open(self.save_path+language+'.no.smoothing.json', 'r') as fp:
            no_smoothing = json.load(fp)

        with open(self.save_path+language+'.good.turing.seen.json', 'r') as fp:
            good_turing_seen = json.load(fp)

        with open(self.save_path+language+'.good.turing.unseen.json', 'r') as fp:
            good_turing_unseen = json.load(fp)

        padded_word = self.pad_word(word, ngram_size)

        sum_of_log_probs_no_smoothing = 0.0
        # for no smoothing approach
        for i,_ in enumerate(padded_word):
            if(i>=ngram_size - 1):
                post = padded_word[i]
                prior = padded_word[i - (ngram_size - 1) : i]
                temp = ''.join(prior) + post
                if temp in no_smoothing:
                    sum_of_log_probs_no_smoothing += math.log(float(no_smoothing[temp]))
                else:
                    #unseen ngram temp in no smoothing method. it should be math.log(0) which is -infinity.
                    sum_of_log_probs_no_smoothing += math.log(0.0000000001)


        sum_of_log_probs_good_turing = 0.0
        # for good_turing approach.
        for i,_ in enumerate(padded_word):
            if(i>=ngram_size - 1):
                post = padded_word[i]
                prior = padded_word[i - (ngram_size - 1) : i]
                temp = ''.join(prior) + post
                if temp in good_turing_seen:
                    sum_of_log_probs_good_turing += math.log(float(good_turing_seen[temp]))
                else:
                    #unseen ngram temp in good_turing.
                    for each in good_turing_unseen.keys():
                        if(int(each)==ngram_size):
                            sum_of_log_probs_good_turing += math.log(float(good_turing_unseen[each]))


        return sum_of_log_probs_no_smoothing, sum_of_log_probs_good_turing



if __name__ == "__main__":
    lm = LM()

    """
    #creating language model over all languages.
    for (_, _, filenames) in walk(lm.train_path):
        for each in filenames:
            lm.run(each)
    """

    #some examples!
    no_smoothing, good_turing = lm.log_probability_of_word("accuracy", 3, "udhr-amc")
    print str(no_smoothing) + "|" + str(good_turing)

    no_smoothing, good_turing = lm.log_probability_of_word("abbreviation", 3, "udhr-amc")
    print str(no_smoothing) + "|" + str(good_turing)

    no_smoothing, good_turing = lm.log_probability_of_word("apple", 3, "udhr-amc")
    print str(no_smoothing) + "|" + str(good_turing)

    no_smoothing, good_turing = lm.log_probability_of_word("microsoft", 3, "udhr-amc")
    print str(no_smoothing) + "|" + str(good_turing)
