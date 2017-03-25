#!/usr/bin/env python
# -*- coding: utf-8 -*-
from scipy import optimize
import numpy as np
import math
import itertools
import json
from os import walk
import os



class LM:
    """
    ngram character-level language model
    """

    #max_n_gram_size.
    max_n_gram_size = 7
    train_path = "./650_a3_train/"
    dev_path = "./650_a3_dev/"
    save_path = "./probs/"
    best_ngram_size_path = "./best_ngram_size/"
    test_path = "./650_a3_test_final/"

    unsmoothed_output_buffer = []
    laplace_output_buffer = []
    good_turing_output_buffer = []

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
        This function counts all ngrams up to max_n_gram_size and adds them in a dictionary of dictionaries.
        For example, ngram_dictionary has a key '2' which has a corresponding value. That value, which is a dictionary, has counts for all bi-grams.

        """

        self.ngram_dictionary = {}

        ngram_size = 1
        while (ngram_size <= self.max_n_gram_size):
            fd = open(self.train_path+self.fileName,"r")
            number_of_words = 0  # to capture number of words in the file.
            temp_dic = {} # to have all ngrams with 'ngram' size.
            lines = fd.readlines()
            lines = [line.strip() for line in lines]
            for line in lines:
                words = line.split()
                number_of_words += len(words)
                for word in words:
                    word = self.robust_decode(word)
                    padded_word = self.pad_word(word, ngram_size)
                    padded_word = list(padded_word)
                    for i,_ in enumerate(padded_word):
                        if(i>=ngram_size - 1):
                            post = padded_word[i]
                            prior = padded_word[i - (ngram_size - 1) : i]
                            ngram = ''.join(prior) + post
                            if ngram in temp_dic:
                                temp_dic[ngram] += 1
                            else:
                                temp_dic[ngram] = 1

            self.ngram_dictionary[ngram_size] = temp_dic
            ngram_size += 1

        self.number_of_words = number_of_words

        # find the total count of all unigram characters.
        self.unigram_total = 0
        self.alphabet = []
        for unigram, count in self.ngram_dictionary[1].iteritems():
            self.alphabet.append(unigram)
            self.unigram_total += count

        #store these data. important when loading
        temp_dic = {}
        temp_dic[0] = self.unigram_total
        temp_dic[1] = self.alphabet
        self.ngram_dictionary[0] = temp_dic

        return

    def compute_unsmoothed_probabilities(self):
        """
        This function computes unsmoothed probabilities for ngrams.
        unsmoothed_probabilities has these probabilities.
        """

        self.unsmoothed_probabilities = {}
        ngram_size = 1
        while(ngram_size <= self.max_n_gram_size):
            if(ngram_size==1):
                temp_dic = {}
                for ngram, count in self.ngram_dictionary[ngram_size].iteritems():
                    temp_dic[ngram] = float(count)/float(self.unigram_total)

                self.unsmoothed_probabilities[ngram_size] = temp_dic
            else:
                temp_dic = {}
                for ngram, count in self.ngram_dictionary[ngram_size].iteritems():
                    if ngram[:-1] in self.ngram_dictionary[ngram_size - 1]:
                        temp_dic[ngram] = float(count)/float(self.ngram_dictionary[ngram_size - 1][ngram[:-1]])
                    else:
                        # ngram is starter symbols
                        temp_dic[ngram] = float(count)/float(self.number_of_words)

                self.unsmoothed_probabilities[ngram_size] = temp_dic

            ngram_size += 1

        return

    def compute_laplace_probabilities(self):
        """
        This function computes laplace probabilities for ngrams.
        laplace_probabilities has these probabilities.
        """

        self.laplace_probabilities = {}
        ngram_size = 1
        while(ngram_size <= self.max_n_gram_size):
            if(ngram_size==1):
                temp_dic = {}
                for ngram, count in self.ngram_dictionary[ngram_size].iteritems():
                    temp_dic[ngram] = float(count + 1)/float(self.unigram_total + len(self.alphabet))

                self.laplace_probabilities[ngram_size] = temp_dic
            else:
                temp_dic = {}
                for ngram, count in self.ngram_dictionary[ngram_size].iteritems():
                    if ngram[:-1] in self.ngram_dictionary[ngram_size - 1]:
                        temp_dic[ngram] = float(count + 1)/float(self.ngram_dictionary[ngram_size - 1][ngram[:-1]] + len(self.alphabet))
                    else:
                        # ngram is starter symbols
                        temp_dic[ngram] = float(count + 1)/float(self.number_of_words + len(self.alphabet))

                self.laplace_probabilities[ngram_size] = temp_dic

            ngram_size += 1

        return

    def count_all_ngrams(self, ngram_size):
        """
        Count all possible ngrams that can be generated from alphabet with respect to ngram size.
        """

        #The is the number of all possible ngrams of ngram_size from alphabet considering start and end symbols.
        set_size = len(self.alphabet)
        num1 = math.pow(set_size, ngram_size)
        num2 = math.pow(set_size, ngram_size-1)
        num3 = (float(num1-1)/float(set_size-1)) - 1
        return num1 + num2 + num3

    def count_unseen_ngrams(self):
        """
        Counts the ngrams that have not been seen in the training data and adds that count for each ngram size in self.N0 dictionary.
        """
        self.N0= {}
        ngram_size = 2
        while(ngram_size <= self.max_n_gram_size):
            num_all_ngrams = self.count_all_ngrams(ngram_size)
            num_seen_ngrams = 0
            for each in self.ngram_dictionary[ngram_size].keys():
                num_seen_ngrams += 1

            self.N0[ngram_size] = num_all_ngrams - num_seen_ngrams
            ngram_size += 1

        return

    def count_seen_ngrams(self, r, ngram_size):
        "Count seen ngrams of ngram_size which appeares r times in the corpus"
        total = 0
        for each in self.ngram_dictionary[ngram_size].keys():
            if (self.ngram_dictionary[ngram_size][each]==r):
                total += 1

        return total

    def smooth_Ns(self, max_r, Ns):
        """
        This function fits a power law function F(r) = a*pow(r,b); b < -1 over r.
        This smoothing is necessary for good turing to smooth Ns as discussed here:
        http://www.csd.uwo.ca/faculty/olga/Courses//Winter2013/CS4442_9542b/L9-NLP-LangModel.pdf

        fitting code link:
        http://scipy.github.io/old-wiki/pages/Cookbook/FittingData
        """

        ##########
        # Fitting the data -- Least Squares Method
        ##########

        # Power-law fitting is best done by first converting
        # to a linear equation and then fitting to a straight line.
        #
        #  y = a * x^b
        #  log(y) = log(a) + b*log(x)
        #


        xdata = np.arange(1, max_r + 2)
        ydata = np.array(Ns) + 1
        logx = np.log10(xdata)
        logy = np.log10(ydata)

        # define our (line) fitting function
        fitfunc = lambda p, x: p[0] + p[1] * x
        errfunc = lambda p, x, y: (y - fitfunc(p, x))

        pinit = [2.0, -2.0]
        out = optimize.leastsq(errfunc, pinit, args=(logx, logy), full_output=1)

        pfinal = out[0]

        # Define function for calculating a power law
        powerlaw = lambda x, amp, index: amp * (x**index)
        index = pfinal[1]
        amp = 10.0**pfinal[0]


        return_list = []
        for i in range(max_r + 1):
            return_list.append(powerlaw(i+1, amp, index))

        return return_list

    def compute_good_turing_probabilities(self):
        """
        Good-Turing smoothing with power law fitting for smoothing Ns.
        link:
        http://www.csd.uwo.ca/faculty/olga/Courses//Winter2013/CS4442_9542b/L9-NLP-LangModel.pdf
        """

        self.good_turing_seen_probabilities = {}
        self.good_turing_unseen_probabilities = {}

        temp_dic = {}
        for each in self.alphabet:
            temp_dic[each] = self.unsmoothed_probabilities[1][each]

        self.good_turing_seen_probabilities[1] = temp_dic

        ngram_size = 2
        while (ngram_size <= self.max_n_gram_size):

            max_r = 0
            for _, count in self.ngram_dictionary[ngram_size].iteritems():
                if(count > max_r):
                        max_r = count

            probs = []
            Ns = []
            for r in range(max_r + 1):
                if(r==0):
                    Ns.append(self.N0[ngram_size])
                else:
                    Ns.append(self.count_seen_ngrams(r, ngram_size))



            total_N = 0
            for r in range(max_r + 1):
                total_N += r * Ns[r]


            Smoothed_Ns = self.smooth_Ns(max_r, Ns)

            for r in range(max_r):
                probs.append( float( ( float(r+1) * float(Smoothed_Ns[r+1]) ) / ( float(total_N) * float(Smoothed_Ns[r]) ) ) )

            probs.append(float(max_r)/float(total_N))

            # make it a probability distribution.
            sum_probs = 0.0
            for r in range(max_r + 1):
                sum_probs += Ns[r] * probs[r]

            for r in range(max_r + 1):
                probs[r] = float(probs[r]) / float(sum_probs)


            temp_dic = {}
            for ngram , count in self.ngram_dictionary[ngram_size].iteritems():
                temp_dic[ngram] = probs[count]

            self.good_turing_seen_probabilities[ngram_size] = temp_dic
            self.good_turing_unseen_probabilities[ngram_size] = probs[0]

            ngram_size += 1

        return

    def save_probabilities(self):
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        with open(self.save_path+self.fileName[:-8]+'.dictionary.json', 'w') as fp:
            json.dump(self.ngram_dictionary, fp)

        with open(self.save_path+self.fileName[:-8]+'.unsmoothing.json', 'w') as fp:
            json.dump(self.unsmoothed_probabilities, fp)

        with open(self.save_path+self.fileName[:-8]+'.laplace.json', 'w') as fp:
            json.dump(self.laplace_probabilities, fp)

        with open(self.save_path+self.fileName[:-8]+'.good.turing.seen.json', 'w') as fp:
            json.dump(self.good_turing_seen_probabilities,fp)

        with open(self.save_path+self.fileName[:-8]+'.good.turing.unseen.json', 'w') as fp:
            json.dump(self.good_turing_unseen_probabilities,fp)

        return

    def run(self, fileName):
        self.fileName = fileName
        self.read_and_count()
        self.compute_unsmoothed_probabilities()
        self.compute_laplace_probabilities()
        self.count_unseen_ngrams()
        self.compute_good_turing_probabilities()
        self.save_probabilities()
        return



    """
        **********************************************
                        Utility function
        **********************************************
    """
    def pad_word(self, word, ngram_size):
        """
        adds paddings to start and end of each word with '<' (start) and '>' (end) symbols.
        """
        if ngram_size > 1:
            start_chars = ['<'] * (ngram_size-1)
            new_word = start_chars + list(word)
            end_char = ['>']
            new_word = new_word + end_char
            return new_word
        else:
            return word




    """
        **********************************************
                        Utility function
        **********************************************
    """
    def log_probability_of_word(self, word, ngram_size, language, dictionary, unsmoothing, laplace, good_turing_seen, good_turing_unseen):
        """
        This function loads probabilites of a language (for example 'udhr-eng').
        It returns sum of log probabilities of ngrams inside the word 'word' with respect to ngram size of 'ngram_size'.
        """

        """
        ngram_size cannot be greater than the maximum ngram size that has been used to trian the language model.
        """
        if(ngram_size < 1):
            print "ERROR!"
            print "ngram_size should be greater than one"
            return


        word = self.robust_decode(word)
        padded_word = self.pad_word(word, ngram_size)


        sum_of_log_probs_no_smoothing = 0.0
        # for unsmoothing approach
        for i,_ in enumerate(padded_word):
            if(i>=ngram_size - 1):
                post = padded_word[i]
                prior = padded_word[i - (ngram_size - 1) : i]
                temp = ''.join(prior) + post
                if temp in unsmoothing[str(ngram_size)]:
                    sum_of_log_probs_no_smoothing += math.log(float(unsmoothing[str(ngram_size)][temp]))
                else:
                    #unseen ngram temp in unsmoothing method. it should be math.log(0) which is -infinity.
                    sum_of_log_probs_no_smoothing += math.log(0.00000000000000000001)


        sum_of_log_probs_laplace = 0.0
        # for laplace approach
        for i,_ in enumerate(padded_word):
            if(i>=ngram_size - 1):
                post = padded_word[i]
                prior = padded_word[i - (ngram_size - 1) : i]
                temp = ''.join(prior) + post
                if temp in laplace[str(ngram_size)]:
                    sum_of_log_probs_laplace += math.log(float(laplace[str(ngram_size)][temp]))
                else:
                    #unseen ngram temp in laplace method.
                    if temp[:-1] in dictionary[str(ngram_size)]:
                        sum_of_log_probs_laplace += math.log(float(1.0)/float(dictionary[str(ngram_size)][temp[:-1]] + len(dictionary["0"]["1"])))
                    else:
                        sum_of_log_probs_laplace += math.log(float(1.0)/float(0 + len(dictionary["0"]["1"])))



        sum_of_log_probs_good_turing = 0.0
        # for good_turing approach.
        for i,_ in enumerate(padded_word):
            if(i>=ngram_size - 1):
                post = padded_word[i]
                prior = padded_word[i - (ngram_size - 1) : i]
                temp = ''.join(prior) + post
                if temp in good_turing_seen[str(ngram_size)]:
                    temp_prob = float(good_turing_seen[str(ngram_size)][temp])
                else:
                    #unseen ngram temp in good_turing.
                    for each in good_turing_unseen.keys():
                        if(int(each)==ngram_size):
                            temp_prob = float(good_turing_unseen[each])


                prior_prob = 1.0

                if(ngram_size>1):
                    prior_ngram = ''.join(prior)
                    if prior_ngram in good_turing_seen[str(ngram_size-1)]:
                        prior_prob = float(good_turing_seen[str(ngram_size-1)][prior_ngram])
                    else:
                        #unseen ngram prior_ngram in good_turing.
                        for each in good_turing_unseen.keys():
                            if(int(each)==(ngram_size-1)):
                                prior_prob = float(good_turing_unseen[each])

                sum_of_log_probs_good_turing += math.log(float(temp_prob)/float(prior_prob))


        return sum_of_log_probs_no_smoothing, sum_of_log_probs_laplace, sum_of_log_probs_good_turing




    """
        **********************************************
                        Utility function
        **********************************************
    """
    def log_preplexity_of_text(self, textPath, fileName, ngram_size, language):
        """
        This function computes log preplexity of a text with a character language model trained in the language 'language'
        with respect to ngram_size.

        log preplexity of a text is negative mean of log probabilities of each word in the text.

        """

        with open(self.save_path+language+'.dictionary.json', 'r') as fp:
            dictionary = json.load(fp)

        with open(self.save_path+language+'.unsmoothing.json', 'r') as fp:
            unsmoothing = json.load(fp)

        with open(self.save_path+language+'.laplace.json', 'r') as fp:
            laplace = json.load(fp)

        with open(self.save_path+language+'.good.turing.seen.json', 'r') as fp:
            good_turing_seen = json.load(fp)

        with open(self.save_path+language+'.good.turing.unseen.json', 'r') as fp:
            good_turing_unseen = json.load(fp)

        fd = open(textPath+fileName,"r")

        unsmoothing_sum_of_log_probabilities = 0.0
        laplace_sum_of_log_probabilities = 0.0
        good_turing_sum_of_log_probabilities = 0.0

        number_of_words = 0  # to capture number of words in the file.
        lines = fd.readlines()
        lines = [line.strip() for line in lines]
        for line in lines:
            words = line.split()
            number_of_words += len(words)
            for word in words:
                t1, t2, t3 = self.log_probability_of_word(word, ngram_size, language, dictionary, unsmoothing, laplace, good_turing_seen, good_turing_unseen)
                unsmoothing_sum_of_log_probabilities += t1
                laplace_sum_of_log_probabilities += t2
                good_turing_sum_of_log_probabilities += t3

        unsmoothing_preplexity = - (unsmoothing_sum_of_log_probabilities/float(number_of_words))
        laplace_preplexity = - (laplace_sum_of_log_probabilities/float(number_of_words))
        good_turing_preplexity = - (good_turing_sum_of_log_probabilities/float(number_of_words))

        fd.close()

        return unsmoothing_preplexity, laplace_preplexity, good_turing_preplexity




    """
        **********************************************
                        Utility function
        **********************************************
    """
    def find_best_ngram_size_for_each_language(self):

        #Find best ngram_size for each language based one dev set.
        if not os.path.exists(self.best_ngram_size_path):
            os.makedirs(self.best_ngram_size_path)

        for (_, _, filenames) in walk(self.dev_path):
            for each in filenames:

                # forget unigram model!
                ngram_size = 2
                no_smoothing_best_preplexity = 1000000000000000
                laplace_best_preplexity = 1000000000000000
                good_turing_best_preplexity = 1000000000000000

                no_smoothing_best_ngram_size = 1
                laplace_best_ngram_size = 1
                good_turing_best_ngram_size = 1

                while(ngram_size <= self.max_n_gram_size):
                    no_smoothing, laplace, good_turing = self.log_preplexity_of_text(self.dev_path, each, ngram_size, each[:-8])

                    if(no_smoothing < no_smoothing_best_preplexity):
                        no_smoothing_best_ngram_size = ngram_size
                        no_smoothing_best_preplexity = no_smoothing


                    if(laplace < laplace_best_preplexity):
                        laplace_best_ngram_size = ngram_size
                        laplace_best_preplexity = laplace


                    if(good_turing < good_turing_best_preplexity):
                        good_turing_best_ngram_size = ngram_size
                        good_turing_best_preplexity = good_turing


                    ngram_size += 1

                temp_dic = {}
                temp_dic["no_smoothing"] = no_smoothing_best_ngram_size
                temp_dic["laplace"] = laplace_best_ngram_size
                temp_dic["good_turing"] = good_turing_best_ngram_size

                with open(self.best_ngram_size_path+each[:-8]+'.best_ngram_size.json', 'w') as fp:
                    json.dump(temp_dic, fp)

        return




    """
        **********************************************
                        Utility function
        **********************************************
    """
    def identify_language_of_text(self, testFileName):
        """
        This function identifies top 3 probable languages of the textFile with no_smoothing, laplace and good-turing methods.

        """

        top_languages_no_smoothing = {}
        top_languages_laplace = {}
        top_languages_good_turing = {}

        language_list = []
        for (_, _, filenames) in walk(self.train_path):
            for each in filenames:
                language_list.append(each[:-8])

        for each in language_list:
            #load best params of this language.
            with open(self.best_ngram_size_path+each+'.best_ngram_size.json', 'r') as fp:
                best_params = json.load(fp)

            no_smoothing_best_ngram_size = best_params["no_smoothing"]
            laplace_best_ngram_size = best_params["laplace"]
            good_turing_best_ngram_size = best_params["good_turing"]

            no_smoothing , _ , _ = self.log_preplexity_of_text(self.test_path, testFileName, no_smoothing_best_ngram_size, each)
            _ , laplace , _ = self.log_preplexity_of_text(self.test_path, testFileName, laplace_best_ngram_size, each)
            _ , _ , good_turing = self.log_preplexity_of_text(self.test_path, testFileName, good_turing_best_ngram_size, each)

            top_languages_no_smoothing[each] = no_smoothing
            top_languages_laplace[each] = laplace
            top_languages_good_turing[each] = good_turing


        top_no_smoothing = sorted(top_languages_no_smoothing.items(), key=lambda x:x[1])
        top_laplace = sorted(top_languages_laplace.items(), key=lambda x:x[1])
        top_good_turing = sorted(top_languages_good_turing.items(), key=lambda x:x[1])

        #saving results in output_buffers

        temp_string = testFileName + "\t" + top_no_smoothing[0][0]+ ".txt.tra" + "\t" + str(top_no_smoothing[0][1]) +  \
                        "\t" + top_no_smoothing[1][0]+ ".txt.tra" + "\t" + str(top_no_smoothing[1][1]) + \
                        "\t" + top_no_smoothing[2][0]+ ".txt.tra" + "\t" + str(top_no_smoothing[2][1])

        self.unsmoothed_output_buffer.append(temp_string)


        temp_string = testFileName + "\t" + top_laplace[0][0]+ ".txt.tra" + "\t" + str(top_laplace[0][1]) + \
                        "\t" + top_laplace[1][0]+ ".txt.tra" + "\t" + str(top_laplace[1][1]) + \
                        "\t" + top_laplace[2][0]+ ".txt.tra" + "\t" + str(top_laplace[2][1])

        self.laplace_output_buffer.append(temp_string)


        temp_string = testFileName + "\t" + top_good_turing[0][0]+ ".txt.tra" + "\t" + str(top_good_turing[0][1]) + \
                        "\t" + top_good_turing[1][0]+ ".txt.tra" + "\t" + str(top_good_turing[1][1]) + \
                        "\t" + top_good_turing[2][0]+ ".txt.tra" + "\t" + str(top_good_turing[2][1])

        self.good_turing_output_buffer.append(temp_string)


        return

    def print_results(self):
        no_smoothing_file = open("./results_unsmoothed.txt", "a")
        laplace_file = open("./results_add_one.txt", "a")
        good_turing_file = open("./results_good_turing.txt", "a")

        # sort to print alphabetically

        for each in sorted(self.unsmoothed_output_buffer):
            no_smoothing_file.write(each)
            no_smoothing_file.write("\n")

        for each in sorted(self.laplace_output_buffer):
            laplace_file.write(each)
            laplace_file.write("\n")

        for each in sorted(self.good_turing_output_buffer):
            good_turing_file.write(each)
            good_turing_file.write("\n")


        no_smoothing_file.close()
        laplace_file.close()
        good_turing_file.close()

        return


if __name__ == "__main__":
    lm = LM()

    #creating language model over all languages.
    for (_, _, filenames) in walk(lm.train_path):
        for each in filenames:
            lm.run(each)


    #examples
    """
    no_smoothing, laplace, good_turing = lm.log_preplexity_of_text(lm.dev_path, "udhr-ssw.txt.dev", 3, "udhr-eng")
    print str(no_smoothing) + "|" + str(laplace) + "|" + str(good_turing)

    no_smoothing, laplace, good_turing = lm.log_preplexity_of_text(lm.dev_path, "udhr-ssw.txt.dev", 3, "udhr-amc")
    print str(no_smoothing) + "|" + str(laplace) + "|" + str(good_turing)

    no_smoothing, laplace, good_turing = lm.log_preplexity_of_text(lm.dev_path, "udhr-ssw.txt.dev", 3, "udhr-ssw")
    print str(no_smoothing) + "|" + str(laplace) + "|" + str(good_turing)
    """

    lm.find_best_ngram_size_for_each_language()

    #examples
    """
    lm.identify_language_of_text("udhr-eng.txt.dev")
    print
    lm.identify_language_of_text("udhr-amc.txt.dev")
    print
    lm.identify_language_of_text("udhr-ssw.txt.dev")
    """


    #test step
    for (_, _, filenames) in walk(lm.test_path):
        for each in filenames:
            lm.identify_language_of_text(each)

    lm.print_results()
