import pickle, string, numpy, getopt, sys, random, time, re, pprint
import datetime, os;

import nltk;
import numpy;
import cProfile


def main():
    # parameter set 1
    input_directory = "./nips-abstract"

    input_directory = input_directory.rstrip("/");
    # corpus_name = os.path.basename(input_directory);

    '''
    output_directory = options.output_directory;
    if not os.path.exists(output_directory):
        os.mkdir(output_directory);
    output_directory = os.path.join(output_directory, corpus_name);
    if not os.path.exists(output_directory):
        os.mkdir(output_directory);
    '''

    # Document
    train_docs_path = os.path.join(input_directory, 'train.dat')
    input_doc_stream = open(train_docs_path, 'r');
    train_docs = [];
    for line in input_doc_stream:
        train_docs.append(line.strip().lower());
    print("successfully load all training docs from %s..." % (os.path.abspath(train_docs_path)));

    # Vocabulary
    vocabulary_path = os.path.join(input_directory, 'voc.dat');
    input_voc_stream = open(vocabulary_path, 'r');
    vocab = [];
    for line in input_voc_stream:
        vocab.append(line.strip().lower().split()[0]);
    vocab = list(set(vocab));
    print("successfully load all the words from %s..." % (os.path.abspath(vocabulary_path)));

    # parameter 2
    number_of_topics = 10;
    alpha_mu = 0;
    alpha_sigma = 1;
    alpha_beta = 1.0 / len(vocab);

    # parameter set 3
    training_iterations = 1;

    import variational_bayes
    ctm_inferencer = variational_bayes.VariationalBayes();

    ctm_inferencer._initialize(train_docs, vocab, number_of_topics, alpha_mu, alpha_sigma, alpha_beta);

    for iteration in range(training_iterations):
        clock = time.time();
        log_likelihood = ctm_inferencer.learning();
        clock = time.time() - clock;

        # print 'training iteration %d finished in %f seconds: number-of-topics = %d, log-likelihood = %f' % (hdp._iteration_counter, clock, hdp._K, log_likelihood);

    # gamma_path = os.path.join(output_directory, 'gamma.txt');
    # numpy.savetxt(gamma_path, hdp._document_topic_distribution);

    # topic_inactive_counts_path = os.path.join(output_directory, "topic_inactive_counts.txt");
    # numpy.savetxt(topic_inactive_counts_path, hdp._topic_inactive_counts);


if __name__ == '__main__':
    main()
