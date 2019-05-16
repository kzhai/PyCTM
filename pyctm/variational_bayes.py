"""
VariationalBayes for Correlated Topic Models
@author: Ke Zhai (zhaike@cs.umd.edu)
"""

import multiprocessing
import nltk
import numpy
import queue
import scipy
import scipy.misc
import scipy.optimize
import sklearn
import sklearn.covariance
import string
import sys
import time

from inferencer import compute_dirichlet_expectation
from inferencer import Inferencer


class Process_E_Step_Queue(multiprocessing.Process):
    def __init__(
            self,
            task_queue,
            model_parameters,
            optimize_doc_lambda,
            # optimize_doc_nu_square,
            optimize_doc_nu_square_in_log_space,
            result_doc_parameter_queue,
            result_log_likelihood_queue,
            result_sufficient_statistics_queue,
            diagonal_covariance_matrix=False,
            parameter_iteration=10,
            parameter_converge_threshold=1e-3):
        multiprocessing.Process.__init__(self)

        self._task_queue = task_queue
        self._result_doc_parameter_queue = result_doc_parameter_queue
        self._result_log_likelihood_queue = result_log_likelihood_queue
        self._result_sufficient_statistics_queue = result_sufficient_statistics_queue

        self._parameter_iteration = parameter_iteration

        self._diagonal_covariance_matrix = diagonal_covariance_matrix
        if self._diagonal_covariance_matrix:
            (
                self._E_log_eta, self._alpha_mu,
                self._alpha_sigma) = model_parameters
        else:
            (
                self._E_log_eta, self._alpha_mu, self._alpha_sigma,
                self._alpha_sigma_inv) = model_parameters
        (self._number_of_topics, self._number_of_types) = self._E_log_eta.shape

        if result_sufficient_statistics_queue != None:
            self._E_log_prob_eta = self._E_log_eta - scipy.misc.logsumexp(
                self._E_log_eta, axis=1)[:, numpy.newaxis]

        self.optimize_doc_lambda = optimize_doc_lambda
        # self.optimize_doc_nu_square = optimize_doc_nu_square;
        self.optimize_doc_nu_square_in_log_space = optimize_doc_nu_square_in_log_space

    def run(self):
        document_log_likelihood = 0
        words_log_likelihood = 0

        # initialize a V-by-K matrix phi sufficient statistics
        phi_sufficient_statistics = numpy.zeros(
            (self._number_of_topics, self._number_of_types))

        # initialize a D-by-K matrix lambda and nu_square values
        # lambda_values = numpy.zeros((number_of_documents, self._number_of_topics)) # + self._alpha_mu[numpy.newaxis, :];
        # nu_square_values = numpy.ones((number_of_documents, self._number_of_topics)) # + self._alpha_sigma[numpy.newaxis, :];

        while not self._task_queue.empty():
            try:
                (doc_id, term_ids, term_counts) = self._task_queue.get_nowait()

            except queue.Empty:
                continue

            doc_lambda = numpy.zeros(self._number_of_topics)
            doc_nu_square = numpy.ones(self._number_of_topics)

            assert term_counts.shape == (1, len(term_ids))
            # compute the total number of words
            doc_word_count = numpy.sum(term_counts)

            # initialize gamma for this document
            # doc_lambda = lambda_values[doc_id, :]
            # doc_nu_square = nu_square_values[doc_id, :]
            '''
            if self._diagonal_covariance_matrix:
                doc_lambda = numpy.random.multivariate_normal(self._alpha_mu, numpy.diag(self._alpha_sigma));
                doc_nu_square = numpy.copy(self._alpha_sigma);
            else:
                #doc_lambda = numpy.random.multivariate_normal(self._alpha_mu[0, :], self._alpha_sigma);
                #doc_nu_square = numpy.copy(numpy.diag(self._alpha_sigma));
                doc_lambda = numpy.random.multivariate_normal(numpy.zeros(self._number_of_topics), numpy.eye(self._number_of_topics))
                doc_nu_square = numpy.ones(self._number_of_topics)
            assert doc_lambda.shape==(self._number_of_topics,)
            assert doc_nu_square.shape==(self._number_of_topics,)
            '''

            # term_ids = word_ids[doc_id];
            # term_counts = word_cts[doc_id];

            # update zeta in close form
            # doc_zeta = numpy.sum(numpy.exp(doc_lambda+0.5*doc_nu_square));
            doc_zeta_factor = doc_lambda + 0.5 * doc_nu_square
            assert doc_zeta_factor.shape == (self._number_of_topics, )
            doc_zeta_factor = numpy.tile(
                doc_zeta_factor, (self._number_of_topics, 1))
            assert doc_zeta_factor.shape == (
                self._number_of_topics, self._number_of_topics)

            for local_parameter_iteration_index in range(
                    self._parameter_iteration):
                # update phi in close form
                assert self._E_log_eta.shape == (
                    self._number_of_topics, self._number_of_types)
                log_phi = self._E_log_eta[:, term_ids] + doc_lambda[:, numpy.
                                                                    newaxis]
                assert log_phi.shape == (self._number_of_topics, len(term_ids))
                log_phi -= scipy.misc.logsumexp(log_phi,
                                                axis=0)[numpy.newaxis, :]
                assert log_phi.shape == (self._number_of_topics, len(term_ids))

                #
                #
                #
                #
                #

                # update lambda
                sum_phi = numpy.exp(
                    scipy.misc.logsumexp(
                        log_phi + numpy.log(term_counts), axis=1))
                arguments = (
                    doc_nu_square, doc_zeta_factor, sum_phi, doc_word_count)
                doc_lambda = self.optimize_doc_lambda(doc_lambda, arguments)
                # print "update lambda of doc %d to %s" % (doc_id, doc_lambda)

                #
                #
                #
                #
                #

                # update zeta in close form
                # doc_zeta = numpy.sum(numpy.exp(doc_lambda+0.5*doc_nu_square));
                doc_zeta_factor = doc_lambda + 0.5 * doc_nu_square
                assert doc_zeta_factor.shape == (self._number_of_topics, )
                doc_zeta_factor = numpy.tile(
                    doc_zeta_factor, (self._number_of_topics, 1))
                assert doc_zeta_factor.shape == (
                    self._number_of_topics, self._number_of_topics)

                #
                #
                #
                #
                #

                # update nu_square
                arguments = (doc_lambda, doc_zeta_factor, doc_word_count)
                # doc_nu_square = self.optimize_doc_nu_square(doc_nu_square, arguments);
                doc_nu_square = self.optimize_doc_nu_square_in_log_space(
                    doc_nu_square, arguments)
                # print "update nu of doc %d to %s" % (doc_id, doc_nu_square)

                #
                #
                #
                #
                #

                # update zeta in close form
                # doc_zeta = numpy.sum(numpy.exp(doc_lambda+0.5*doc_nu_square));
                doc_zeta_factor = doc_lambda + 0.5 * doc_nu_square
                assert doc_zeta_factor.shape == (self._number_of_topics, )
                doc_zeta_factor = numpy.tile(
                    doc_zeta_factor, (self._number_of_topics, 1))
                assert doc_zeta_factor.shape == (
                    self._number_of_topics, self._number_of_topics)

                # mean_change = numpy.mean(abs(gamma_update - lambda_values[doc_id, :]));
                # lambda_values[doc_id, :] = gamma_update;
                # if mean_change <= local_parameter_converge_threshold:
                # break;

                # print doc_id, local_parameter_iteration_index

            # print "process document %d..." % doc_id

            # document_log_likelihood -= 0.5 * self._number_of_topics * numpy.log(2 * numpy.pi)
            if self._diagonal_covariance_matrix:
                document_log_likelihood -= 0.5 * numpy.sum(
                    numpy.log(self._alpha_sigma))
                document_log_likelihood -= 0.5 * numpy.sum(
                    doc_nu_square / self._alpha_sigma)
                document_log_likelihood -= 0.5 * numpy.sum(
                    (doc_lambda - self._alpha_mu)**2 / self._alpha_sigma)
            else:
                # document_log_likelihood -= 0.5 * numpy.log(numpy.linalg.det(self._alpha_sigma));
                document_log_likelihood -= 0.5 * numpy.log(
                    scipy.linalg.det(self._alpha_sigma) + 1e-30)
                document_log_likelihood -= 0.5 * numpy.sum(
                    doc_nu_square * numpy.diag(self._alpha_sigma_inv))
                document_log_likelihood -= 0.5 * numpy.dot(
                    numpy.dot(
                        (self._alpha_mu - doc_lambda[numpy.newaxis, :]),
                        self._alpha_sigma_inv),
                    (self._alpha_mu - doc_lambda[numpy.newaxis, :]).T)

            document_log_likelihood += numpy.sum(
                numpy.sum(numpy.exp(log_phi) * term_counts, axis=1) *
                doc_lambda)
            # use the fact that doc_zeta = numpy.sum(numpy.exp(doc_lambda+0.5*doc_nu_square)), to cancel the factors
            document_log_likelihood -= scipy.misc.logsumexp(
                doc_lambda + 0.5 * doc_nu_square) * doc_word_count

            document_log_likelihood += 0.5 * self._number_of_topics
            # document_log_likelihood += 0.5 * self._number_of_topics * numpy.log(2 * numpy.pi)
            document_log_likelihood += 0.5 * numpy.sum(
                numpy.log(doc_nu_square))

            document_log_likelihood -= numpy.sum(
                numpy.exp(log_phi) * log_phi * term_counts)

            # Note: all terms including E_q[p(\eta | \beta)], i.e., terms involving \Psi(\eta), are cancelled due to \eta updates in M-step
            if self._result_sufficient_statistics_queue == None:
                # compute the phi terms
                words_log_likelihood += numpy.sum(
                    numpy.exp(log_phi + numpy.log(term_counts)) *
                    self._E_log_prob_eta[:, term_ids])

            # lambda_values[doc_id, :] = doc_lambda;
            # nu_square_values[doc_id, :] = doc_nu_square;

            assert numpy.all(doc_nu_square > 0)

            assert log_phi.shape == (self._number_of_topics, len(term_ids))
            assert term_counts.shape == (1, len(term_ids))
            phi_sufficient_statistics[:, term_ids] += numpy.exp(
                log_phi + numpy.log(term_counts))

            # if (doc_id+1) % 1000==0:
            # print "successfully processed %d documents..." % (doc_id+1);

            self._result_doc_parameter_queue.put(
                (doc_id, doc_lambda, doc_nu_square))

            self._task_queue.task_done()

        if self._result_sufficient_statistics_queue == None:
            self._result_log_likelihood_queue.put(words_log_likelihood)
        else:
            self._result_log_likelihood_queue.put(document_log_likelihood)
            self._result_sufficient_statistics_queue.put(
                phi_sufficient_statistics)


class VariationalBayes(Inferencer):
    """
    """

    def __init__(
            self, scipy_optimization_method=None,
            hessian_free_optimization=False, diagonal_covariance_matrix=False,
            hyper_parameter_optimize_interval=1,
            hessian_direction_approximation_epsilon=1e-6
            # hyper_parameter_iteration=100,
            # hyper_parameter_decay_factor=0.9,
            # hyper_parameter_maximum_decay=10,
            # hyper_parameter_converge_threshold=1e-6,

            # model_converge_threshold=1e-6
    ):
        Inferencer.__init__(self, hyper_parameter_optimize_interval)
        self._scipy_optimization_method = scipy_optimization_method

        self._hessian_free_optimization = hessian_free_optimization
        self._diagonal_covariance_matrix = diagonal_covariance_matrix

        self._hessian_direction_approximation_epsilon = hessian_direction_approximation_epsilon

    """
    @param num_topics: the number of topics
    @param data: a defaultdict(dict) data type, first indexed by doc id then indexed by term id
    take note: words are not terms, they are repeatable and thus might be not unique
    """

    def _initialize(
            self, corpus, vocab, number_of_topics, alpha_mu, alpha_sigma,
            alpha_beta):
        Inferencer._initialize(
            self, vocab, number_of_topics, alpha_mu, alpha_sigma, alpha_beta)

        self._corpus = corpus
        self._parsed_corpus = self.parse_data()

        # define the total number of document
        self._number_of_documents = len(self._parsed_corpus[0])

        # initialize a D-by-K matrix gamma
        self._lambda = numpy.zeros(
            (self._number_of_documents, self._number_of_topics))
        self._nu_square = numpy.ones(
            (self._number_of_documents, self._number_of_topics))

        # initialize a V-by-K matrix beta, subject to the sum over every row is 1
        self._eta = numpy.random.gamma(
            100., 1. / 100., (self._number_of_topics, self._number_of_types))

    def parse_data(self, corpus=None):
        if corpus is None:
            corpus = self._corpus

        doc_count = 0

        word_ids = []
        word_cts = []

        for document_line in corpus:
            # words = document_line.split();
            document_word_dict = {}
            for token in document_line.split():
                if token not in self._type_to_index:
                    continue

                type_id = self._type_to_index[token]
                if type_id not in document_word_dict:
                    document_word_dict[type_id] = 0
                document_word_dict[type_id] += 1

            if len(document_word_dict) == 0:
                sys.stderr.write("warning: document collapsed during parsing")
                continue

            word_ids.append(numpy.array(list(document_word_dict.keys())))
            word_cts.append(
                numpy.array(list(
                    document_word_dict.values()))[numpy.newaxis, :])

            doc_count += 1
            if doc_count % 10000 == 0:
                print("successfully parse %d documents..." % doc_count)

        assert len(word_ids) == len(word_cts)
        print("successfully parse %d documents..." % (doc_count))

        return (word_ids, word_cts)

    #
    #
    #
    #
    #

    def e_step_process_queue(
            self,
            parsed_corpus=None,
            number_of_processes=0,
            local_parameter_iteration=10,
            local_parameter_converge_threshold=1e-3,
    ):
        if parsed_corpus == None:
            word_ids = self._parsed_corpus[0]
            word_cts = self._parsed_corpus[1]
        else:
            word_ids = parsed_corpus[0]
            word_cts = parsed_corpus[1]

        assert len(word_ids) == len(word_cts)
        number_of_documents = len(word_ids)

        E_log_eta = compute_dirichlet_expectation(self._eta)
        assert E_log_eta.shape == (
            self._number_of_topics, self._number_of_types)
        # if parsed_corpus!=None:
        # E_log_prob_eta = E_log_eta-scipy.misc.logsumexp(E_log_eta, axis=1)[:, numpy.newaxis]

        task_queue = multiprocessing.JoinableQueue()
        for (doc_id, word_id, word_ct) in zip(list(range(number_of_documents)),
                                              word_ids, word_cts):
            task_queue.put((doc_id, word_id, word_ct))

        result_doc_parameter_queue = multiprocessing.Queue()
        result_log_likelihood_queue = multiprocessing.Queue()
        if parsed_corpus == None:
            result_sufficient_statistics_queue = multiprocessing.Queue()
        else:
            result_sufficient_statistics_queue = None

        if self._diagonal_covariance_matrix:
            e_step_parameters = (E_log_eta, self._alpha_mu, self._alpha_sigma)
        else:
            e_step_parameters = (
                E_log_eta, self._alpha_mu, self._alpha_sigma,
                self._alpha_sigma_inv)

        # start consumers
        if number_of_processes <= 1:
            number_of_processes = multiprocessing.cpu_count()
        print('creating %d processes' % number_of_processes)
        processes_e_step = [
            Process_E_Step_Queue(
                task_queue,
                e_step_parameters,
                self.optimize_doc_lambda,
                # self.optimize_doc_nu_square,
                self.optimize_doc_nu_square_in_log_space,
                result_doc_parameter_queue,
                result_log_likelihood_queue,
                result_sufficient_statistics_queue,
                diagonal_covariance_matrix=self._diagonal_covariance_matrix,
                parameter_iteration=local_parameter_iteration,
            ) for process_index in range(number_of_processes)
        ]

        for process_e_step in processes_e_step:
            process_e_step.start()

        task_queue.join()

        task_queue.close()

        # initialize a D-by-K matrix lambda and nu_square values
        lambda_values = numpy.zeros(
            (number_of_documents,
             self._number_of_topics))  # + self._alpha_mu[numpy.newaxis, :];
        nu_square_values = numpy.zeros(
            (number_of_documents,
             self._number_of_topics))  # + self._alpha_sigma[numpy.newaxis, :];

        # for result_queue_element_index in xrange(result_doc_parameter_queue.qsize()):
        # while not result_doc_parameter_queue.empty():
        for result_queue_element_index in range(number_of_documents):
            (doc_id, doc_lambda,
             doc_nu_square) = result_doc_parameter_queue.get()

            assert doc_id >= 0 and doc_id < number_of_documents
            lambda_values[doc_id, :] = doc_lambda
            nu_square_values[doc_id, :] = doc_nu_square

        log_likelihood = 0
        # for result_queue_element_index in result_log_likelihood_queue.qsize():
        # while not result_log_likelihood_queue.empty():
        for result_queue_element_index in range(number_of_processes):
            log_likelihood += result_log_likelihood_queue.get()
            # print "log_likelihood is", log_likelihood;

        if parsed_corpus == None:
            self._lambda = lambda_values
            self._nu_square = nu_square_values

            # initialize a K-by-V matrix phi sufficient statistics
            phi_sufficient_statistics = numpy.zeros(
                (self._number_of_topics, self._number_of_types))

            # for result_queue_element_index in xrange(result_sufficient_statistics_queue.qsize()):
            # while not result_sufficient_statistics_queue.empty():
            for result_queue_element_index in range(number_of_processes):
                phi_sufficient_statistics += result_sufficient_statistics_queue.get(
                )
                # print "phi_sufficient_statistics", phi_sufficient_statistics

        for process_e_step in processes_e_step:
            process_e_step.join()

        if parsed_corpus == None:
            return log_likelihood, phi_sufficient_statistics
        else:
            return log_likelihood, lambda_values, nu_square_values
        '''
        if parsed_corpus==None:
            document_log_likelihood, lambda_values, nu_square_values, phi_sufficient_statistics = self.format_result_queues(number_of_documents,
                                                                                                                            result_doc_parameter_queue,
                                                                                                                            result_log_likelihood_queue,
                                                                                                                            result_sufficient_statistics_queue
                                                                                                                            );
                                                                                                                   
            self._lambda = lambda_values;
            self._nu_square = nu_square_values;
        
            return document_log_likelihood, phi_sufficient_statistics
        else:
            words_log_likelihood, lambda_values, nu_square_values = self.format_result_queues(number_of_documents,
                                                                                              result_doc_parameter_queue,
                                                                                              result_log_likelihood_queue,
                                                                                              );
                                                                                                 
            return words_log_likelihood, lambda_values, nu_square_values
        '''

    def format_result_queues(
            self, number_of_documents, result_doc_parameter_queue,
            result_log_likelihood_queue,
            result_sufficient_statistics_queue=None):
        # initialize a D-by-K matrix lambda and nu_square values
        lambda_values = numpy.zeros(
            (number_of_documents,
             self._number_of_topics))  # + self._alpha_mu[numpy.newaxis, :];
        nu_square_values = numpy.zeros(
            (number_of_documents,
             self._number_of_topics))  # + self._alpha_sigma[numpy.newaxis, :];

        counter = 0
        # for result_queue_element_index in xrange(result_doc_parameter_queue.qsize()):
        while not result_doc_parameter_queue.empty():
            (doc_id, doc_lambda,
             doc_nu_square) = result_doc_parameter_queue.get()

            assert doc_id >= 0 and doc_id < number_of_documents
            lambda_values[doc_id, :] = doc_lambda
            nu_square_values[doc_id, :] = doc_nu_square

            counter += 1
        assert counter == number_of_documents, counter

        log_likelihood = 0
        # for result_queue_element_index in result_log_likelihood_queue.qsize():
        while not result_log_likelihood_queue.empty():
            log_likelihood += result_log_likelihood_queue.get()
            # print "log_likelihood is", log_likelihood;

        if result_sufficient_statistics_queue == None:
            return log_likelihood, lambda_values, nu_square_values
        else:
            # initialize a K-by-V matrix phi sufficient statistics
            phi_sufficient_statistics = numpy.zeros(
                (self._number_of_topics, self._number_of_types))

            # for result_queue_element_index in xrange(result_sufficient_statistics_queue.qsize()):
            while not result_sufficient_statistics_queue.empty():
                phi_sufficient_statistics += result_sufficient_statistics_queue.get(
                )
                # print "phi_sufficient_statistics", phi_sufficient_statistics

            return log_likelihood, lambda_values, nu_square_values, phi_sufficient_statistics

    #
    #
    #
    #
    #

    def e_step(
            self,
            parsed_corpus=None,
            local_parameter_iteration=10,
            local_parameter_converge_threshold=1e-3,
    ):
        if parsed_corpus == None:
            word_ids = self._parsed_corpus[0]
            word_cts = self._parsed_corpus[1]
        else:
            word_ids = parsed_corpus[0]
            word_cts = parsed_corpus[1]

        assert len(word_ids) == len(word_cts)
        number_of_documents = len(word_ids)

        E_log_eta = compute_dirichlet_expectation(self._eta)
        assert E_log_eta.shape == (
            self._number_of_topics, self._number_of_types)
        if parsed_corpus != None:
            E_log_prob_eta = E_log_eta - scipy.misc.logsumexp(
                E_log_eta, axis=1)[:, numpy.newaxis]

        document_log_likelihood = 0
        words_log_likelihood = 0

        # initialize a V_matrix-by-K matrix phi sufficient statistics
        phi_sufficient_statistics = numpy.zeros(
            (self._number_of_topics, self._number_of_types))

        # initialize a D-by-K matrix lambda and nu_square values
        lambda_values = numpy.zeros(
            (number_of_documents,
             self._number_of_topics))  # + self._alpha_mu[numpy.newaxis, :];
        nu_square_values = numpy.ones(
            (number_of_documents,
             self._number_of_topics))  # + self._alpha_sigma[numpy.newaxis, :];

        # iterate over all documents
        for doc_id in numpy.random.permutation(number_of_documents):
            # initialize gamma for this document
            doc_lambda = lambda_values[doc_id, :]
            doc_nu_square = nu_square_values[doc_id, :]
            '''
            if self._diagonal_covariance_matrix:
                doc_lambda = numpy.random.multivariate_normal(self._alpha_mu, numpy.diag(self._alpha_sigma));
                doc_nu_square = numpy.copy(self._alpha_sigma);
            else:
                #doc_lambda = numpy.random.multivariate_normal(self._alpha_mu[0, :], self._alpha_sigma);
                #doc_nu_square = numpy.copy(numpy.diag(self._alpha_sigma));
                doc_lambda = numpy.random.multivariate_normal(numpy.zeros(self._number_of_topics), numpy.eye(self._number_of_topics))
                doc_nu_square = numpy.ones(self._number_of_topics)
            assert doc_lambda.shape==(self._number_of_topics,)
            assert doc_nu_square.shape==(self._number_of_topics,)
            '''

            term_ids = word_ids[doc_id]
            term_counts = word_cts[doc_id]
            assert term_counts.shape == (1, len(term_ids))
            # compute the total number of words
            doc_word_count = numpy.sum(word_cts[doc_id])

            # update zeta in close form
            # doc_zeta = numpy.sum(numpy.exp(doc_lambda+0.5*doc_nu_square));
            doc_zeta_factor = doc_lambda + 0.5 * doc_nu_square
            assert doc_zeta_factor.shape == (self._number_of_topics, )
            doc_zeta_factor = numpy.tile(
                doc_zeta_factor, (self._number_of_topics, 1))
            assert doc_zeta_factor.shape == (
                self._number_of_topics, self._number_of_topics)

            for local_parameter_iteration_index in range(
                    local_parameter_iteration):
                # update phi in close form
                assert E_log_eta.shape == (
                    self._number_of_topics, self._number_of_types)
                log_phi = E_log_eta[:, term_ids] + doc_lambda[:, numpy.newaxis]
                assert log_phi.shape == (self._number_of_topics, len(term_ids))
                log_phi -= scipy.misc.logsumexp(log_phi,
                                                axis=0)[numpy.newaxis, :]
                assert log_phi.shape == (self._number_of_topics, len(term_ids))

                #
                #
                #
                #
                #

                # update lambda
                sum_phi = numpy.exp(
                    scipy.misc.logsumexp(
                        log_phi + numpy.log(term_counts), axis=1))
                arguments = (
                    doc_nu_square, doc_zeta_factor, sum_phi, doc_word_count)
                doc_lambda = self.optimize_doc_lambda(doc_lambda, arguments)
                '''
                if self._hessian_free_optimization:
                    assert not self._diagonal_covariance_matrix
                    doc_lambda = self.hessian_free_lambda(doc_lambda, doc_nu_square, doc_zeta_factor, sum_phi, doc_word_count);
                else:
                    doc_lambda = self.newton_method_lambda(doc_lambda, doc_nu_square, doc_zeta_factor, sum_phi, doc_word_count);
                '''
                # print "update lambda of doc %d to %s" % (doc_id, doc_lambda)

                #
                #
                #
                #
                #

                # update zeta in close form
                # doc_zeta = numpy.sum(numpy.exp(doc_lambda+0.5*doc_nu_square));
                doc_zeta_factor = doc_lambda + 0.5 * doc_nu_square
                assert doc_zeta_factor.shape == (self._number_of_topics, )
                doc_zeta_factor = numpy.tile(
                    doc_zeta_factor, (self._number_of_topics, 1))
                assert doc_zeta_factor.shape == (
                    self._number_of_topics, self._number_of_topics)

                #
                #
                #
                #
                #

                # update nu_square
                arguments = (doc_lambda, doc_zeta_factor, doc_word_count)
                # doc_nu_square = self.optimize_doc_nu_square(doc_nu_square, arguments);
                doc_nu_square = self.optimize_doc_nu_square_in_log_space(
                    doc_nu_square, arguments)
                '''
                if self._hessian_free_optimization:
                    assert not self._diagonal_covariance_matrix
                    #doc_nu_square = self.hessian_free_nu_square(doc_lambda, doc_nu_square, doc_zeta_factor, doc_word_count);
                    doc_nu_square = self.hessian_free_nu_square_in_log_space(doc_lambda, doc_nu_square, doc_zeta_factor, doc_word_count);
                else:
                    #doc_nu_square = self.newton_method_nu_square(doc_lambda, doc_nu_square, doc_zeta_factor, doc_word_count);
                    doc_nu_square = self.newton_method_nu_square_in_log_space(doc_lambda, doc_nu_square, doc_zeta_factor, doc_word_count);
                '''
                # print "update nu of doc %d to %s" % (doc_id, doc_nu_square)

                #
                #
                #
                #
                #

                # update zeta in close form
                # doc_zeta = numpy.sum(numpy.exp(doc_lambda+0.5*doc_nu_square));
                doc_zeta_factor = doc_lambda + 0.5 * doc_nu_square
                assert doc_zeta_factor.shape == (self._number_of_topics, )
                doc_zeta_factor = numpy.tile(
                    doc_zeta_factor, (self._number_of_topics, 1))
                assert doc_zeta_factor.shape == (
                    self._number_of_topics, self._number_of_topics)

                # mean_change = numpy.mean(abs(gamma_update - lambda_values[doc_id, :]));
                # lambda_values[doc_id, :] = gamma_update;
                # if mean_change <= local_parameter_converge_threshold:
                # break;

                # print doc_id, local_parameter_iteration_index

            # print "process document %d..." % doc_id

            # document_log_likelihood -= 0.5 * self._number_of_topics * numpy.log(2 * numpy.pi)
            if self._diagonal_covariance_matrix:
                document_log_likelihood -= 0.5 * numpy.sum(
                    numpy.log(self._alpha_sigma))
                document_log_likelihood -= 0.5 * numpy.sum(
                    doc_nu_square / self._alpha_sigma)
                document_log_likelihood -= 0.5 * numpy.sum(
                    (doc_lambda - self._alpha_mu)**2 / self._alpha_sigma)
            else:
                # document_log_likelihood -= 0.5 * numpy.log(numpy.linalg.det(self._alpha_sigma));
                document_log_likelihood -= 0.5 * numpy.log(
                    scipy.linalg.det(self._alpha_sigma) + 1e-30)
                document_log_likelihood -= 0.5 * numpy.sum(
                    doc_nu_square * numpy.diag(self._alpha_sigma_inv))
                document_log_likelihood -= 0.5 * numpy.dot(
                    numpy.dot(
                        (self._alpha_mu - doc_lambda[numpy.newaxis, :]),
                        self._alpha_sigma_inv),
                    (self._alpha_mu - doc_lambda[numpy.newaxis, :]).T)

            document_log_likelihood += numpy.sum(
                numpy.sum(numpy.exp(log_phi) * term_counts, axis=1) *
                doc_lambda)
            # use the fact that doc_zeta = numpy.sum(numpy.exp(doc_lambda+0.5*doc_nu_square)), to cancel the factors
            document_log_likelihood -= scipy.misc.logsumexp(
                doc_lambda + 0.5 * doc_nu_square) * doc_word_count

            document_log_likelihood += 0.5 * self._number_of_topics
            # document_log_likelihood += 0.5 * self._number_of_topics * numpy.log(2 * numpy.pi)
            document_log_likelihood += 0.5 * numpy.sum(
                numpy.log(doc_nu_square))

            document_log_likelihood -= numpy.sum(
                numpy.exp(log_phi) * log_phi * term_counts)

            # Note: all terms including E_q[p(\eta | \beta)], i.e., terms involving \Psi(\eta), are cancelled due to \eta updates in M-step
            if parsed_corpus != None:
                # compute the phi terms
                words_log_likelihood += numpy.sum(
                    numpy.exp(log_phi + numpy.log(term_counts)) *
                    E_log_prob_eta[:, term_ids])

            lambda_values[doc_id, :] = doc_lambda
            nu_square_values[doc_id, :] = doc_nu_square

            assert log_phi.shape == (self._number_of_topics, len(term_ids))
            assert term_counts.shape == (1, len(term_ids))
            phi_sufficient_statistics[:, term_ids] += numpy.exp(
                log_phi + numpy.log(term_counts))

            if (doc_id + 1) % 1000 == 0:
                print("successfully processed %d documents..." % (doc_id + 1))

        assert numpy.all(nu_square_values > 0)

        if parsed_corpus == None:
            self._lambda = lambda_values
            self._nu_square = nu_square_values
            return document_log_likelihood, phi_sufficient_statistics
        else:
            return words_log_likelihood, lambda_values, nu_square_values

    #
    #
    #
    #
    #

    def optimize_doc_lambda(self, doc_lambda, arguments):

        optimize_result = scipy.optimize.minimize(
            self.f_doc_lambda,
            doc_lambda,
            args=arguments,
            method=self._scipy_optimization_method,
            jac=self.f_prime_doc_lambda,
            hess=self.f_hessian_doc_lambda,
            # hess=None,
            hessp=self.f_hessian_direction_doc_lambda,
            bounds=None,
            constraints=(),
            tol=None,
            callback=None,
            options={'disp': False})

        return optimize_result.x

    def f_doc_lambda(self, doc_lambda, *args):
        (doc_nu_square, doc_zeta_factor, sum_phi, total_word_count) = args

        assert doc_zeta_factor.shape == (
            self._number_of_topics, self._number_of_topics)
        assert sum_phi.shape == (self._number_of_topics, )
        # if doc_lambda.shape==(1, self._number_of_topics):
        # doc_lambda = doc_lambda[0, :];
        assert doc_lambda.shape == (self._number_of_topics, )

        exp_over_doc_zeta = scipy.misc.logsumexp(
            doc_zeta_factor - doc_lambda[:, numpy.newaxis] -
            0.5 * doc_nu_square[:, numpy.newaxis], axis=1)
        exp_over_doc_zeta = numpy.exp(-exp_over_doc_zeta)

        function_doc_lambda = numpy.sum(sum_phi * doc_lambda)

        if self._diagonal_covariance_matrix:
            mean_adjustment = doc_lambda - self._alpha_mu
            assert mean_adjustment.shape == (self._number_of_topics, )
            function_doc_lambda += -0.5 * numpy.sum(
                (mean_adjustment**2) / self._alpha_sigma)
        else:
            mean_adjustment = doc_lambda[numpy.newaxis, :] - self._alpha_mu
            assert mean_adjustment.shape == (1, self._number_of_topics), (
                doc_lambda.shape, mean_adjustment.shape, self._alpha_mu.shape)
            function_doc_lambda += -0.5 * numpy.dot(
                numpy.dot(mean_adjustment, self._alpha_sigma_inv),
                mean_adjustment.T)

        function_doc_lambda += -total_word_count * numpy.sum(exp_over_doc_zeta)

        return numpy.asscalar(-function_doc_lambda)

    def f_prime_doc_lambda(self, doc_lambda, *args):
        (doc_nu_square, doc_zeta_factor, sum_phi, total_word_count) = args

        assert doc_zeta_factor.shape == (
            self._number_of_topics, self._number_of_topics)
        assert sum_phi.shape == (self._number_of_topics, )

        exp_over_doc_zeta = scipy.misc.logsumexp(
            doc_zeta_factor - doc_lambda[:, numpy.newaxis] -
            0.5 * doc_nu_square[:, numpy.newaxis], axis=1)
        exp_over_doc_zeta = numpy.exp(-exp_over_doc_zeta)
        assert exp_over_doc_zeta.shape == (self._number_of_topics, )

        if self._diagonal_covariance_matrix:
            function_prime_doc_lambda = (
                self._alpha_mu - doc_lambda) / self._alpha_sigma
        else:
            function_prime_doc_lambda = numpy.dot(
                (self._alpha_mu - doc_lambda[numpy.newaxis, :]),
                self._alpha_sigma_inv)[0, :]

        function_prime_doc_lambda += sum_phi
        function_prime_doc_lambda -= total_word_count * exp_over_doc_zeta

        assert function_prime_doc_lambda.shape == (self._number_of_topics, )

        return numpy.asarray(-function_prime_doc_lambda)

    def f_hessian_doc_lambda(self, doc_lambda, *args):
        (doc_nu_square, doc_zeta_factor, sum_phi, total_word_count) = args
        exp_over_doc_zeta = scipy.misc.logsumexp(
            doc_zeta_factor - doc_lambda[:, numpy.newaxis] -
            0.5 * doc_nu_square[:, numpy.newaxis], axis=1)
        exp_over_doc_zeta = numpy.exp(-exp_over_doc_zeta)

        if self._diagonal_covariance_matrix:
            function_hessian_doc_lambda = -1.0 / self._alpha_sigma
            function_hessian_doc_lambda -= total_word_count * exp_over_doc_zeta
        else:
            function_hessian_doc_lambda = -self._alpha_sigma_inv
            assert function_hessian_doc_lambda.shape == (
                self._number_of_topics, self._number_of_topics)
            function_hessian_doc_lambda -= total_word_count * numpy.diag(
                exp_over_doc_zeta)
            assert function_hessian_doc_lambda.shape == (
                self._number_of_topics, self._number_of_topics)

        return numpy.asarray(-function_hessian_doc_lambda)

    def f_hessian_direction_doc_lambda(
            self, doc_lambda, direction_vector, *args):
        (doc_nu_square, doc_zeta_factor, sum_phi, total_word_count) = args

        assert doc_lambda.shape == (self._number_of_topics, )
        assert doc_nu_square.shape == (self._number_of_topics, )
        assert doc_zeta_factor.shape == (
            self._number_of_topics, self._number_of_topics)
        assert direction_vector.shape == (self._number_of_topics, )

        log_exp_over_doc_zeta_a = scipy.misc.logsumexp(
            doc_zeta_factor - doc_lambda[:, numpy.newaxis] -
            direction_vector[:, numpy.newaxis] *
            self._hessian_direction_approximation_epsilon -
            0.5 * doc_nu_square[:, numpy.newaxis], axis=1)
        log_exp_over_doc_zeta_b = scipy.misc.logsumexp(
            doc_zeta_factor - doc_lambda[:, numpy.newaxis] -
            0.5 * doc_nu_square[:, numpy.newaxis], axis=1)
        assert log_exp_over_doc_zeta_a.shape == (self._number_of_topics, )
        assert log_exp_over_doc_zeta_b.shape == (self._number_of_topics, )

        # function_hessian_direction_doc_lambda = total_word_count * numpy.exp(numpy.log(1 - numpy.exp(log_exp_over_doc_zeta_b - log_exp_over_doc_zeta_a)) - log_exp_over_doc_zeta_b)
        function_hessian_direction_doc_lambda = total_word_count * numpy.exp(
            -log_exp_over_doc_zeta_b
        ) * (1 - numpy.exp(log_exp_over_doc_zeta_b - log_exp_over_doc_zeta_a))

        if self._diagonal_covariance_matrix:
            function_hessian_direction_doc_lambda += -direction_vector * self._hessian_direction_approximation_epsilon / self._alpha_sigma
        else:
            function_hessian_direction_doc_lambda += -numpy.dot(
                direction_vector[numpy.newaxis, :] *
                self._hessian_direction_approximation_epsilon,
                self._alpha_sigma_inv)[0, :]
        assert function_hessian_direction_doc_lambda.shape == (
            self._number_of_topics, )

        function_hessian_direction_doc_lambda /= self._hessian_direction_approximation_epsilon

        return numpy.asarray(-function_hessian_direction_doc_lambda)

    #
    #
    #
    #
    #

    def optimize_doc_nu_square(self, doc_nu_square, arguments):
        variable_bounds = tuple([(0, None)] * self._number_of_topics)

        optimize_result = scipy.optimize.minimize(
            self.f_doc_nu_square,
            doc_nu_square,
            args=arguments,
            method=self._scipy_optimization_method,
            jac=self.f_prime_doc_nu_square,
            hess=self.f_hessian_doc_nu_square,
            # hess=None,
            hessp=self.f_hessian_direction_doc_nu_square,
            bounds=variable_bounds,
            constraints=(),
            tol=None,
            callback=None,
            options={'disp': False})

        return optimize_result.x

    def f_doc_nu_square(self, doc_nu_square, *args):
        (doc_lambda, doc_zeta_factor, total_word_count) = args

        assert doc_zeta_factor.shape == (
            self._number_of_topics, self._number_of_topics)

        exp_over_doc_zeta = scipy.misc.logsumexp(
            doc_zeta_factor - doc_lambda[:, numpy.newaxis] -
            0.5 * doc_nu_square[:, numpy.newaxis], axis=1)
        exp_over_doc_zeta = numpy.exp(-exp_over_doc_zeta)

        function_doc_nu_square = 0.5 * numpy.sum(numpy.log(doc_nu_square))

        if self._diagonal_covariance_matrix:
            function_doc_nu_square += -0.5 * numpy.sum(
                doc_nu_square / self._alpha_sigma)
        else:
            function_doc_nu_square += -0.5 * numpy.sum(
                doc_nu_square * numpy.diag(self._alpha_sigma_inv))

        function_doc_nu_square += -total_word_count * numpy.sum(
            exp_over_doc_zeta)

        return numpy.asscalar(-function_doc_nu_square)

    def f_prime_doc_nu_square(self, doc_nu_square, *args):
        (doc_lambda, doc_zeta_factor, total_word_count) = args

        assert doc_zeta_factor.shape == (
            self._number_of_topics, self._number_of_topics)

        exp_over_doc_zeta = scipy.misc.logsumexp(
            doc_zeta_factor - doc_lambda[:, numpy.newaxis] -
            0.5 * doc_nu_square[:, numpy.newaxis], axis=1)
        exp_over_doc_zeta = numpy.exp(-exp_over_doc_zeta)

        if self._diagonal_covariance_matrix:
            function_prime_doc_nu_square = -0.5 / self._alpha_sigma
        else:
            function_prime_doc_nu_square = -0.5 * numpy.diag(
                self._alpha_sigma_inv)
        function_prime_doc_nu_square += 0.5 / doc_nu_square
        function_prime_doc_nu_square -= 0.5 * total_word_count * exp_over_doc_zeta

        return numpy.asarray(-function_prime_doc_nu_square)

    def f_hessian_doc_nu_square(self, doc_nu_square, *args):
        (doc_lambda, doc_zeta_factor, total_word_count) = args

        assert doc_zeta_factor.shape == (
            self._number_of_topics, self._number_of_topics)

        exp_over_doc_zeta = scipy.misc.logsumexp(
            doc_zeta_factor - doc_lambda[:, numpy.newaxis] -
            0.5 * doc_nu_square[:, numpy.newaxis], axis=1)
        exp_over_doc_zeta = numpy.exp(-exp_over_doc_zeta)

        function_hessian_doc_nu_square = -0.5 / (doc_nu_square**2)
        function_hessian_doc_nu_square += -0.25 * total_word_count * exp_over_doc_zeta

        function_hessian_doc_nu_square = numpy.diag(
            function_hessian_doc_nu_square)

        assert function_hessian_doc_nu_square.shape == (
            self._number_of_topics, self._number_of_topics)

        return numpy.asarray(-function_hessian_doc_nu_square)

    def f_hessian_direction_doc_nu_square(
            self, doc_nu_square, direction_vector, *args):
        (doc_lambda, doc_zeta_factor, total_word_count) = args

        assert direction_vector.shape == (self._number_of_topics, )

        # assert doc_lambda.shape==(self._number_of_topics,)
        # assert doc_nu_square.shape==(self._number_of_topics,)
        assert doc_zeta_factor.shape == (
            self._number_of_topics, self._number_of_topics)

        log_exp_over_doc_zeta_a = scipy.misc.logsumexp(
            doc_zeta_factor - doc_lambda[:, numpy.newaxis] - 0.5 * (
                doc_nu_square[:, numpy.newaxis] +
                direction_vector[:, numpy.newaxis] *
                self._hessian_direction_approximation_epsilon), axis=1)
        log_exp_over_doc_zeta_b = scipy.misc.logsumexp(
            doc_zeta_factor - doc_lambda[:, numpy.newaxis] -
            0.5 * doc_nu_square[:, numpy.newaxis], axis=1)

        # function_hessian_direction_doc_nu_square = total_word_count * numpy.exp(numpy.log(1 - numpy.exp(log_exp_over_doc_zeta_b - log_exp_over_doc_zeta_a)) - log_exp_over_doc_zeta_b)
        function_hessian_direction_doc_nu_square = total_word_count * numpy.exp(
            -log_exp_over_doc_zeta_b
        ) * (1 - numpy.exp(log_exp_over_doc_zeta_b - log_exp_over_doc_zeta_a))

        function_hessian_direction_doc_nu_square += 0.5 / (
            doc_nu_square +
            self._hessian_direction_approximation_epsilon * direction_vector)
        function_hessian_direction_doc_nu_square -= 0.5 / (doc_nu_square)

        function_hessian_direction_doc_nu_square /= self._hessian_direction_approximation_epsilon

        assert function_hessian_direction_doc_nu_square.shape == (
            self._number_of_topics, )

        return numpy.asarray(-function_hessian_direction_doc_nu_square)

    #
    #
    #
    #
    #

    def optimize_doc_nu_square_in_log_space(
            self, doc_nu_square, arguments, method_name=None):
        log_doc_nu_square = numpy.log(doc_nu_square)

        optimize_result = scipy.optimize.minimize(
            self.f_log_doc_nu_square,
            log_doc_nu_square,
            args=arguments,
            method=method_name,
            jac=self.f_prime_log_doc_nu_square,
            hess=self.f_hessian_log_doc_nu_square,
            # hess=None,
            hessp=self.f_hessian_direction_log_doc_nu_square,
            bounds=None,
            constraints=(),
            tol=None,
            callback=None,
            options={'disp': False})

        log_doc_nu_square_update = optimize_result.x

        return numpy.exp(log_doc_nu_square_update)

    def f_log_doc_nu_square(self, log_doc_nu_square, *args):
        return self.f_doc_nu_square(numpy.exp(log_doc_nu_square), *args)

    def f_prime_log_doc_nu_square(self, log_doc_nu_square, *args):
        (doc_lambda, doc_zeta_factor, total_word_count) = args

        assert log_doc_nu_square.shape == (self._number_of_topics, )
        assert doc_zeta_factor.shape == (
            self._number_of_topics, self._number_of_topics)

        exp_log_doc_nu_square = numpy.exp(log_doc_nu_square)

        exp_over_doc_zeta = scipy.misc.logsumexp(
            doc_zeta_factor - doc_lambda[:, numpy.newaxis] -
            0.5 * exp_log_doc_nu_square[:, numpy.newaxis], axis=1)
        exp_over_doc_zeta = numpy.exp(-exp_over_doc_zeta)

        if self._diagonal_covariance_matrix:
            function_prime_log_doc_nu_square = -0.5 * exp_log_doc_nu_square / self._alpha_sigma
        else:
            function_prime_log_doc_nu_square = -0.5 * exp_log_doc_nu_square * numpy.diag(
                self._alpha_sigma_inv)
        function_prime_log_doc_nu_square += 0.5
        function_prime_log_doc_nu_square -= 0.5 * total_word_count * exp_over_doc_zeta * exp_log_doc_nu_square

        assert function_prime_log_doc_nu_square.shape == (
            self._number_of_topics, )

        return numpy.asarray(-function_prime_log_doc_nu_square)

    def f_hessian_log_doc_nu_square(self, log_doc_nu_square, *args):
        (doc_lambda, doc_zeta_factor, total_word_count) = args

        assert log_doc_nu_square.shape == (self._number_of_topics, )
        assert doc_zeta_factor.shape == (
            self._number_of_topics, self._number_of_topics)

        exp_doc_log_nu_square = numpy.exp(log_doc_nu_square)

        exp_over_doc_zeta = scipy.misc.logsumexp(
            doc_zeta_factor - doc_lambda[:, numpy.newaxis] -
            0.5 * exp_doc_log_nu_square[:, numpy.newaxis], axis=1)
        exp_over_doc_zeta = numpy.exp(-exp_over_doc_zeta)

        if self._diagonal_covariance_matrix:
            function_hessian_log_doc_nu_square = -0.5 * exp_doc_log_nu_square / self._alpha_sigma
        else:
            function_hessian_log_doc_nu_square = -0.5 * exp_doc_log_nu_square * numpy.diag(
                self._alpha_sigma_inv)
        function_hessian_log_doc_nu_square -= 0.5 * total_word_count * exp_over_doc_zeta * exp_doc_log_nu_square * (
            1 + 0.5 * exp_doc_log_nu_square)

        function_hessian_log_doc_nu_square = numpy.diag(
            function_hessian_log_doc_nu_square)

        assert function_hessian_log_doc_nu_square.shape == (
            self._number_of_topics, self._number_of_topics)

        return numpy.asarray(-function_hessian_log_doc_nu_square)

    def f_hessian_direction_log_doc_nu_square(
            self, log_doc_nu_square, direction_vector, *args):
        (doc_lambda, doc_zeta_factor, total_word_count) = args

        # assert doc_lambda.shape==(self._number_of_topics,)
        assert log_doc_nu_square.shape == (self._number_of_topics, )
        assert direction_vector.shape == (self._number_of_topics, )

        assert doc_zeta_factor.shape == (
            self._number_of_topics, self._number_of_topics)

        exp_log_doc_nu_square = numpy.exp(log_doc_nu_square)
        exp_log_doc_nu_square_epsilon_direction = numpy.exp(
            log_doc_nu_square +
            direction_vector * self._hessian_direction_approximation_epsilon)

        log_exp_over_doc_zeta_epsilon_direction = scipy.misc.logsumexp(
            doc_zeta_factor - doc_lambda[:, numpy.newaxis] -
            0.5 * exp_log_doc_nu_square_epsilon_direction[:, numpy.newaxis],
            axis=1)
        log_exp_over_doc_zeta = scipy.misc.logsumexp(
            doc_zeta_factor - doc_lambda[:, numpy.newaxis] -
            0.5 * exp_log_doc_nu_square[:, numpy.newaxis], axis=1)

        # function_hessian_direction_log_doc_nu_square = 0.5 * total_word_count * numpy.exp(log_doc_nu_square - log_exp_over_doc_zeta)
        # function_hessian_direction_log_doc_nu_square += - 0.5 * total_word_count * numpy.exp(log_doc_nu_square + direction_vector * epsilon - log_exp_over_doc_zeta_epsilon_direction)

        function_hessian_direction_log_doc_nu_square = 1 - numpy.exp(
            direction_vector * self._hessian_direction_approximation_epsilon -
            log_exp_over_doc_zeta_epsilon_direction + log_exp_over_doc_zeta)
        function_hessian_direction_log_doc_nu_square *= 0.5 * total_word_count * numpy.exp(
            log_doc_nu_square - log_exp_over_doc_zeta)

        if self._diagonal_covariance_matrix:
            function_hessian_direction_log_doc_nu_square += 0.5 * (
                exp_log_doc_nu_square -
                exp_log_doc_nu_square_epsilon_direction) / self._alpha_sigma
        else:
            function_hessian_direction_log_doc_nu_square += 0.5 * (
                exp_log_doc_nu_square -
                exp_log_doc_nu_square_epsilon_direction) * numpy.diag(
                    self._alpha_sigma_inv)

        function_hessian_direction_log_doc_nu_square /= self._hessian_direction_approximation_epsilon

        assert function_hessian_direction_log_doc_nu_square.shape == (
            self._number_of_topics, )

        return numpy.asarray(-function_hessian_direction_log_doc_nu_square)

    #
    #
    #
    #
    #

    def m_step(self, phi_sufficient_statistics):
        # Note: all terms including E_q[p(\eta|\beta)], i.e., terms involving \Psi(\eta), are cancelled due to \eta updates

        # compute the beta terms
        topic_log_likelihood = self._number_of_topics * (
            scipy.special.gammaln(numpy.sum(self._alpha_beta)) -
            numpy.sum(scipy.special.gammaln(self._alpha_beta)))
        # compute the eta terms
        topic_log_likelihood += numpy.sum(
            numpy.sum(scipy.special.gammaln(self._eta), axis=1) -
            scipy.special.gammaln(numpy.sum(self._eta, axis=1)))

        self._eta = phi_sufficient_statistics + self._alpha_beta
        assert (
            self._eta.shape == (self._number_of_topics, self._number_of_types))

        return topic_log_likelihood

    """
    """

    def learning(self, number_of_processes=1):
        self._counter += 1

        clock_e_step = time.time()
        if number_of_processes == 1:
            document_log_likelihood, phi_sufficient_statistics = self.e_step()
        else:
            document_log_likelihood, phi_sufficient_statistics = self.e_step_process_queue(
                None, number_of_processes)
        clock_e_step = time.time() - clock_e_step

        clock_m_step = time.time()
        topic_log_likelihood = self.m_step(phi_sufficient_statistics)
        clock_m_step = time.time() - clock_m_step

        print(document_log_likelihood, topic_log_likelihood)
        joint_log_likelihood = document_log_likelihood + topic_log_likelihood

        print(
            "e_step and m_step of iteration %d finished in %g and %g seconds respectively with log likelihood %g"
            %
            (self._counter, clock_e_step, clock_m_step, joint_log_likelihood))

        clock_hyper_opt = time.time()
        if self._hyper_parameter_optimize_interval > 0 and self._counter % self._hyper_parameter_optimize_interval == 0:
            self.optimize_hyperparameter()
        clock_hyper_opt = time.time() - clock_hyper_opt
        print(
            "hyper-parameter optimization of iteration %d finished in %g seconds"
            % (self._counter, clock_hyper_opt))

        # if abs((joint_log_likelihood - old_likelihood) / old_likelihood) < self._model_converge_threshold:
        # print "model likelihood converged..."
        # break
        # old_likelihood = joint_log_likelihood;

        return joint_log_likelihood

    def inference(self, corpus):
        parsed_corpus = self.parse_data(corpus)
        number_of_documents = len(parsed_corpus[0])

        clock_e_step = time.time()
        document_log_likelihood, lambda_values, nu_square_values = self.e_step(
            parsed_corpus)
        clock_e_step = time.time() - clock_e_step

        return document_log_likelihood, lambda_values, nu_square_values

    def optimize_hyperparameter(self):
        assert self._lambda.shape == (
            self._number_of_documents, self._number_of_topics)
        self._alpha_mu = numpy.mean(self._lambda, axis=0)
        print("update hyper-parameter mu to %s" % self._alpha_mu)

        assert self._nu_square.shape == (
            self._number_of_documents, self._number_of_topics)
        if self._diagonal_covariance_matrix:
            self._alpha_sigma = numpy.mean(
                self._nu_square +
                (self._lambda - self._alpha_mu[numpy.newaxis, :])**2, axis=0)
            print("update hyper-parameter sigma to %s" % self._alpha_sigma)
        else:
            self._alpha_mu = self._alpha_mu[numpy.newaxis, :]

            assert self._lambda.shape == (
                self._number_of_documents, self._number_of_topics)
            self._alpha_sigma = numpy.copy(
                numpy.diag(numpy.mean(self._nu_square, axis=0)))
            adjusted_lambda = self._lambda - self._alpha_mu
            assert adjusted_lambda.shape == (
                self._number_of_documents, self._number_of_topics)
            self._alpha_sigma += numpy.dot(
                adjusted_lambda.T, adjusted_lambda) / self._number_of_documents

            # self._alpha_sigma_inv = scipy.linalg.pinv(self._alpha_sigma);
            self._alpha_sigma_inv = scipy.linalg.inv(self._alpha_sigma)
            print("update hyper-parameter sigma to")
            print("%s" % self._alpha_sigma)

        return

    """
    @param alpha_vector: a dict data type represents dirichlet prior, indexed by topic_id
    @param alpha_sufficient_statistics: a dict data type represents alpha sufficient statistics for alpha updating, indexed by topic_id
    """

    def optimize_hyperparameter_old(self):
        assert self._lambda.shape == (
            self._number_of_documents, self._number_of_topics)
        self._alpha_mu = numpy.mean(self._lambda, axis=0)
        print("update hyper-parameter mu to %s" % self._alpha_mu)

        assert self._nu_square.shape == (
            self._number_of_documents, self._number_of_topics)
        if self._diagonal_covariance_matrix:
            self._alpha_sigma = numpy.mean(
                self._nu_square +
                (self._lambda - self._alpha_mu[numpy.newaxis, :])**2, axis=0)
            print("update hyper-parameter sigma to %s" % self._alpha_sigma)
        else:
            self._alpha_mu = self._alpha_mu[numpy.newaxis, :]

            self._alpha_sigma = sklearn.covariance.empirical_covariance(
                self._lambda, assume_centered=True)

            # self._alpha_sigma_inv = scipy.linalg.pinv(self._alpha_sigma);
            self._alpha_sigma_inv = scipy.linalg.inv(self._alpha_sigma)
            print("update hyper-parameter sigma to")
            print("%s" % self._alpha_sigma)

        return

    def export_beta(self, exp_beta_path, top_display=-1):
        output = open(exp_beta_path, 'w')
        E_log_eta = compute_dirichlet_expectation(self._eta)
        for topic_index in range(self._number_of_topics):
            output.write("==========\t%d\t==========\n" % (topic_index))

            beta_probability = numpy.exp(
                E_log_eta[topic_index, :] -
                scipy.misc.logsumexp(E_log_eta[topic_index, :]))

            i = 0
            for type_index in reversed(numpy.argsort(beta_probability)):
                i += 1
                output.write(
                    "%s\t%g\n" % (
                        self._index_to_type[type_index],
                        beta_probability[type_index]))
                if top_display > 0 and i >= top_display:
                    break

        output.close()

    #
    #
    #
    #
    #

    def newton_method_lambda(
            self,
            doc_lambda,
            doc_nu_square,
            doc_zeta_factor,
            sum_phi,
            total_word_count,
            newton_method_iteration=10,
            newton_method_decay_factor=0.9,
            # newton_method_step_size=0.1,
            eigen_value_tolerance=1e-9):
        assert doc_zeta_factor.shape == (
            self._number_of_topics, self._number_of_topics)
        assert sum_phi.shape == (self._number_of_topics, )

        newton_method_power_index = 0
        for newton_method_iteration_index in range(newton_method_iteration):
            exp_over_doc_zeta = scipy.misc.logsumexp(
                doc_zeta_factor - doc_lambda[:, numpy.newaxis] -
                0.5 * doc_nu_square[:, numpy.newaxis], axis=1)
            exp_over_doc_zeta = numpy.exp(-exp_over_doc_zeta)
            assert exp_over_doc_zeta.shape == (self._number_of_topics, )

            if self._diagonal_covariance_matrix:
                first_derivative_lambda = (
                    self._alpha_mu - doc_lambda) / self._alpha_sigma
                first_derivative_lambda += sum_phi
                first_derivative_lambda -= total_word_count * exp_over_doc_zeta
            else:
                first_derivative_lambda = numpy.dot(
                    (self._alpha_mu - doc_lambda[numpy.newaxis, :]),
                    self._alpha_sigma_inv)
                assert first_derivative_lambda.shape == (
                    1, self._number_of_topics)
                first_derivative_lambda += sum_phi[numpy.newaxis, :]
                first_derivative_lambda -= total_word_count * exp_over_doc_zeta[
                    numpy.newaxis, :]
                assert first_derivative_lambda.shape == (
                    1, self._number_of_topics)

            if self._diagonal_covariance_matrix:
                second_derivative_lambda = -1.0 / self._alpha_sigma
                second_derivative_lambda -= total_word_count * exp_over_doc_zeta
            else:
                second_derivative_lambda = -self._alpha_sigma_inv
                assert second_derivative_lambda.shape == (
                    self._number_of_topics, self._number_of_topics)
                second_derivative_lambda -= total_word_count * numpy.diag(
                    exp_over_doc_zeta)
                assert second_derivative_lambda.shape == (
                    self._number_of_topics, self._number_of_topics)

            if self._diagonal_covariance_matrix:
                if not numpy.all(second_derivative_lambda) > 0:
                    sys.stderr.write(
                        "Hessian matrix is not positive definite: %s\n" %
                        second_derivative_lambda)
                    break
            else:
                pass
                '''
                print "%s" % second_derivative_lambda;
                E_vector, V_matrix = scipy.linalg.eigh(second_derivative_lambda);
                while not numpy.all(E_vector>eigen_value_tolerance):
                    second_derivative_lambda += numpy.eye(self._number_of_topics);
                    E_vector, V_matrix = scipy.linalg.eigh(second_derivative_lambda);
                    print "%s" % E_vector
                '''

            if self._diagonal_covariance_matrix:
                step_change = first_derivative_lambda / second_derivative_lambda
            else:
                # step_change = numpy.dot(first_derivative_lambda, numpy.linalg.pinv(second_derivative_lambda))[0, :]
                step_change = numpy.dot(
                    first_derivative_lambda,
                    scipy.linalg.pinv(second_derivative_lambda))[0, :]

            # step_change *= newton_method_step_size;
            step_change /= numpy.sqrt(numpy.sum(step_change**2))

            # if numpy.any(numpy.isnan(step_change)) or numpy.any(numpy.isinf(step_change)):
            # break;

            step_alpha = numpy.power(
                newton_method_decay_factor, newton_method_power_index)

            doc_lambda -= step_alpha * step_change
            assert doc_lambda.shape == (self._number_of_topics, )

            # if numpy.all(numpy.abs(step_change) <= local_parameter_converge_threshold):
            # break;

            # print "update lambda to %s" % (doc_lambda)

        return doc_lambda

    def newton_method_nu_square(
            self,
            doc_lambda,
            doc_nu_square,
            doc_zeta_factor,
            total_word_count,
            newton_method_iteration=10,
            newton_method_decay_factor=0.9,
            # newton_method_step_size=0.1,
            eigen_value_tolerance=1e-9):
        assert doc_zeta_factor.shape == (
            self._number_of_topics, self._number_of_topics)

        newton_method_power_index = 0
        for newton_method_iteration_index in range(newton_method_iteration):
            # print doc_zeta_factor - doc_lambda[:, numpy.newaxis] - 0.5 * doc_nu_square[:, numpy.newaxis]
            # exp_over_doc_zeta = 1.0 / numpy.sum(numpy.exp(doc_zeta_factor - doc_lambda[:, numpy.newaxis] - 0.5 * doc_nu_square[:, numpy.newaxis]), axis=1);
            # print scipy.misc.logsumexp(doc_zeta_factor - doc_lambda[:, numpy.newaxis] - 0.5 * doc_nu_square[:, numpy.newaxis], axis=1)
            # exp_over_doc_zeta = numpy.exp(-scipy.misc.logsumexp(doc_zeta_factor - doc_lambda[:, numpy.newaxis] - 0.5 * doc_nu_square[:, numpy.newaxis], axis=1));
            exp_over_doc_zeta = scipy.misc.logsumexp(
                doc_zeta_factor - doc_lambda[:, numpy.newaxis] -
                0.5 * doc_nu_square[:, numpy.newaxis], axis=1)
            # exp_over_doc_zeta = numpy.clip(exp_over_doc_zeta, -10, +10);
            exp_over_doc_zeta = numpy.exp(-exp_over_doc_zeta)

            if self._diagonal_covariance_matrix:
                first_derivative_nu_square = -0.5 / self._alpha_sigma
            else:
                first_derivative_nu_square = -0.5 * numpy.diag(
                    self._alpha_sigma_inv)
            first_derivative_nu_square += 0.5 / doc_nu_square
            # first_derivative_nu_square -= 0.5 * (total_word_count / doc_zeta) * numpy.exp(doc_lambda+0.5*doc_nu_square)
            first_derivative_nu_square -= 0.5 * total_word_count * exp_over_doc_zeta

            second_derivative_nu_square = -0.5 / (doc_nu_square**2)
            # second_derivative_nu_square += -0.25 * (total_word_count / doc_zeta) * numpy.exp(doc_lambda+0.5*doc_nu_square);
            second_derivative_nu_square += -0.25 * total_word_count * exp_over_doc_zeta

            if self._diagonal_covariance_matrix:
                if not numpy.all(second_derivative_nu_square) > 0:
                    print(
                        "Hessian matrix is not positive definite: ",
                        second_derivative_nu_square)
                    break
            else:
                pass
                '''
                print "%s" % second_derivative_nu_square;
                E_vector, V_matrix = scipy.linalg.eigh(second_derivative_nu_square);
                while not numpy.all(E_vector>eigen_value_tolerance):
                    second_derivative_nu_square += numpy.eye(self._number_of_topics);
                    E_vector, V_matrix = scipy.linalg.eigh(second_derivative_nu_square);
                    print "%s" % E_vector
                '''

            step_change = first_derivative_nu_square / second_derivative_nu_square

            # step_change *= newton_method_step_size;
            step_change /= numpy.sqrt(numpy.sum(step_change**2))

            step_alpha = numpy.power(
                newton_method_decay_factor, newton_method_power_index)
            while numpy.any(doc_nu_square <= step_alpha * step_change):
                newton_method_power_index += 1
                step_alpha = numpy.power(
                    newton_method_decay_factor, newton_method_power_index)

            doc_nu_square -= step_alpha * step_change

            assert numpy.all(doc_nu_square > 0), (
                doc_nu_square, step_change, first_derivative_nu_square,
                second_derivative_nu_square)

        return doc_nu_square

    def newton_method_nu_square_in_log_space(
            self, doc_lambda, doc_nu_square, doc_zeta_factor, total_word_count,
            newton_method_iteration=10, newton_method_decay_factor=0.9,
            newton_method_step_size=0.1, eigen_value_tolerance=1e-9):
        assert doc_zeta_factor.shape == (
            self._number_of_topics, self._number_of_topics)

        doc_log_nu_square = numpy.log(doc_nu_square)
        exp_doc_log_nu_square = numpy.exp(doc_log_nu_square)

        newton_method_power_index = 0
        for newton_method_iteration_index in range(newton_method_iteration):
            log_exp_over_doc_zeta_combine = scipy.misc.logsumexp(
                doc_zeta_factor - doc_lambda[:, numpy.newaxis] -
                0.5 * exp_doc_log_nu_square[:, numpy.newaxis] -
                doc_log_nu_square[:, numpy.newaxis], axis=1)
            exp_over_doc_zeta_combine = numpy.exp(
                -log_exp_over_doc_zeta_combine)

            if self._diagonal_covariance_matrix:
                first_derivative_log_nu_square = -0.5 / self._alpha_sigma * exp_doc_log_nu_square
            else:
                first_derivative_log_nu_square = -0.5 * numpy.diag(
                    self._alpha_sigma_inv) * exp_doc_log_nu_square
            first_derivative_log_nu_square += 0.5
            first_derivative_log_nu_square += -0.5 * total_word_count * exp_over_doc_zeta_combine

            if self._diagonal_covariance_matrix:
                second_derivative_log_nu_square = -0.5 / self._alpha_sigma * exp_doc_log_nu_square
            else:
                second_derivative_log_nu_square = -0.5 * numpy.diag(
                    self._alpha_sigma_inv) * exp_doc_log_nu_square
            second_derivative_log_nu_square += -0.5 * total_word_count * exp_over_doc_zeta_combine * (
                1 + 0.5 * exp_doc_log_nu_square)

            step_change = first_derivative_log_nu_square / second_derivative_log_nu_square

            # step_change *= newton_method_step_size;
            step_change /= numpy.sqrt(numpy.sum(step_change**2))

            # if numpy.any(numpy.isnan(step_change)) or numpy.any(numpy.isinf(step_change)):
            # break;

            step_alpha = numpy.power(
                newton_method_decay_factor, newton_method_power_index)

            doc_log_nu_square -= step_alpha * step_change
            exp_doc_log_nu_square = numpy.exp(doc_log_nu_square)

            # if numpy.all(numpy.abs(step_change) <= local_parameter_converge_threshold):
            # break;

            # print "update nu to %s" % (doc_nu_square)

        doc_nu_square = numpy.exp(doc_log_nu_square)

        return doc_nu_square

    #
    #
    #
    #
    #

    def hessian_free_lambda(
            self,
            doc_lambda,
            doc_nu_square,
            doc_zeta_factor,
            sum_phi,
            total_word_count,
            hessian_free_iteration=10,
            hessian_free_threshold=1e-9,
    ):
        for hessian_free_iteration_index in range(hessian_free_iteration):
            delta_doc_lambda = self.conjugate_gradient_delta_lambda(
                doc_lambda, doc_nu_square, doc_zeta_factor, sum_phi,
                total_word_count, self._number_of_topics)

            # delta_doc_lambda /= numpy.sqrt(numpy.sum(delta_doc_lambda**2));

            # print "check point 2", numpy.sum(delta_doc_lambda)
            # print delta_doc_lambda

            doc_lambda += delta_doc_lambda

        return doc_lambda

    def conjugate_gradient_delta_lambda(
            self, doc_lambda, doc_nu_square, doc_zeta_factor, sum_phi,
            total_word_count, conjugate_gradient_iteration=100,
            conjugate_gradient_threshold=1e-9,
            precondition_hessian_matrix=True):
        # delta_doc_lambda = numpy.random.random(self._number_of_topics);
        delta_doc_lambda = numpy.zeros(self._number_of_topics)
        # delta_doc_lambda = numpy.ones(self._number_of_topics);

        if precondition_hessian_matrix:
            hessian_lambda = self.second_derivative_lambda(
                doc_lambda, doc_nu_square, doc_zeta_factor, total_word_count)
            if not numpy.all(numpy.isfinite(hessian_lambda)):
                return numpy.zeros(self._number_of_topics)
            M_inverse = 1.0 / numpy.diag(hessian_lambda)
        # print numpy.linalg.cond(hessian_lambda), ">>>", numpy.linalg.cond(numpy.dot(numpy.diag(1.0/numpy.diag(hessian_lambda)), hessian_lambda)), ">>>", numpy.linalg.cond(numpy.dot(numpy.linalg.cholesky(hessian_lambda), hessian_lambda));

        r_vector = -self.first_derivative_lambda(
            doc_lambda, doc_nu_square, doc_zeta_factor, sum_phi,
            total_word_count)
        r_vector -= self.hessian_damping_direction_approximation_lambda(
            doc_lambda, doc_nu_square, doc_zeta_factor, sum_phi,
            total_word_count, delta_doc_lambda)

        if precondition_hessian_matrix:
            z_vector = M_inverse * r_vector
        else:
            z_vector = numpy.copy(r_vector)

        p_vector = numpy.copy(z_vector)
        r_z_vector_square_old = numpy.sum(r_vector * z_vector)

        for conjugate_gradient_iteration_index in range(
                conjugate_gradient_iteration):
            # hessian_p_vector = self.hessian_direction_approximation_lambda(doc_lambda, doc_nu_square, doc_zeta_factor, total_word_count, p_vector);
            hessian_p_vector = self.hessian_damping_direction_approximation_lambda(
                doc_lambda, doc_nu_square, doc_zeta_factor, sum_phi,
                total_word_count, p_vector)

            alpha_value = r_z_vector_square_old / numpy.sum(
                p_vector * hessian_p_vector)

            delta_doc_lambda += alpha_value * p_vector

            r_vector -= alpha_value * hessian_p_vector

            if numpy.sqrt(numpy.sum(r_vector**
                                    2)) <= conjugate_gradient_threshold:
                break

            if precondition_hessian_matrix:
                z_vector = M_inverse * r_vector
            else:
                z_vector = numpy.copy(r_vector)

            r_z_vector_square_new = numpy.sum(r_vector * z_vector)

            p_vector *= r_z_vector_square_new / r_z_vector_square_old

            p_vector += z_vector

            r_z_vector_square_old = r_z_vector_square_new

        return delta_doc_lambda

    def function_lambda(
            self, doc_lambda, doc_nu_square, doc_zeta_factor, sum_phi,
            total_word_count):
        assert doc_zeta_factor.shape == (
            self._number_of_topics, self._number_of_topics)
        assert sum_phi.shape == (self._number_of_topics, )

        exp_over_doc_zeta = scipy.misc.logsumexp(
            doc_zeta_factor - doc_lambda[:, numpy.newaxis] -
            0.5 * doc_nu_square[:, numpy.newaxis], axis=1)
        exp_over_doc_zeta = numpy.exp(-exp_over_doc_zeta)

        function_lambda = numpy.sum(sum_phi * doc_lambda)

        if self._diagonal_covariance_matrix:
            mean_adjustment = doc_lambda - self._alpha_mu
            assert mean_adjustment.shape == (self._number_of_topics, )
            function_lambda += -0.5 * numpy.sum(
                (mean_adjustment**2) / self._alpha_sigma)
        else:
            mean_adjustment = doc_lambda[numpy.newaxis, :] - self._alpha_mu
            assert mean_adjustment.shape == (1, self._number_of_topics)
            function_lambda += -0.5 * numpy.dot(
                numpy.dot(mean_adjustment, self._alpha_sigma_inv),
                mean_adjustment.T)

        function_lambda += -total_word_count * numpy.sum(exp_over_doc_zeta)

        return function_lambda

    def first_derivative_lambda(
            self, doc_lambda, doc_nu_square, doc_zeta_factor, sum_phi,
            total_word_count):
        assert doc_zeta_factor.shape == (
            self._number_of_topics, self._number_of_topics)
        assert sum_phi.shape == (self._number_of_topics, )

        exp_over_doc_zeta = scipy.misc.logsumexp(
            doc_zeta_factor - doc_lambda[:, numpy.newaxis] -
            0.5 * doc_nu_square[:, numpy.newaxis], axis=1)
        exp_over_doc_zeta = numpy.exp(-exp_over_doc_zeta)
        assert exp_over_doc_zeta.shape == (self._number_of_topics, )

        if self._diagonal_covariance_matrix:
            first_derivative_lambda = (
                self._alpha_mu - doc_lambda) / self._alpha_sigma
        else:
            first_derivative_lambda = numpy.dot(
                (self._alpha_mu - doc_lambda[numpy.newaxis, :]),
                self._alpha_sigma_inv)[0, :]

        first_derivative_lambda += sum_phi
        first_derivative_lambda -= total_word_count * exp_over_doc_zeta
        assert first_derivative_lambda.shape == (self._number_of_topics, )

        return first_derivative_lambda

    def second_derivative_lambda(
            self, doc_lambda, doc_nu_square, doc_zeta_factor,
            total_word_count):
        exp_over_doc_zeta = scipy.misc.logsumexp(
            doc_zeta_factor - doc_lambda[:, numpy.newaxis] -
            0.5 * doc_nu_square[:, numpy.newaxis], axis=1)
        exp_over_doc_zeta = numpy.exp(-exp_over_doc_zeta)

        if self._diagonal_covariance_matrix:
            second_derivative_lambda = -1.0 / self._alpha_sigma
            second_derivative_lambda -= total_word_count * exp_over_doc_zeta
        else:
            second_derivative_lambda = -self._alpha_sigma_inv
            assert second_derivative_lambda.shape == (
                self._number_of_topics, self._number_of_topics)
            second_derivative_lambda -= total_word_count * numpy.diag(
                exp_over_doc_zeta)
            assert second_derivative_lambda.shape == (
                self._number_of_topics, self._number_of_topics)

        return second_derivative_lambda

    def hessian_direction_approximation_lambda(
            self, doc_lambda, doc_nu_square, doc_zeta_factor, total_word_count,
            direction_vector, epsilon=1e-6):
        assert doc_lambda.shape == (self._number_of_topics, )
        assert doc_nu_square.shape == (self._number_of_topics, )
        assert doc_zeta_factor.shape == (
            self._number_of_topics, self._number_of_topics)
        assert direction_vector.shape == (self._number_of_topics, )

        log_exp_over_doc_zeta_a = scipy.misc.logsumexp(
            doc_zeta_factor - doc_lambda[:, numpy.newaxis] -
            direction_vector[:, numpy.newaxis] * epsilon -
            0.5 * doc_nu_square[:, numpy.newaxis], axis=1)
        log_exp_over_doc_zeta_b = scipy.misc.logsumexp(
            doc_zeta_factor - doc_lambda[:, numpy.newaxis] -
            0.5 * doc_nu_square[:, numpy.newaxis], axis=1)
        assert log_exp_over_doc_zeta_a.shape == (self._number_of_topics, )
        assert log_exp_over_doc_zeta_b.shape == (self._number_of_topics, )

        # hessian_direction_lambda = total_word_count * numpy.exp(numpy.log(1 - numpy.exp(log_exp_over_doc_zeta_b - log_exp_over_doc_zeta_a)) - log_exp_over_doc_zeta_b)
        hessian_direction_lambda = total_word_count * numpy.exp(
            -log_exp_over_doc_zeta_b
        ) * (1 - numpy.exp(log_exp_over_doc_zeta_b - log_exp_over_doc_zeta_a))

        if self._diagonal_covariance_matrix:
            hessian_direction_lambda = -direction_vector * epsilon / self._alpha_sigma
        else:
            hessian_direction_lambda += -numpy.dot(
                direction_vector[numpy.newaxis, :] * epsilon,
                self._alpha_sigma_inv)[0, :]
        assert hessian_direction_lambda.shape == (self._number_of_topics, )

        hessian_direction_lambda /= epsilon

        return hessian_direction_lambda

    def hessian_damping_direction_approximation_lambda(
            self, doc_lambda, doc_nu_square, doc_zeta_factor, sum_phi,
            total_word_count, direction_vector,
            damping_factor_initialization=0.1, damping_factor_iteration=10):
        damping_factor_numerator = self.function_lambda(
            doc_lambda + direction_vector, doc_nu_square, doc_zeta_factor,
            sum_phi, total_word_count)
        damping_factor_numerator -= self.function_lambda(
            doc_lambda, doc_nu_square, doc_zeta_factor, sum_phi,
            total_word_count)

        hessian_direction_approximation = self.hessian_direction_approximation_lambda(
            doc_lambda, doc_nu_square, doc_zeta_factor, total_word_count,
            direction_vector)

        damping_factor_denominator_temp = self.first_derivative_lambda(
            doc_lambda, doc_nu_square, doc_zeta_factor, sum_phi,
            total_word_count)
        assert damping_factor_denominator_temp.shape == (
            self._number_of_topics, )
        damping_factor_denominator_temp += 0.5 * hessian_direction_approximation
        assert damping_factor_denominator_temp.shape == (
            self._number_of_topics, )

        damping_factor_lambda = damping_factor_initialization
        for damping_factor_iteration_index in range(damping_factor_iteration):
            damping_factor_denominator = damping_factor_denominator_temp + 0.5 * damping_factor_lambda * direction_vector
            assert damping_factor_denominator.shape == (
                self._number_of_topics, )
            damping_factor_denominator *= direction_vector
            damping_factor_denominator = numpy.sum(damping_factor_denominator)

            damping_factor_rho = damping_factor_numerator / damping_factor_denominator
            if damping_factor_rho < 0.25:
                damping_factor_lambda *= 1.5
            elif damping_factor_rho > 0.75:
                damping_factor_lambda /= 1.5
            else:
                return hessian_direction_approximation + damping_factor_lambda * direction_vector

        return hessian_direction_approximation

    #
    #
    #
    #
    #

    def hessian_free_nu_square(
            self, doc_lambda, doc_nu_square, doc_zeta_factor, total_word_count,
            hessian_free_iteration=10, hessian_free_decay_factor=0.9,
            hessian_free_reset_interval=100):
        for hessian_free_iteration_index in range(hessian_free_iteration):
            delta_doc_nu_square = self.conjugate_gradient_delta_nu_square(
                doc_lambda, doc_nu_square, doc_zeta_factor, total_word_count,
                self._number_of_topics)

            # delta_doc_nu_square /= numpy.sqrt(numpy.sum(delta_doc_nu_square**2));

            conjugate_gradient_power_index = 0
            step_alpha = numpy.power(
                hessian_free_decay_factor, conjugate_gradient_power_index)
            while numpy.any(
                    doc_nu_square + step_alpha * delta_doc_nu_square <= 0):
                conjugate_gradient_power_index += 1
                step_alpha = numpy.power(
                    hessian_free_decay_factor, conjugate_gradient_power_index)
                if conjugate_gradient_power_index >= hessian_free_reset_interval:
                    print("power index larger than 100", delta_doc_nu_square)
                    step_alpha = 0
                    break

            doc_nu_square += step_alpha * delta_doc_nu_square
            assert numpy.all(doc_nu_square > 0)

        return doc_nu_square

    def conjugate_gradient_delta_nu_square(
            self,
            doc_lambda,
            doc_nu_square,
            doc_zeta_factor,
            total_word_count,
            conjugate_gradient_iteration=100,
            conjugate_gradient_threshold=1e-6,
            conjugate_gradient_decay_factor=0.9,
            conjugate_gradient_reset_interval=100,
    ):
        doc_nu_square_copy = numpy.copy(doc_nu_square)
        # delta_doc_nu_square = numpy.ones(self._number_of_topics);
        delta_doc_nu_square = numpy.zeros(self._number_of_topics)
        # delta_doc_nu_square = numpy.random.random(self._number_of_topics);

        r_vector = -self.first_derivative_nu_square(
            doc_lambda, doc_nu_square_copy, doc_zeta_factor, total_word_count)
        # r_vector -= self.hessian_direction_approximation_nu_square(doc_lambda, doc_nu_square_copy, doc_zeta_factor, total_word_count, delta_doc_nu_square, damping_coefficient=1);
        r_vector -= self.hessian_damping_direction_approximation_nu_square(
            doc_lambda, doc_nu_square_copy, doc_zeta_factor, total_word_count,
            delta_doc_nu_square)

        p_vector = numpy.copy(r_vector)

        r_vector_square_old = numpy.sum(r_vector**2)

        for conjugate_gradient_iteration_index in range(
                conjugate_gradient_iteration):
            assert not numpy.any(numpy.isnan(doc_lambda))
            assert not numpy.any(numpy.isnan(doc_nu_square_copy))
            assert not numpy.any(numpy.isnan(doc_zeta_factor))
            assert not numpy.any(numpy.isnan(p_vector))

            # hessian_p_vector = self.hessian_direction_approximation_nu_square(doc_lambda, doc_nu_square_copy, doc_zeta_factor, total_word_count, p_vector, damping_coefficient=1);
            hessian_p_vector = self.hessian_damping_direction_approximation_nu_square(
                doc_lambda, doc_nu_square_copy, doc_zeta_factor,
                total_word_count, p_vector)
            assert not numpy.any(numpy.isnan(hessian_p_vector))

            alpha_value = r_vector_square_old / numpy.sum(
                p_vector * hessian_p_vector)
            assert not numpy.isnan(alpha_value), (
                r_vector_square_old, numpy.sum(p_vector * hessian_p_vector))
            '''
            conjugate_gradient_power_index = 0
            step_alpha = numpy.power(conjugate_gradient_decay_factor, conjugate_gradient_power_index);
            while numpy.any(delta_doc_nu_square <= -step_alpha * alpha_value * p_vector):
                conjugate_gradient_power_index += 1;
                step_alpha = numpy.power(conjugate_gradient_decay_factor, conjugate_gradient_power_index);
                if conjugate_gradient_power_index>=100:
                    print "power index larger than 100", delta_doc_nu_square, alpha_value * p_vector
                    break;

            delta_doc_nu_square += step_alpha * alpha_value * p_vector;
            assert not numpy.any(numpy.isnan(delta_doc_nu_square))
            '''

            # p_vector /= numpy.sqrt(numpy.sum(p_vector**2));

            delta_doc_nu_square += alpha_value * p_vector
            assert not numpy.any(
                numpy.isnan(delta_doc_nu_square)), (alpha_value, p_vector)
            '''
            if conjugate_gradient_iteration_index % conjugate_gradient_reset_interval==0:
                r_vector = -self.first_derivative_nu_square(doc_lambda, doc_nu_square_copy, doc_zeta_factor, total_word_count);
                r_vector -= self.hessian_direction_approximation_nu_square(doc_lambda, doc_nu_square_copy, doc_zeta_factor, total_word_count, delta_doc_nu_square);
            else:
                r_vector -= alpha_value * hessian_p_vector;
            '''
            r_vector -= alpha_value * hessian_p_vector
            assert not numpy.any(numpy.isnan(r_vector))

            r_vector_square_new = numpy.sum(r_vector**2)
            assert not numpy.isnan(r_vector_square_new)

            if numpy.sqrt(r_vector_square_new) <= conjugate_gradient_threshold:
                break

            p_vector *= r_vector_square_new / r_vector_square_old
            assert not numpy.any(numpy.isnan(p_vector))
            p_vector += r_vector
            assert not numpy.any(numpy.isnan(p_vector))

            r_vector_square_old = r_vector_square_new

        return delta_doc_nu_square

    def function_nu_square(
            self, doc_lambda, doc_nu_square, doc_zeta_factor,
            total_word_count):
        assert doc_zeta_factor.shape == (
            self._number_of_topics, self._number_of_topics)

        exp_over_doc_zeta = scipy.misc.logsumexp(
            doc_zeta_factor - doc_lambda[:, numpy.newaxis] -
            0.5 * doc_nu_square[:, numpy.newaxis], axis=1)
        exp_over_doc_zeta = numpy.exp(-exp_over_doc_zeta)

        function_nu_square = 0.5 * numpy.sum(numpy.log(doc_nu_square))

        if self._diagonal_covariance_matrix:
            function_nu_square += -0.5 * numpy.sum(
                doc_nu_square / self._alpha_sigma)
        else:
            function_nu_square += -0.5 * numpy.sum(
                doc_nu_square * numpy.diag(self._alpha_sigma_inv))

        function_nu_square += -total_word_count * numpy.sum(exp_over_doc_zeta)

        return function_nu_square

    def first_derivative_nu_square(
            self, doc_lambda, doc_nu_square, doc_zeta_factor,
            total_word_count):
        assert doc_zeta_factor.shape == (
            self._number_of_topics, self._number_of_topics)

        exp_over_doc_zeta = scipy.misc.logsumexp(
            doc_zeta_factor - doc_lambda[:, numpy.newaxis] -
            0.5 * doc_nu_square[:, numpy.newaxis], axis=1)
        exp_over_doc_zeta = numpy.exp(-exp_over_doc_zeta)

        if self._diagonal_covariance_matrix:
            first_derivative_nu_square = -0.5 / self._alpha_sigma
        else:
            first_derivative_nu_square = -0.5 * numpy.diag(
                self._alpha_sigma_inv)
        first_derivative_nu_square += 0.5 / doc_nu_square
        first_derivative_nu_square -= 0.5 * total_word_count * exp_over_doc_zeta

        return first_derivative_nu_square

    def second_derivative_nu_square(
            self, doc_lambda, doc_nu_square, doc_zeta_factor,
            total_word_count):
        assert doc_zeta_factor.shape == (
            self._number_of_topics, self._number_of_topics)

        exp_over_doc_zeta = scipy.misc.logsumexp(
            doc_zeta_factor - doc_lambda[:, numpy.newaxis] -
            0.5 * doc_nu_square[:, numpy.newaxis], axis=1)
        exp_over_doc_zeta = numpy.exp(-exp_over_doc_zeta)

        second_derivative_nu_square = -0.5 / (doc_nu_square**2)
        second_derivative_nu_square += -0.25 * total_word_count * exp_over_doc_zeta

        return second_derivative_nu_square

    def hessian_direction_approximation_nu_square(
            self, doc_lambda, doc_nu_square, doc_zeta_factor, total_word_count,
            direction_vector, epsilon=1e-6):
        assert doc_lambda.shape == (self._number_of_topics, )
        assert doc_nu_square.shape == (self._number_of_topics, )
        assert doc_zeta_factor.shape == (
            self._number_of_topics, self._number_of_topics)
        assert direction_vector.shape == (self._number_of_topics, )

        log_exp_over_doc_zeta_a = scipy.misc.logsumexp(
            doc_zeta_factor - doc_lambda[:, numpy.newaxis] - 0.5 * (
                doc_nu_square[:, numpy.newaxis] +
                direction_vector[:, numpy.newaxis] * epsilon), axis=1)
        log_exp_over_doc_zeta_b = scipy.misc.logsumexp(
            doc_zeta_factor - doc_lambda[:, numpy.newaxis] -
            0.5 * doc_nu_square[:, numpy.newaxis], axis=1)

        # hessian_direction_nu_square = total_word_count * numpy.exp(numpy.log(1 - numpy.exp(log_exp_over_doc_zeta_b - log_exp_over_doc_zeta_a)) - log_exp_over_doc_zeta_b)
        hessian_direction_nu_square = total_word_count * numpy.exp(
            -log_exp_over_doc_zeta_b
        ) * (1 - numpy.exp(log_exp_over_doc_zeta_b - log_exp_over_doc_zeta_a))

        hessian_direction_nu_square += 0.5 / (
            doc_nu_square + epsilon * direction_vector)
        hessian_direction_nu_square -= 0.5 / (doc_nu_square)

        hessian_direction_nu_square /= epsilon

        return hessian_direction_nu_square

    def hessian_damping_direction_approximation_nu_square(
            self, doc_lambda, doc_nu_square, doc_zeta_factor, total_word_count,
            direction_vector, damping_factor_initialization=0.1,
            damping_factor_iteration=10):
        damping_factor_numerator = self.function_nu_square(
            doc_lambda, doc_nu_square + direction_vector, doc_zeta_factor,
            total_word_count)
        damping_factor_numerator -= self.function_nu_square(
            doc_lambda, doc_nu_square, doc_zeta_factor, total_word_count)

        hessian_direction_approximation = self.hessian_direction_approximation_nu_square(
            doc_lambda, doc_nu_square, doc_zeta_factor, total_word_count,
            direction_vector)

        damping_factor_denominator_temp = self.first_derivative_nu_square(
            doc_lambda, doc_nu_square, doc_zeta_factor, total_word_count)
        assert damping_factor_denominator_temp.shape == (
            self._number_of_topics, )
        damping_factor_denominator_temp += 0.5 * hessian_direction_approximation
        assert damping_factor_denominator_temp.shape == (
            self._number_of_topics, )

        damping_factor_lambda = damping_factor_initialization
        for damping_factor_iteration_index in range(damping_factor_iteration):
            damping_factor_denominator = damping_factor_denominator_temp + 0.5 * damping_factor_lambda * direction_vector
            assert damping_factor_denominator.shape == (
                self._number_of_topics, )
            damping_factor_denominator *= direction_vector
            damping_factor_denominator = numpy.sum(damping_factor_denominator)

            damping_factor_rho = damping_factor_numerator / damping_factor_denominator
            if damping_factor_rho < 0.25:
                damping_factor_lambda *= 1.5
            elif damping_factor_rho > 0.75:
                damping_factor_lambda /= 1.5
            else:
                return hessian_direction_approximation + damping_factor_lambda * direction_vector

        return hessian_direction_approximation

    #
    #
    #
    #
    #

    def hessian_free_nu_square_in_log_space(
            self,
            doc_lambda,
            doc_nu_square,
            doc_zeta_factor,
            total_word_count,
            hessian_free_iteration=10,
            conjugate_gradient_threshold=1e-9,
            conjugate_gradient_reset_interval=100,
    ):
        for hessian_free_iteration_index in range(hessian_free_iteration):
            delta_doc_log_nu_square = self.conjugate_gradient_delta_log_nu_square(
                doc_lambda, doc_nu_square, doc_zeta_factor, total_word_count,
                self._number_of_topics)

            # print "check point 1", numpy.sum(numpy.exp(delta_doc_log_nu_square)**2), numpy.sum(delta_doc_log_nu_square**2)
            # print numpy.sum(numpy.exp(delta_doc_log_nu_square)**2), numpy.exp(delta_doc_log_nu_square);

            # delta_doc_log_nu_square /= numpy.sqrt(numpy.sum(delta_doc_log_nu_square**2));

            doc_nu_square *= numpy.exp(delta_doc_log_nu_square)

        return doc_nu_square

    '''
    nu_square must be greater than 0, conjugate gradient does not perform very well on constrained optimization problem
    update nu_square in log scale, convert the constrained optimization problem to an unconstrained optimization
    '''

    def conjugate_gradient_delta_log_nu_square(
            self,
            doc_lambda,
            doc_nu_square,
            doc_zeta_factor,
            total_word_count,
            conjugate_gradient_iteration=100,
            conjugate_gradient_threshold=1e-9,
            conjugate_gradient_reset_interval=100,
            precondition_hessian_matrix=True,
    ):
        doc_log_nu_square = numpy.log(doc_nu_square)
        # delta_doc_log_nu_square = numpy.random.random(self._number_of_topics);
        delta_doc_log_nu_square = numpy.zeros(self._number_of_topics)
        # delta_doc_log_nu_square = numpy.log(doc_nu_square);

        if precondition_hessian_matrix:
            hessian_log_nu_square = self.second_derivative_log_nu_square(
                doc_lambda, doc_log_nu_square, doc_zeta_factor,
                total_word_count)
            if not numpy.all(numpy.isfinite(hessian_log_nu_square)):
                return numpy.zeros(self._number_of_topics)
            M_inverse = 1.0 / hessian_log_nu_square
        # print numpy.linalg.cond(hessian_log_nu_square), ">>>", numpy.linalg.cond(numpy.dot(numpy.diag(1.0/numpy.diag(hessian_log_nu_square)), hessian_log_nu_square)), ">>>", numpy.linalg.cond(numpy.dot(numpy.linalg.cholesky(hessian_log_nu_square), hessian_log_nu_square));

        r_vector = -self.first_derivative_log_nu_square(
            doc_lambda, doc_log_nu_square, doc_zeta_factor, total_word_count)
        # r_vector -= self.hessian_direction_approximation_log_nu_square(doc_lambda, log_doc_nu_square, doc_zeta_factor, total_word_count, delta_doc_log_nu_square);
        r_vector -= self.hessian_damping_direction_approximation_log_nu_square(
            doc_lambda, doc_log_nu_square, doc_zeta_factor, total_word_count,
            delta_doc_log_nu_square)

        if precondition_hessian_matrix:
            z_vector = M_inverse * r_vector
        else:
            z_vector = numpy.copy(r_vector)

        p_vector = numpy.copy(z_vector)
        r_z_vector_square_old = numpy.sum(r_vector * z_vector)

        for conjugate_gradient_iteration_index in range(
                conjugate_gradient_iteration):
            assert numpy.all(numpy.isfinite(doc_lambda)), (
                conjugate_gradient_iteration_index, doc_lambda)
            assert numpy.all(numpy.isfinite(doc_log_nu_square)), (
                conjugate_gradient_iteration_index, doc_log_nu_square)
            assert numpy.all(numpy.isfinite(doc_zeta_factor)), (
                conjugate_gradient_iteration_index, doc_zeta_factor)
            assert numpy.all(numpy.isfinite(r_vector)), (
                conjugate_gradient_iteration_index, r_vector, doc_nu_square,
                -self.first_derivative_log_nu_square(
                    doc_lambda, doc_log_nu_square, doc_zeta_factor,
                    total_word_count),
                -self.hessian_direction_approximation_log_nu_square(
                    doc_lambda, doc_log_nu_square, doc_zeta_factor,
                    total_word_count, delta_doc_log_nu_square))
            assert numpy.all(numpy.isfinite(p_vector)), (
                conjugate_gradient_iteration_index, p_vector)

            # hessian_p_vector = self.hessian_direction_approximation_log_nu_square(doc_lambda, log_doc_nu_square, doc_zeta_factor, total_word_count, p_vector);
            hessian_p_vector = self.hessian_damping_direction_approximation_log_nu_square(
                doc_lambda, doc_log_nu_square, doc_zeta_factor,
                total_word_count, p_vector)

            alpha_value = r_z_vector_square_old / numpy.sum(
                p_vector * hessian_p_vector)

            delta_doc_log_nu_square += alpha_value * p_vector
            assert not numpy.any(numpy.isnan(delta_doc_log_nu_square))
            '''
            if conjugate_gradient_iteration_index % conjugate_gradient_reset_interval==0:
                r_vector = -self.first_derivative_log_nu_square(doc_lambda, log_doc_nu_square, doc_zeta_factor, total_word_count);
                r_vector -= self.hessian_direction_approximation_log_nu_square(doc_lambda, log_doc_nu_square, doc_zeta_factor, total_word_count, delta_doc_log_nu_square);
            else:
                r_vector -= alpha_value * hessian_p_vector;
            '''
            r_vector -= alpha_value * hessian_p_vector
            assert not numpy.any(numpy.isnan(r_vector)), (
                alpha_value, hessian_p_vector, r_vector)

            if numpy.sqrt(numpy.sum(r_vector**
                                    2)) <= conjugate_gradient_threshold:
                break

            if precondition_hessian_matrix:
                z_vector = M_inverse * r_vector
            else:
                z_vector = numpy.copy(r_vector)

            r_z_vector_square_new = numpy.sum(r_vector * z_vector)

            p_vector *= r_z_vector_square_new / r_z_vector_square_old
            assert not numpy.any(numpy.isnan(p_vector))

            p_vector += z_vector
            assert not numpy.any(numpy.isnan(p_vector))

            r_z_vector_square_old = r_z_vector_square_new

        return delta_doc_log_nu_square

    def function_log_nu_square(
            self, doc_lambda, doc_log_nu_square, doc_zeta_factor,
            total_word_count):
        return self.function_nu_square(
            doc_lambda, numpy.exp(doc_log_nu_square), doc_zeta_factor,
            total_word_count)

    def first_derivative_log_nu_square(
            self, doc_lambda, doc_log_nu_square, doc_zeta_factor,
            total_word_count):
        assert doc_log_nu_square.shape == (self._number_of_topics, )
        assert doc_zeta_factor.shape == (
            self._number_of_topics, self._number_of_topics)

        exp_doc_log_nu_square = numpy.exp(doc_log_nu_square)

        exp_over_doc_zeta = scipy.misc.logsumexp(
            doc_zeta_factor - doc_lambda[:, numpy.newaxis] -
            0.5 * exp_doc_log_nu_square[:, numpy.newaxis], axis=1)
        exp_over_doc_zeta = numpy.exp(-exp_over_doc_zeta)

        if self._diagonal_covariance_matrix:
            first_derivative_log_nu_square = -0.5 * exp_doc_log_nu_square / self._alpha_sigma
        else:
            first_derivative_log_nu_square = -0.5 * exp_doc_log_nu_square * numpy.diag(
                self._alpha_sigma_inv)
        first_derivative_log_nu_square += 0.5
        first_derivative_log_nu_square -= 0.5 * total_word_count * exp_over_doc_zeta * exp_doc_log_nu_square

        return first_derivative_log_nu_square

    def second_derivative_log_nu_square(
            self, doc_lambda, doc_log_nu_square, doc_zeta_factor,
            total_word_count):
        assert doc_log_nu_square.shape == (self._number_of_topics, )
        assert doc_zeta_factor.shape == (
            self._number_of_topics, self._number_of_topics)

        exp_doc_log_nu_square = numpy.exp(doc_log_nu_square)

        exp_over_doc_zeta = scipy.misc.logsumexp(
            doc_zeta_factor - doc_lambda[:, numpy.newaxis] -
            0.5 * exp_doc_log_nu_square[:, numpy.newaxis], axis=1)
        exp_over_doc_zeta = numpy.exp(-exp_over_doc_zeta)

        if self._diagonal_covariance_matrix:
            second_derivative_log_nu_square = -0.5 * exp_doc_log_nu_square / self._alpha_sigma
        else:
            second_derivative_log_nu_square = -0.5 * exp_doc_log_nu_square * numpy.diag(
                self._alpha_sigma_inv)
        second_derivative_log_nu_square -= 0.5 * total_word_count * exp_over_doc_zeta * exp_doc_log_nu_square * (
            1 + 0.5 * exp_doc_log_nu_square)

        return second_derivative_log_nu_square

    def hessian_direction_approximation_log_nu_square(
            self, doc_lambda, doc_log_nu_square, doc_zeta_factor,
            total_word_count, direction_vector, epsilon=1e-6):
        assert doc_lambda.shape == (self._number_of_topics, )
        assert doc_log_nu_square.shape == (self._number_of_topics, )
        assert doc_zeta_factor.shape == (
            self._number_of_topics, self._number_of_topics)
        assert direction_vector.shape == (self._number_of_topics, )

        exp_doc_log_nu_square = numpy.exp(doc_log_nu_square)
        exp_doc_log_nu_square_epsilon_direction = numpy.exp(
            doc_log_nu_square + direction_vector * epsilon)

        log_exp_over_doc_zeta_epsilon_direction = scipy.misc.logsumexp(
            doc_zeta_factor - doc_lambda[:, numpy.newaxis] -
            0.5 * exp_doc_log_nu_square_epsilon_direction[:, numpy.newaxis],
            axis=1)
        log_exp_over_doc_zeta = scipy.misc.logsumexp(
            doc_zeta_factor - doc_lambda[:, numpy.newaxis] -
            0.5 * exp_doc_log_nu_square[:, numpy.newaxis], axis=1)

        # hessian_direction_log_nu_square = 0.5 * total_word_count * numpy.exp(log_doc_nu_square - log_exp_over_doc_zeta)
        # hessian_direction_log_nu_square += - 0.5 * total_word_count * numpy.exp(log_doc_nu_square + direction_vector * epsilon - log_exp_over_doc_zeta_epsilon_direction)

        hessian_direction_log_nu_square = 1 - numpy.exp(
            direction_vector * epsilon -
            log_exp_over_doc_zeta_epsilon_direction + log_exp_over_doc_zeta)
        hessian_direction_log_nu_square *= 0.5 * total_word_count * numpy.exp(
            doc_log_nu_square - log_exp_over_doc_zeta)

        if self._diagonal_covariance_matrix:
            hessian_direction_log_nu_square += 0.5 * (
                exp_doc_log_nu_square -
                exp_doc_log_nu_square_epsilon_direction) / self._alpha_sigma
        else:
            hessian_direction_log_nu_square += 0.5 * (
                exp_doc_log_nu_square -
                exp_doc_log_nu_square_epsilon_direction) * numpy.diag(
                    self._alpha_sigma_inv)

        hessian_direction_log_nu_square /= epsilon

        return hessian_direction_log_nu_square

    def hessian_damping_direction_approximation_log_nu_square(
            self, doc_lambda, doc_log_nu_square, doc_zeta_factor,
            total_word_count, direction_vector,
            damping_factor_initialization=0.1, damping_factor_iteration=10):
        # print "=========="
        # print log_doc_nu_square + direction_vector, numpy.exp(log_doc_nu_square + direction_vector)
        # print log_doc_nu_square, numpy.exp(log_doc_nu_square);

        damping_factor_numerator = self.function_log_nu_square(
            doc_lambda, doc_log_nu_square + direction_vector, doc_zeta_factor,
            total_word_count)
        damping_factor_numerator -= self.function_log_nu_square(
            doc_lambda, doc_log_nu_square, doc_zeta_factor, total_word_count)

        hessian_direction_approximation = self.hessian_direction_approximation_log_nu_square(
            doc_lambda, doc_log_nu_square, doc_zeta_factor, total_word_count,
            direction_vector)

        damping_factor_denominator_temp = self.first_derivative_log_nu_square(
            doc_lambda, doc_log_nu_square, doc_zeta_factor, total_word_count)
        assert damping_factor_denominator_temp.shape == (
            self._number_of_topics, )
        damping_factor_denominator_temp += 0.5 * hessian_direction_approximation
        assert damping_factor_denominator_temp.shape == (
            self._number_of_topics, )

        damping_factor_lambda = damping_factor_initialization
        for damping_factor_iteration_index in range(damping_factor_iteration):
            damping_factor_denominator = damping_factor_denominator_temp + 0.5 * damping_factor_lambda * direction_vector
            assert damping_factor_denominator.shape == (
                self._number_of_topics, )
            damping_factor_denominator *= direction_vector
            damping_factor_denominator = numpy.sum(damping_factor_denominator)

            # print "check point 2", damping_factor_numerator, damping_factor_denominator
            damping_factor_rho = damping_factor_numerator / damping_factor_denominator
            if damping_factor_rho < 0.25:
                damping_factor_lambda *= 1.5
            elif damping_factor_rho > 0.75:
                damping_factor_lambda /= 1.5
            else:
                return hessian_direction_approximation + damping_factor_lambda * direction_vector

        # print damping_factor_numerator, damping_factor_denominator, damping_factor_lambda
        # print "check point 1", hessian_direction_approximation, hessian_direction_approximation + damping_factor_lambda * direction_vector

        damping_factor_lambda = damping_factor_initialization

        return hessian_direction_approximation + damping_factor_lambda * direction_vector

    #
    #
    #
    #
    #


if __name__ == "__main__":
    print("not implemented...")
