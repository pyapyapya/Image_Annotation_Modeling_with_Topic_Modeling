from collections import Counter

import numpy as np
from tqdm import tqdm


class LDA:
    def __init__(self):
        counter = Counter()

        docunames = ['Doc1', 'Doc2', 'Doc3', 'Doc4', 'Doc5', 'Doc6']
        termnames = ['Baseball', 'Basketball', 'Boxing', 'Money', 'Interest', 'Rate', 'Democrat',
                     'Republican', 'Cocus', 'President']
        """
        self.corpus = [["Hadoop", "Big Data", "HBase", "Java", "Spark", "Storm", "Cassandra"],
                    ["NoSQL", "MongoDB", "Cassandra", "HBase", "Postgres"],
                    ["Python", "scikit-learn", "scipy", "numpy", "statsmodels", "pandas"],
                    ["R", "Python", "statistics", "regression", "probability"],
                    ["machine learning", "regression", "decision trees", "libsvm"],
                    ["Python", "R", "Java", "C++", "Haskell", "programming languages"],
                    ["statistics", "probability", "mathematics", "theory"],
                    ["machine learning", "scikit-learn", "Mahout", "neural networks"],
                    ["neural networks", "deep learning", "Big Data", "artificial intelligence"],
                    ["Hadoop", "Java", "MapReduce", "Big Data"],
                    ["statistics", "R", "statsmodels"],
                    ["C++", "deep learning", "artificial intelligence", "probability"],
                    ["pandas", "R", "Python"],
                    ["databases", "HBase", "Postgres", "MySQL", "MongoDB"],
                    ["libsvm", "regression", "support vector machines"]]
        """

        self.corpus = np.array([[1, 2, 0, 0, 0, 0],
                               [3, 1, 0, 0, 0, 0],
                               [2, 0, 0, 0, 0, 0],
                               [3, 3, 2, 3, 2, 4],
                               [0, 0, 3, 2, 0, 0],
                               [0, 0, 4, 1, 0, 0],
                               [0, 0, 0, 0, 4, 3],
                               [0, 0, 0, 0, 2, 1],
                               [0, 0, 0, 0, 3, 2],
                               [0, 0, 1, 0, 2, 3]])

        # self.list2np()
        # self.new_corpus = self.make_corpus()
        # self.doc2BOW()
        self.n_iter = 100
        self.n_topic = 3
        self.n_docs = self.corpus.shape[0]
        self.n_terms = self.corpus.shape[1]

        self.topic_corpus = []
        self.doc_topic_count = np.zeros((self.n_docs, self.n_topic), dtype=np.int32)
        self.topic_term_count = np.zeros((self.n_terms, self.n_topic), dtype=np.int32)
        self.topic_count = np.zeros(self.n_topic, dtype=np.int32)
        self.doc_length = np.array([self.corpus[i].sum() for i in range(self.n_docs)])
        for doc_idx, docs in enumerate(self.corpus):
            doc_topic = []

            for term_idx, term in enumerate(docs):
                if term == 0:
                    continue
                for i in range(term):
                    topic = np.random.randint(self.n_topic)
                    doc_topic.append((term_idx, topic))
                    self.doc_topic_count[doc_idx][topic] += 1
                    self.topic_term_count[term_idx][topic] += 1
                    self.topic_count[topic] += 1
            self.topic_corpus.append(doc_topic)

        self.lda()

    def lda(self):
        for iter in tqdm(range(self.n_iter)):
            print("iter", iter+1, "/", self.n_iter)
            for doc_idx, docs in enumerate(self.topic_corpus):
                for i, (term_idx, topic) in enumerate(self.topic_corpus[doc_idx]):
                    if self.corpus[doc_idx][term_idx] == 0:
                        continue
                    self.doc_topic_count[doc_idx][topic] -= 1
                    self.topic_term_count[term_idx][topic] -= 1
                    self.topic_count[topic] -= 1
                    self.doc_length[doc_idx] -= 1
                    new_topic = self.new_topic(doc_idx, term_idx)
                    self.topic_corpus[doc_idx][i] = (term_idx, new_topic)
                    self.doc_topic_count[doc_idx][new_topic] += 1
                    self.topic_term_count[term_idx][new_topic] += 1
                    self.topic_count[new_topic] += 1
                    self.doc_length[doc_idx] += 1
            self.likelihood()
            print(self.topic_term_count[:, 0].sum()/self.topic_count.sum())
            print(self.topic_term_count[:, 1].sum()/self.topic_count.sum())
            print(self.topic_term_count[:, 2].sum()/self.topic_count.sum())

    def topic_given_doc(self, doc, topic):
        alpha = 0.1
        a = (self.doc_topic_count[doc, topic] + alpha) / (self.doc_length[doc] + self.n_topic * alpha)
        return a

    def term_given_topic(self, term, topic):
        beta = 0.01
        b = (self.topic_term_count[term, topic] + beta) / (self.topic_count[topic] + self.n_terms * beta)
        return b

    def new_topic(self, doc, term):
        topic_weight = np.zeros(self.n_topic)
        for topic in range(self.n_topic):
            a = self.topic_given_doc(doc, topic)
            b = self.term_given_topic(term, topic)
            topic_weight[topic] = a * b
        topic_weight /= topic_weight.sum()
        new_topic = np.random.multinomial(1, topic_weight).argmax()
        return new_topic

    def likelihood(self):
        likelihood = 0
        for doc_idx, docs in enumerate(self.corpus):
            for term_idx, term in enumerate(docs):
                if not term == 0:
                    n_id = self.corpus[doc_idx][term_idx]
                    p_w = 0
                    for topic in range(self.n_topic):
                        p_w += self.doc_topic_count[doc_idx][topic] * self.topic_term_count[term_idx][topic]
                    p_w = np.log10(p_w)
                    likelihood += n_id * p_w
        perplexity = np.exp(-likelihood/np.sum(self.corpus))
        print('perplexity', perplexity)

    def result(self):
        doc_topic = np.zeros(self.n_topic)
        for doc in range(self.n_docs):
            print(self.doc_topic_count[doc].argmax())
        for term in range(self.n_terms):
            print(self.topic_term_count[term].argmax())


LDA()
