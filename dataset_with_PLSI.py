import json
import os
import time
from typing import List, Dict, Set

import numpy as np
import pandas as pd
from tqdm import tqdm
from nltk.stem import WordNetLemmatizer
from matplotlib import pyplot as plt


class Preprocess:
    @staticmethod
    def name_tokenize(object_name, lm):
        if object_name[0] == '"':
            object_name = object_name[1:-1]
        object_name = object_name.replace(' ', '_')
        object_name = lm.lemmatize(object_name, pos='n')
        return object_name

    @staticmethod
    def object_size(object_list: List) -> int:
        return len(object_list)

    @staticmethod
    def make_word() -> [np.array, List, List]:
        image_id_list: List = []
        object_id_name: Dict = {}

        with open('C:\dataset\\visualgenome\scene_graphs.json') as json_data:
            lm = WordNetLemmatizer()
            visual_genome_data = json.load(json_data)
            for idx, data in tqdm(enumerate(visual_genome_data[:300])):
                image_id: str = str(data['image_id'])
                image_id_list.append(image_id)

                for objects in data['objects']:
                    object_name: str = Preprocess.name_tokenize(objects['names'][0], lm)
                    object_id_name[objects['object_id']] = object_name

            object_list: List = sorted(set(object_id_name.values()))
            corpus = np.zeros((len(image_id_list), len(object_list)))

            for doc_idx, data in tqdm(enumerate(visual_genome_data[:300])):
                for objects in data['objects']:
                    object_name: str = Preprocess.name_tokenize(objects['names'][0], lm)
                    term_index = object_list.index(object_name)
                    corpus[doc_idx][term_index] += 1

        return corpus, image_id_list, object_list


class ParseTag:
    def __init__(self):
        self.image_id_set: Set = set()
        self.tag_set: Set = set()
        self.parse_tag()
        self.docnames, self.termnames, self.corpus = self.make_corpus()

    def parse_tag(self):
        with open('C:\dataset\\visualgenome\image_tag_data1.csv', 'r', encoding='UTF-8-sig') as csv_file:
            for line in csv_file.readlines()[1:301]:
                line = line.split(',')
                self.image_id_set.add(line[0])
                for tag in line[1:]:
                    if tag == '':
                        break
                    self.tag_set.add(tag)

    def make_corpus(self):
        docnames: np.array = np.array(sorted(self.image_id_set, key=int))
        termnames: np.array = np.array(sorted(self.tag_set))
        corpus: np.array = np.zeros((len(termnames), len(docnames)))
        with open('C:\dataset\\visualgenome\image_tag_data1.csv', 'r', encoding='UTF-8-sig') as csv_file:
            for line in csv_file.readlines()[1:301]:
                line = line.split(',')
                image_id = line[0]
                for tag in line[1:]:
                    if tag == '':
                        break
                    # doc_idx = docnames.index(image_id)
                    # term_idx = termnames.index(tag)
                    corpus[termnames == tag, docnames == str(image_id)] += 1
        return docnames, termnames, corpus

    @staticmethod
    def example_corpus():
        corpus = np.array([[1, 2, 0, 0, 0, 0],
                           [3, 1, 0, 0, 0, 0],
                           [2, 0, 0, 0, 0, 0],
                           [3, 3, 2, 3, 2, 4],
                           [0, 0, 3, 2, 0, 0],
                           [0, 0, 4, 1, 0, 0],
                           [0, 0, 0, 0, 4, 3],
                           [0, 0, 0, 0, 2, 1],
                           [0, 0, 0, 0, 3, 2],
                           [0, 0, 1, 0, 2, 3]])

        docunames = np.array(['Doc1', 'Doc2', 'Doc3', 'Doc4', 'Doc5', 'Doc6'])
        termnames = np.array(['Baseball', 'Basketball', 'Boxing', 'Money', 'Interest', 'Rate', 'Democrat',
                              'Republican', 'Cocus', 'President'])

        return docunames, termnames, corpus

    @property
    def get_corpus(self):
        return self.docnames, self.termnames, self.corpus


class PLSA:
    def __init__(self, n_iter: int, ntopic, parse_tag: ParseTag):
        start = time.time()
        # self.docunames, self.termnames, self.corpus = parse_tag.get_corpus
        self.docunames, self.termnames, self.corpus = parse_tag.example_corpus()

        self.n_iter = n_iter

        self.ntopic = ntopic
        self.ndocs = len(self.docunames)
        self.nterms = len(self.termnames)
        self.ncorpus = self.ndocs * self.nterms

        self.likelihood = 0.0
        self.beta = 0.8

        idx_lst: List = []
        loss_lst: List = []
        pz_0: List = []
        pz_1: List = []
        pz_2: List = []

        # Initialize Matrix
        posterior_matrix: np.array = np.random.uniform(low=0, high=1, size=(self.ntopic, self.nterms, self.ndocs))
        self.init_posterior = posterior_matrix

        pz: np.array = np.random.uniform(low=0, high=1, size=self.ntopic)
        pz = pz / np.sum(pz)

        pdz_matrix: np.array = np.random.uniform(low=0, high=1, size=(self.ndocs, self.ntopic))
        pdz_matrix = self.normalize(pdz_matrix)

        pwz_matrix: np.array = np.random.uniform(low=0, high=1, size=(self.nterms, self.ntopic))
        pwz_matrix = self.normalize(pwz_matrix)

        pzd_matrix: np.array = np.random.uniform(low=0, high=1, size=(self.ndocs, self.ntopic))

        posterior_matrix = self._e_step(pwz_matrix, pdz_matrix, pz, posterior_matrix)
        pwz_matrix, pdz_matrix, pz = self._m_step(posterior_matrix, pwz_matrix, pdz_matrix, pz)

        idx = 0
        cur = 0

        # start PLSA algorithm
        while True:
            posterior_matrix = self._e_step(pwz_matrix, pdz_matrix, pz, posterior_matrix)
            pwz_matrix, pdz_matrix, pz = self._m_step(posterior_matrix, pwz_matrix, pdz_matrix, pz)
            self.cal_log_likelihood(pwz_matrix, pdz_matrix, pz)
            delta = self.likelihood - cur
            idx += 1
            print(idx)
            print(pz)
            print(delta)

            if cur != 0:
                loss_lst.append(delta)
                idx_lst.append(idx)
                pz_0.append(pz[0])
                pz_1.append(pz[1])
                pz_2.append(pz[2])

            if cur != 0 and delta < 1e-6:
                break
            cur = self.likelihood

        # Update Posterior
        posterior_matrix = self._e_step(pwz_matrix, pdz_matrix, pz, posterior_matrix)

        pzd_matrix = self._cal_pzd(pzd_matrix, pwz_matrix, pdz_matrix, pz)
        aic, bic, caic, icl_bic, likelihood = self.model_selection_criteria(posterior_matrix)

        # Save Result
        self.pwz, self.pdz, self.pz, self.pzd, self.posterior = \
            self.make_dataframe(pwz_matrix, pdz_matrix, pz, pzd_matrix, posterior_matrix)

        save_result = SaveResult(n_iter, parse_tag)
        save_result.save_probability(self.pwz, self.pdz, self.pz, self.pzd, self.posterior, self.init_posterior)
        save_result.save_criteria(aic, bic, caic, icl_bic, likelihood)
        save_result.result_graph(idx_lst, loss_lst, pz_0, pz_1, pz_2)

        print(idx)
        print(self.pwz)
        print(self.pdz)
        print(self.pz)
        print(self.pzd)
        print("time: ", time.time() - start)

    # Expectation Step
    def _e_step(self, pwz_matrix: np.array, pdz_matrix: np.array, pz: np.array, posterior_matrix: np.array) -> np.array:

        # Update p(z|d, w)
        for i in range(self.ncorpus):
            term_idx = i % self.nterms
            doc_idx = i // self.nterms
            denominator = np.sum(pz * pwz_matrix[term_idx] * pdz_matrix[doc_idx])
            if denominator == 0.0:
                continue
            for j in range(self.ntopic):
                numerator = pz[j] * pwz_matrix[term_idx][j] * pdz_matrix[doc_idx][j]
                posterior_matrix[j][term_idx][doc_idx] = numerator / denominator
        return posterior_matrix

    # Maximization Step
    def _m_step(self, posterior_matrix, pwz_matrix, pdz_matrix, pz):
        # Update p(w|z)
        for i in range(self.ntopic):
            pwz_denominator = np.sum(self.corpus * posterior_matrix[i])
            if pwz_denominator == 0.0:
                continue
            for j in range(self.nterms):
                pwz_numerator = np.sum(self.corpus[j, :] * posterior_matrix[i][j])
                pwz_matrix[j][i] = pwz_numerator / pwz_denominator

        # Update p(d|z)
        for i in range(self.ntopic):
            pdz_denominator = np.sum(self.corpus * posterior_matrix[i])
            if pdz_denominator == 0.0:
                continue
            for j in range(self.ndocs):
                pdz_numerator = np.sum(self.corpus[:, j] * posterior_matrix[i][:, j])
                pdz_matrix[j][i] = pdz_numerator / pdz_denominator

        # Update p(z)
        for i in range(self.ntopic):
            pz_numerator = np.sum(posterior_matrix[i] * self.corpus)
            pz_denominator = np.sum(self.corpus)
            pz[i] = pz_numerator / pz_denominator
        return pwz_matrix, pdz_matrix, pz

    def cal_log_likelihood(self, pwz, pdz, pz):
        self.likelihood = 0.0
        for doc in range(self.ndocs):
            for term in range(self.nterms):
                pdw = self._cal_pdw(pwz, pdz, pz, doc, term)
                if pdw > 0:
                    self.likelihood += self.corpus[term][doc] * np.log10(pdw)

    def model_selection_criteria(self, posterior):
        delta = self.likelihood
        nu = (self.ntopic - 1) * (1 + self.ndocs - 1 + self.nterms - 1)
        n = self.ncorpus
        posterior_sum = 0

        for doc in range(self.ndocs):
            for term in range(self.nterms):
                for topic in range(self.ntopic):
                    posterior_prob = posterior[topic][term][doc]
                    if posterior_prob > 0:
                        posterior_sum += posterior_prob * np.log10(posterior_prob)

        aic = -2 * delta + 2 * nu
        bic = -2 * delta + nu * np.log10(n)
        caic = -2 * delta + nu * (np.log10(n) + 1)
        icl_bic = bic + -2 * posterior_sum
        return aic, bic, caic, icl_bic, self.likelihood

    def make_dataframe(self, pwz_matrix, pdz_matrix, pz, pzd_matrix, posterior_matrix: np.array):
        topicnames: List = ['topic' + str(x + 1) for x in range(self.ntopic)]
        dtnames: List = []
        prob_word: List = []
        prob_word.append('prob')

        idx = 0

        for doc in self.docunames:
            for term in self.termnames:
                dtnames.append(doc + ' ' + term)
                idx += 1

        pdz = pd.DataFrame(pdz_matrix)
        pdz.columns = topicnames
        pdz.index = self.docunames

        pwz = pd.DataFrame(pwz_matrix)
        pwz.columns = topicnames
        pwz.index = self.termnames

        pz = pd.DataFrame(pz)
        pz.columns = prob_word
        pz.index = topicnames

        pzd = pd.DataFrame(pzd_matrix)
        pzd.columns = topicnames
        pzd.index = self.docunames

        posterior_matrix = posterior_matrix.reshape(-1, self.ndocs)
        posterior = pd.DataFrame(posterior_matrix)
        posterior_index = np.tile(self.termnames, self.ntopic)
        posterior.columns = self.docunames
        posterior.index = posterior_index

        self.init_posterior = self.init_posterior.reshape(-1, self.ndocs)
        self.init_posterior = pd.DataFrame(self.init_posterior)
        self.init_posterior.columns = self.docunames
        self.init_posterior.index = posterior_index

        return pwz, pdz, pz, pzd, posterior

    def _cal_pdw(self, pwz, pdz, pz, doc, term):
        pdw = 0.0
        for topic in range(self.ntopic):
            pdw += pz[topic] * pdz[doc][topic] * pwz[term][topic]

        return pdw

    def _cal_pzd(self, pzd, pwz, pdz, pz) -> np.array:
        for doc in range(self.ndocs):
            pzd_denominater = 0.0
            for term in range(self.nterms):
                pzd_denominater += self._cal_pdw(pwz, pdz, pz, doc, term)
            for topic in range(self.ntopic):
                pzd_numerator = 0.0
                for term in range(self.nterms):
                    pzd_numerator += pz[topic] * pdz[doc][topic] * pwz[term][topic]
                pzd[doc][topic] = pzd_numerator/pzd_denominater

        return pzd

    def normalize(self, lst):
        n_col = self.get_col(lst)
        n_row = self.get_row(lst)
        for c in range(n_col):
            max_sum = 0
            for r in range(n_row):
                max_sum += lst[r][c]

            for r in range(n_row):
                lst[r][c] /= max_sum
        return lst

    @classmethod
    def get_row(cls, lst):
        return len(lst)

    @classmethod
    def get_col(cls, lst):
        return lst.shape[1]

    @property
    def get_dataframe(self):
        return self.pwz, self.pdz, self.pz, self.pzd, self.posterior

    @property
    def get_likelihood(self):
        return self.likelihood

    @property
    def get_init_posterior(self):
        return self.init_posterior


class SaveResult:
    def __init__(self, idx: int, parse_tag, ntopic=3):
        self.parse_tag = parse_tag
        # self.docunames, self.termnames, self.corpus = parse_tag.get_corpus
        self.docunames, self.termnames, self.corpus = parse_tag.example_corpus()

        self.ntopic = ntopic
        self.ndocs = len(self.docunames)
        self.nterms = len(self.termnames)
        self.ncorpus = self.ndocs * self.nterms

        self.idx: str = str(idx)

    def save_probability(self, pwz: pd.DataFrame, pdz: pd.DataFrame, pz: pd.DataFrame,
                         pzd: pd.DataFrame, posterior: pd.DataFrame, init_posterior: pd.DataFrame):
        self._save_pwz(pwz)
        self._save_pdz(pdz)
        self._save_pz(pz)
        self._save_pzd(pzd)
        self._save_posterior(posterior)
        self._save_init_posterior(init_posterior)

    def save_criteria(self, aic, bic, caic, icl_bic, likelihood):
        criteria: pd.DataFrame = self._make_dataframe_criteria(aic, bic, caic, icl_bic, likelihood)
        file_name = 'criteria' + self.idx + '.csv'
        path = self._get_path(file_name)
        criteria.to_csv(path, mode='w', encoding='UTF-8')

    def _save_pwz(self, pwz):
        file_name = 'pwz' + self.idx + '.csv'
        path = self._get_path(file_name)
        pwz.to_csv(path, mode='w', encoding='UTF-8')

    def _save_pdz(self, pdz):
        file_name = 'pdz' + self.idx + '.csv'
        path = self._get_path(file_name)
        pdz.to_csv(path, mode='w', encoding='UTF-8')

    def _save_pz(self, pz):
        file_name = 'pz' + self.idx + '.csv'
        path = self._get_path(file_name)
        pz.to_csv(path, mode='w', encoding='UTF-8')

    def _save_pzd(self, pzd):
        file_name = 'pzd' + self.idx + '.csv'
        path = self._get_path(file_name)
        pzd.to_csv(path, mode='w', encoding='UTF-8')

    def _save_posterior(self, posterior):
        file_name = 'posterior' + self.idx + '.csv'
        path = self._get_path(file_name)
        posterior.to_csv(path, mode='w', encoding='UTF-8')

    def _save_init_posterior(self, init_posterior):
        file_name = 'init_posterior' + self.idx + '.csv'
        path = self._get_path(file_name)
        init_posterior.to_csv(path, mode='w', encoding='UTF-8')

    def result_graph(self, idx_lst, loss_lst, pz_0, pz_1, pz_2):
        file_name = 'result_graph' + self.idx + '.png'
        plt.subplot(211)
        plt.plot(idx_lst, loss_lst)
        plt.xlabel('Iter')
        plt.ylabel('Loss')

        plt.subplot(212)
        plt.plot(idx_lst, pz_0)
        plt.plot(idx_lst, pz_1)
        plt.plot(idx_lst, pz_2)
        plt.xlabel('Iter')
        plt.ylabel('pz')
        path = self._get_path(file_name)
        plt.savefig(path)
        plt.clf()
        # plt.show()

    @staticmethod
    def _get_path(file_name):
        save_path = 'C:\dataset\\visualgenome'
        return os.path.join(save_path, file_name)

    @staticmethod
    def _make_dataframe_criteria(aic, bic, caic, icl_bic, likelihood) -> pd.DataFrame:
        criteria = {'aic': aic,
                    'bic': bic,
                    'caic': caic,
                    'icl_bic': icl_bic,
                    'likelihood': likelihood}
        return pd.DataFrame(criteria, index=[0])


def main():
    parsetag = ParseTag()
    likelihood_lst: List = []
    init_posterior_lst: List = []

    for idx in range(1):
        plsa = PLSA(idx, 3, parsetag)
        likelihood_lst.append(plsa.get_likelihood)
        init_posterior_lst.append(plsa.get_init_posterior)


main()
