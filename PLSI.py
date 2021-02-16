"""
reference:
https://ratsgo.github.io/from%20frequency%20to%20semantics/2017/05/25/plsa/
"""

from copy import deepcopy
import time
import numpy as np
import pandas as pd


def get_row(lst):
    return len(lst)


def get_col(lst):
    return lst.shape[1]


def normalize(lst):
    n_col = get_col(lst)
    n_row = get_row(lst)
    for c in range(n_col):
        max_sum = 0
        for r in range(n_row):
            max_sum += lst[r][c]

        for r in range(n_row):
            lst[r][c] /= max_sum
    return lst


start = time.time()
# prepare corpus
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

# initialize parameters
docunames = ['Doc1', 'Doc2', 'Doc3', 'Doc4', 'Doc5', 'Doc6']
termnames = ['Baseball', 'Basketball', 'Boxing', 'Money', 'Interest', 'Rate', 'Democrat',
             'Republican', 'Cocus', 'President']

ntopic = 3
ndocs = len(docunames)
nterms = len(termnames)
ncorpus = len(docunames)*len(termnames)

posterior_matrix = np.random.uniform(low=0, high=1, size=(ntopic, nterms, ndocs))

pz = np.random.uniform(low=0, high=1, size=(1, ntopic))
pz = pz / np.sum(pz)

pdz_matrix = np.random.uniform(low=0, high=1, size=(ndocs, ntopic))
pdz_matrix = normalize(pdz_matrix)

pwz_matrix = np.random.uniform(low=0, high=1, size=(nterms, ntopic))
pwz_matrix = normalize(pwz_matrix)

parameter = [pwz_matrix, pdz_matrix, pz]


def make_dataframe(docunames, termnames, parameter):
    pwz_matrix = parameter[0]
    pdz_matrix = parameter[1]
    topicnames = ['topic' + str(x + 1) for x in range(3)]
    dtnames = []
    idx = 0

    for doc in docunames:
        for term in termnames:
            dtnames.append(doc + ' ' + term)
            idx += 1

    pdz = pd.DataFrame(pdz_matrix)
    pdz.columns = topicnames
    pdz.index = docunames

    pwz = pd.DataFrame(pwz_matrix)
    pwz.columns = topicnames
    pwz.index = termnames
    return [pwz, pdz, pz]


# Expectation Step
def e_step(parameter, posterior_matrix):
    pwz_matrix = parameter[0]
    pdz_matrix = parameter[1]
    pz = parameter[2]
    # Update p(z|d, w)
    for i in range(ncorpus):
        denominator = np.sum(pz * pwz_matrix[i % nterms][:] * pdz_matrix[i//nterms][:])
        for j in range(ntopic):
            numerator = pz[0, j] * pwz_matrix[i % nterms][j] * pdz_matrix[i//nterms][j]
            posterior_matrix[j][i % nterms][i // nterms] = numerator/denominator

    return posterior_matrix


# Maximization Step
def m_step(posterior_matrix, parameter):
    pwz_matrix = parameter[0]
    pdz_matrix = parameter[1]
    pz = parameter[2]

    # Update p(w|z)
    for i in range(ntopic):
        pwz_denominator = np.sum(corpus.reshape(ncorpus, -1) * posterior_matrix[i].reshape(ncorpus, -1))
        for j in range(nterms):
            pwz_numerator = np.sum(corpus[j, :] * posterior_matrix[i][j][:])
            pwz_matrix[j][i] = pwz_numerator/pwz_denominator

    # Update p(d|z)
    for i in range(ntopic):
        pdz_denominator = np.sum(corpus.reshape(ncorpus, -1) * posterior_matrix[i].reshape(ncorpus, -1))
        pdz_n = 0
        for j in range(ndocs):
            pdz_numerator = np.sum(corpus[:, j] * posterior_matrix[i][:, j])
            pdz_n += pdz_numerator
            pdz_matrix[j][i] = pdz_numerator/pdz_denominator

    # Update p(z)
    for i in range(ntopic):
        pz_numerator = np.sum(posterior_matrix[i].reshape(ncorpus, -1) * corpus.reshape(ncorpus, -1))
        pz_denominator = np.sum(corpus.reshape(ncorpus, -1))
        pz[0, i] = pz_numerator/pz_denominator

    return [pwz_matrix, pdz_matrix, pz]


posterior = e_step(parameter, posterior_matrix)
parameter = m_step(posterior_matrix, parameter)

idx = 0
while True:
    pz = deepcopy(parameter[2])
    posterior = e_step(parameter, posterior)
    parameter = m_step(posterior, parameter)
    idx += 1
    if np.all(pz == parameter[2]):
        break

parameter = make_dataframe(docunames, termnames, parameter)
print(idx)
print(parameter[0])
print(parameter[1])
print(parameter[2])
print("time: ", time.time() - start)
