# internal test for https://github.com/SungjoonPark/KoreanWordVectors
# testsets are from https://github.com/SungjoonPark/KoreanWordVectors
# Byeon Davin 2019.05.

from gensim.models.keyedvectors import KeyedVectors
import numpy as np
import numpy.linalg as la
import scipy.stats as st
import argparse
import decompose_dir

testsets = ["test_dataset/kor_ws353.csv", "WS353_korean.csv"]


def normalize(array):
    norm = la.norm(array)
    return array / norm


def create_word_vector(word, pos_vectors):
    word_vector = pos_vectors.word_vec(decompose_dir.jamo_split(word))
    return normalize(word_vector)


def word_sim_test(filename, pos_vectors):
    delim = ','
    actual_sim_list, pred_sim_list = [], []
    missed = 0

    with open(filename, 'r') as pairs:
        for pair in pairs:
            w1, w2, actual_sim = pair.strip().split(delim)

            try:
                w1_vec = create_word_vector(w1, pos_vectors)
                w2_vec = create_word_vector(w2, pos_vectors)
                pred = float(np.inner(w1_vec, w2_vec))
                actual_sim_list.append(float(actual_sim))
                pred_sim_list.append(pred)

            except KeyError:
                missed += 1

    spearman, _ = st.spearmanr(actual_sim_list, pred_sim_list)
    pearson, _ = st.pearsonr(actual_sim_list, pred_sim_list)

    return spearman, pearson, missed


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("pos_file", type=str, help="trained vectors file")
    parser.add_argument("testset_num", type=int)
    args = parser.parse_args()
    print(args)

    print("loading vectors...")
    pos_vectors = KeyedVectors.load_word2vec_format(args.pos_file, binary=False)
    print("testing...")

    spearman, pearson, missed = word_sim_test(testsets[args.testset_num], pos_vectors)
    print("Missing words :", missed)
    print("Spearman coefficient :", spearman)
    print("Pearson coefficient :", pearson)
