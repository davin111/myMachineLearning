# internal test for https://github.com/SungjoonPark/KoreanWordVectors
# testsets are from https://github.com/SungjoonPark/KoreanWordVectors
# Byeon Davin 2019.05.

from gensim.models.keyedvectors import KeyedVectors
import argparse
import decompose_dir
import numpy as np

testsets = ["test_dataset/kor_analogy_semantic.txt", "word_analogy_korean.txt"]


def normalize(array):
    norm = np.linalg.norm(array)
    return array / norm


def analogy_test(vectors, oov):
    correct, correct_se, correct_sy, total, missed = 0, 0, 0, 0, 0
    cor_se, cor_sy = [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]
    oov_cnt = 0
    with open(testsets[1], 'r') as lines:
        for line in lines:
            if line.startswith(":") or line.startswith("#") or len(line) <= 1:
                continue

            words = line.strip().split(" ")
            poss = list()
            for word in words:
                poss.append(decompose_dir.jamo_split(word))
            total += 1

            vec = []
            for p in poss:
                try:
                    v = vectors.get_vector(p)
                except KeyError:
                    v = oov[oov_cnt]
                    oov_cnt += 1
                vec.append(v)

                
            similar_vec = normalize(vec[3])
            problem_vec = normalize(vec[1] + vec[2] - vec[0])

            cos_sim = float(np.dot(similar_vec, problem_vec))
            correct += cos_sim
            if total <= 5000:
                correct_se += cos_sim
                cor_se[int((total-1)/1000)] += cos_sim
            else:
                correct_sy += cos_sim
                cor_sy[int((total-1)/1000-5)] += cos_sim

    print(oov_cnt)
    return correct, correct_se, correct_sy, cor_se, cor_sy, total, missed-oov_cnt


def generate_vector(oov_file):
    with open(oov_file, 'r') as lines:
        oovs = []
        for line in lines:
            values = line.strip().split(" ")
            del values[0]
            v = np.array(values, dtype=np.float32)
            oovs.append(v)

    oovs = np.array(oovs)
    print(oovs.shape)
    return oovs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("vec_file", type=str, help="trained word vectors file")
    parser.add_argument("OOV_file", type=str)
    args = parser.parse_args()
    print(args)

    oov = generate_vector(args.OOV_file)

    print("loading vectors...")
    vectors = KeyedVectors.load_word2vec_format(args.vec_file, binary=False)

    print("analogy testing...")
    correct, correct_se, correct_sy, cor_se, cor_sy, total, missed = analogy_test(vectors, oov)

    print("Similarity: ", str(correct) + "/" + str(total) + " = " + str(correct/total))
    print("Similarity-semantic: ", str(correct_se) + "/5000 = " + str(correct_se/5000))
    for i in range(len(cor_se)):
        print("Similarity-se[" + str(i) + "]: " + str(cor_se[i]) + "/1000 = " + str(cor_se[i]/1000))
    print("Similarity-syntactic: ", str(correct_sy) + "/5000 = " + str(correct_sy/5000))
    for i in range(len(cor_sy)):
        print("Similarity-sy[" + str(i) + "]: " + str(cor_sy[i]) + "/1000 = " + str(cor_sy[i]/1000))
