# internal test for https://github.com/SungjoonPark/KoreanWordVectors
# testsets are from https://github.com/SungjoonPark/KoreanWordVectors
# Byeon Davin 2019.05.

from gensim.models.keyedvectors import KeyedVectors
import similarity_test
import analogy_test
import argparse
import codecs
import sys

analogy_testsets = ["test_dataset/kor_analogy_semantic.txt", "word_analogy_korean.txt"]
similarity_testsets = ["test_dataset/kor_ws353.csv", "WS353_korean.csv"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-vec_file", dest = "vec_file", type=str, help="trained vectors file", default="")
    parser.add_argument("-OOV_file", dest = "OOV_file", type=str, default="")
    parser.add_argument("-testset_num", dest = "testset_num", type=int, default=1)
    parser.add_argument("-out_file", dest = "out_file", type=str, default="")
    parser.add_argument("-stdio", dest = "stdio", type=int, default=1)
    parser.add_argument("-minjn", dest = "minjn", type=int)
    parser.add_argument("-maxjn", dest = "maxjn", type=int)
    parser.add_argument("-minn", dest = "minn", type=int)
    parser.add_argument("-maxn", dest = "maxn", type=int)
    args = parser.parse_args()

    if(args.stdio==1):
        s = "wiki_sejong_jn"
        s = s + str(args.minjn) + str(args.maxjn) + "_n" + str(args.minn) + str(args.maxn) + "_lr025"
        print(s)
        args.vec_file = "../fasttext_output/" + s + ".vec"
        args.OOV_file = "oov/OOV_" + s + ".txt"
        args.out_file = "result/result_" + s + ".txt"
        print(args.vec_file)
        print(args.OOV_file)
        print(args.out_file)
        #sys.exit(0)

    print("loading vectors...")
    vectors = KeyedVectors.load_word2vec_format(args.vec_file, binary=False)

    print("analogy testing...")
    oov = analogy_test.generate_vector(args.OOV_file)
    correct, correct_se, correct_sy, cor_se, cor_sy, total, analogy_missed = analogy_test.analogy_test(vectors, oov)
    print("Similarity: ", str(correct) + "/" + str(total) + " = " + str(correct/total))
    print("Similarity-semantic: ", str(correct_se) + "/5000 = " + str(correct_se/5000))
    for i in range(len(cor_se)):
        print("Similarity-se[" + str(i) + "]: " + str(cor_se[i]) + "/1000 = " + str(cor_se[i]/1000))
    print("Similarity-syntactic: ", str(correct_sy) + "/5000 = " + str(correct_sy/5000))
    for i in range(len(cor_sy)):
        print("Similarity-sy[" + str(i) + "]: " + str(cor_sy[i]) + "/1000 = " + str(cor_sy[i]/1000))

    print("\nsimilarity testing...")
    spearman, pearson, similarity_missed = similarity_test.word_sim_test(similarity_testsets[args.testset_num], vectors)
    print("Missing words: ", similarity_missed)
    print("Spearman coefficient: ", spearman)
    print("Pearson coefficient: ", pearson)

    with codecs.open(args.out_file, 'w', encoding='utf8') as output:
        output.write("test: " + args.vec_file + '\n')
        output.write("\n[analogy test]\n")
        output.write("Similarity :" + str(correct) + "/" + str(total) + " = " + str(correct/total) + '\n')
        output.write("Similarity-semantic: " +  str(correct_se) + "/5000 = " + str(correct_se/5000) + '\n')
        for i in range(len(cor_se)):
            output.write("Similarity-se[" + str(i) + "]: " + str(cor_se[i]) + "/1000 = " + str(cor_se[i]/1000) + '\n')
        output.write("Similarity-syntactic: " + str(correct_sy) + "/5000 = " + str(correct_sy/5000) + '\n')
        for i in range(len(cor_sy)):
            output.write("Similarity-sy[" + str(i) + "]: " + str(cor_sy[i]) + "/1000 = " + str(cor_sy[i]/1000) + '\n')
        output.write("\n[similarity test]\nMissing words :" + str(similarity_missed) + '\n')
        output.write("Spearman coefficient :" + str(spearman) + '\n')
        output.write("Pearson coefficient :" + str(pearson) + '\n\n\n')
