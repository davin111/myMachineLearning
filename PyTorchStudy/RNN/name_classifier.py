#studying: https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html

from __future import unicode_literals, print_function, division
from io import open
import glob
import os


"""
step1 < data preprocessing >
"""
def findFiles(path):
	return glob.glob(path)

print(findFiles('data/names/*.txt'))

import unicodedata
import string

all_letters = string.ascii_letters + ".,:'"
n_letters = len(all_letters)

def unicodeToAscii(s):
	return ''.join(
		c for c in unicodedata.normalize('NFD', s)
		if unicodedata.category(c) != 'Mn'
			and c in all_letters
	)

print(unicodeToAscii('Ślusàrski'))



category_lines = {} #dictionary mapping each category to a list of names
all_categories = [] #just a list of languages

def readLines(filename):
	lines = open(filename, encoding='utf-8').read().strip().split('\n')
	return [unicodeToAscii(line) for line in lines]


for filename in findFiles('data/names/*.txt'):
	category = os.path.splitext(os.path.basename(filename))[0]
	all_categories.append(category)
	lines = readLines(filename)
	category_lines[category] = lines

n_categories = len(all_categories)


"""
step2 < names into Tensors >
"""
#use one-hot encoding for a single letter
import torch

def lettertoIndex(letter):
	return all_letters.find(letter)

def letterToTensor(letter):
	return all_
