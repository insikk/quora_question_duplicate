"""
Preprocessing Quora Question Duplicates dataset.
It generate simplified version. 
"""

import argparse
import json
import os
import csv

from collections import Counter

from tqdm import tqdm



def main():
    args = get_args()
    prepro(args)


def get_args():
    parser = argparse.ArgumentParser()
    home = os.path.expanduser("~")
    # source_dir = os.path.join("..", "data", "quora")
    # target_dir = os.path.join("..", "data", "quora")
    source_dir = os.path.join("tiny")
    target_dir = os.path.join("tiny")
    glove_dir = os.path.join(home, "data", "glove")
    parser.add_argument('-s', "--source_dir", default=source_dir)
    parser.add_argument('-t', "--target_dir", default=target_dir)
    parser.add_argument('-d', "--debug", action='store_true')
    
    parser.add_argument("--glove_corpus", default="840B")
    parser.add_argument("--glove_dir", default=glove_dir)
    parser.add_argument("--glove_vec_size", default=300, type=int)
    parser.add_argument("--tokenizer", default="PTB", type=str)
    
    parser.add_argument("--split", action='store_true') # SNLI is single sentence. we do not need split. 

    return parser.parse_args()

def prepro(args):
    if not os.path.exists(args.target_dir):
        os.makedirs(args.target_dir)

    prepro_each(args, 'train', out_name='train')
    prepro_each(args, 'dev', out_name='dev')
    prepro_each(args, 'test', out_name='test')

def get_word2vec(args, word_counter):
    glove_path = os.path.join(args.glove_dir, "glove.{}.{}d.txt".format(args.glove_corpus, args.glove_vec_size))
    sizes = {'6B': int(4e5), '42B': int(1.9e6), '840B': int(2.2e6), '2B': int(1.2e6)}
    total = sizes[args.glove_corpus]
    word2vec_dict = {}
    with open(glove_path, 'r', encoding='utf-8') as fh:
        for line in tqdm(fh, total=total):
            array = line.lstrip().rstrip().split(" ")
            word = array[0]
            vector = list(map(float, array[1:]))
            if word in word_counter:
                word2vec_dict[word] = vector
            elif word.capitalize() in word_counter:
                word2vec_dict[word.capitalize()] = vector
            elif word.lower() in word_counter:
                word2vec_dict[word.lower()] = vector
            elif word.upper() in word_counter:
                word2vec_dict[word.upper()] = vector

    print("{}/{} of word vocab have corresponding vectors in {}".format(len(word2vec_dict), len(word_counter), glove_path))
    return word2vec_dict


def process_tokens(temp_tokens):
    import re
    tokens = []
    for token in temp_tokens:
        flag = False
        l = ("-", "\u2212", "\u2014", "\u2013", "/", "~", '"', "'", "\u201C", "\u2019", "\u201D", "\u2018", "\u00B0")
        # \u2013 is en-dash. Used for number to nubmer
        # l = ("-", "\u2212", "\u2014", "\u2013")
        # l = ("\u2013",)
        tokens.extend(re.split("([{}])".format("".join(l)), token))
    return tokens


def save(args, data, shared, data_type):
    data_path = os.path.join(args.target_dir, "data_{}.json".format(data_type))
    shared_path = os.path.join(args.target_dir, "shared_{}.json".format(data_type))
    json.dump(data, open(data_path, 'w'))
    json.dump(shared, open(shared_path, 'w'))

def prepro_each(args, data_type, out_name="default"):    
    if args.tokenizer == "PTB":
        import nltk
        sent_tokenize = nltk.sent_tokenize
        def word_tokenize(tokens):
            return [token.replace("''", '"').replace("``", '"') for token in nltk.word_tokenize(tokens)]
    elif args.tokenizer == 'Stanford':
        # from my.corenlp_interface import CoreNLPInterface
        # interface = CoreNLPInterface(args.url, args.port)
        # sent_tokenize = interface.split_doc
        # word_tokenize = interface.split_sent
        pass
    else:
        raise Exception()

    if not args.split:
        sent_tokenize = lambda para: [para]

    source_path = os.path.join(args.source_dir, "quora_wang_{}.tsv".format(data_type))
    source_reader = csv.reader(open(source_path, 'r'), delimiter='\t')
    # next(source_reader, None)  # skip the headers
    
    x_list, cx_list = [], []
    y_list, cy_list = [], []
    z_list = []

    word_counter, char_counter, lower_word_counter = Counter(), Counter(), Counter()    
    for items in tqdm(source_reader, total=source_reader.line_num):
        if len(items) == 0: # skip blank lines between data point
            continue

        sent1 = items[1]
        sent2 = items[2]
        label = int(items[0])

        # entry_id = int(items[0])
        # qid1 = int(items[1])
        # qid2 = int(items[2])
        # sent1 = items[3]
        # sent2 = items[4]
        # label = int(items[5])

        def process_sentence(sent):
            """
            get list of processed tokens, and count the words

            return xi: list of word tokens
            return cxi: list of list of characters
            """
            xi = word_tokenize(sent)
            xi = process_tokens(xi) # process tokens
            # given xi, add chars
            cxi = [[xijk for xijk in xij] for xij in xi]            
            
            # print(xi)
            for xij in xi:
                word_counter[xij] += 1
                lower_word_counter[xij.lower()] += 1
                for xijk in xij:
                        char_counter[xijk] += 1
            return xi, cxi

        x, cx = process_sentence(sent1)

        x_list.append(x)
        cx_list.append(cx)

        x, cx = process_sentence(sent2)

        y_list.append(x)
        cy_list.append(cx)

        if label == 0:
            z = [1, 0] # no duplicate
        else:
            z = [0, 1] # duplicate
        
        z_list.append(z)

        
        if args.debug:
            break

    word2vec_dict = get_word2vec(args, word_counter)
    lower_word2vec_dict = get_word2vec(args, lower_word_counter)

    # add context here
    data = {'x_list': x_list, 'cx_list': cx_list, 'y_list': y_list, 'cy_list': cy_list, 'z_list': z_list}
    shared = {'word_counter': word_counter, 'char_counter': char_counter, 'lower_word_counter': lower_word_counter,
              'word2vec': word2vec_dict, 'lower_word2vec': lower_word2vec_dict}

    print("saving ...")
    save(args, data, shared, out_name)

if __name__ == "__main__":
    main()