import xml.dom.minidom
# import metapy
import string
from nltk.stem import LancasterStemmer, WordNetLemmatizer
from collections import Counter
import pickle
import math
import numpy as np
import json
import time
import os
import sys
import random
import itertools
# import krovetz
import copy
from scipy import stats

lemmatizer = WordNetLemmatizer()
lemmatized_words = {}
# ks = krovetz.PyKrovetzStemmer()
stemmed_words = {}
# stopwords_file = "../Session_track_2014/clueweb_snippet/lemur-stopwords.txt"
stopwords_file = "../lemur-stopwords.txt"
stemmed_reverse_words = {}


def postprocess_query(query, topic_desc):
    # words = topic_desc.split()
    topic_desc, stemmed_reverse_words_topic = preprocess(topic_desc, stopwords_file, lemmatizing=True,
                                                         get_stemmed_reverse_words=True)
    '''
    postprocess_words = {}
    for word in words:
        prepr_word = preprocess(word, stopwords_file, lemmatizing = True)
        if prepr_word != "":
            try:
                postprocess_words[prepr_word] += [word]
            except:
                postprocess_words[prepr_word] = [word]
    '''
    post_pr_query = []
    for word in query.split():
        try:
            there = stemmed_reverse_words_topic[word]
            postprocess_word = random.choice(stemmed_reverse_words_topic[word])
            post_pr_query += [postprocess_word]
        except:
            try:
                there = stemmed_reverse_words[word]
                postprocess_word = random.choice(stemmed_reverse_words[word])
                post_pr_query += [postprocess_word]
            except:
                post_pr_query += [word]
    return " ".join(post_pr_query)


class Document(object):
    """
    Basic representation of a document - including a unique identifier (index ID), a title, document content (body), and an additional identifier (e.g. collection ID).
    Parameters title, content and the additional identifier are optional.
    """

    def __init__(self, id, title=None, content=None, doc_id=None):
        """
        Instantiates an instance of the Document.
        """
        self.id = id
        self.title = title
        self.content = content
        self.doc_id = id
        self.judgment = -1

        if self.doc_id:
            self.doc_id = doc_id

    def __str__(self):
        """
        Returns a string representation of a given instance of Document.
        """
        return "<Document ID: '{0}' Title: '{1}' Content: '{2}'".format(self.id, self.title, self.content)


class Session():
    def __init__(self, session_xml=None, session_xml_dataset=None, topic_num=None):
        self.topic_num = topic_num
        self.interactions = []
        self.session_num = None
        if session_xml != None:
            self.get_actual_session(session_xml, session_xml_dataset)

    def add_sim_interaction(self, query, results, clicked_results, action_code):
        inte = Interaction()
        inte.query = query
        inte.results = []
        for result in results:
            result_dict = {}
            result_dict["docid"] = result["docid"]
            result_dict["title"] = result["title"]
            result_dict["snippet"] = result["snippet"]
            if ("full_text_lm" in result):
                result_dict["full_text_lm"] = result["full_text_lm"]
            inte.results += [result_dict]
        inte.clicks = []
        for rank, (result_doc_id, click) in enumerate(clicked_results):
            if click == 1:
                inte.clicks += [(result_doc_id, rank + 1, 35)]
        if (action_code == 1):
            inte.type = "reformulate"
        else:
            inte.type = "page"
        self.interactions += [inte]

    def get_actual_session(self, session_xml, session_xml_dataset):
        self.topic_num = session_xml.getElementsByTagName("topic")[0].getAttribute("num")
        self.session_num = session_xml.getAttribute("num")
        interactions = session_xml.getElementsByTagName("interaction")
        self.interactions = []
        # print ("MAKING SESSION...")
        for inte_idx, interaction in enumerate(interactions):
            inte = Interaction()
            query_text = getText(interaction.getElementsByTagName("query")[0].childNodes)
            inte.query = query_text.split()
            results = interaction.getElementsByTagName("result")
            if session_xml_dataset == "Session_track_2012":
                cluewebid = "clueweb09id"
            else:
                cluewebid = "clueweb12id"
            clicks = []
            clicked_items = interaction.getElementsByTagName("click")
            for click in clicked_items:
                # print ("GETTING CLICK TIME...")
                # time = float(click.getAttribute("endtime")) - float(click.getAttribute("starttime"))
                time = 30
                # print ("GOT CLICK TIME...")
                if session_xml_dataset == "Session_track_2012":
                    rank = int(getText(click.getElementsByTagName("rank")[0].childNodes))
                    first_rank = int(results[0].getAttribute("rank"))
                    clicks += [
                        (getText(results[rank - first_rank].getElementsByTagName(cluewebid)[0].childNodes), rank, time)]
                else:
                    rank = int(getText(click.getElementsByTagName("rank")[0].childNodes))
                    first_rank = int(results[0].getAttribute("rank"))
                    clicks += [
                        (getText(results[rank - first_rank].getElementsByTagName(cluewebid)[0].childNodes), rank, time)]

            inte.clicks = clicks
            # print ("GETTING RESULTS...")
            inte.results = []
            for result in results:
                result_dict = {}
                if session_xml_dataset == "Session_track_2012":
                    cluewebid = "clueweb09id"
                else:
                    cluewebid = "clueweb12id"
                result_dict["docid"] = getText(result.getElementsByTagName(cluewebid)[0].childNodes)
                result_dict["title"] = getText(result.getElementsByTagName("title")[0].childNodes)
                result_dict["snippet"] = getText(result.getElementsByTagName("snippet")[0].childNodes)
                # result_dict["title"] = doc_collection[result_dict["doc_id"]]["title"]
                # result_dict["snippet"] = getText(result.getElementsByTagName("clueweb12id")[0].childNodes)
                inte.results += [result_dict]
            # print ("GOT RESULTS...")
            if session_xml_dataset == 'Session_track_2014':
                inte.type = interaction.getAttribute("type")
                self.interactions += [inte]
            else:
                if inte_idx - 1 != -1:
                    previous_query = self.interactions[-1].query
                    if (' '.join(inte.query) == ' '.join(previous_query)):
                        inte.type = 'page'
                    else:
                        inte.type = 'reformulate'
                else:
                    inte.type = 'reformulate'
            self.interactions += [inte]


class Interaction():
    def __init__(self):
        self.query = None
        self.results = []
        self.clicks = []
        self.type = None


def read_cb9_catb_ids():
    cb09_trec_ids = {}
    with open("../cb09_catb_trec_ids.txt", "r") as infile:
        trec_id = ""
        for line in infile:
            if (line.strip() == "c"):
                if trec_id != "":
                    cb09_trec_ids[trec_id] = 1
                trec_id = line.strip()
            else:
                trec_id += line.strip()
    print(list(cb09_trec_ids.keys())[:10])
    topic_rel_docs = read_judgements("Session_track_2012")
    filtered_rel_docs = {}
    for topic_num in topic_rel_docs:
        filtered_rel_docs[topic_num] = {}
        for rel_doc in topic_rel_docs[topic_num]:
            try:
                there = cb09_trec_ids[rel_doc]
                filtered_rel_docs[topic_num][rel_doc] = topic_rel_docs[topic_num][rel_doc]
                print("coming here")
            except:
                pass
    for topic_num in topic_rel_docs:
        print(topic_num, len(list(filtered_rel_docs[topic_num].keys())))
    pickle.dump(filtered_rel_docs, open("../Session_track_2012/filtered_judgements.pk", "wb"))
    return cb09_trec_ids


def new_word_probability(d):
    mu = 8
    return (float(mu) / float(d + mu)) * (float(1) / float(d))


def getText(nodelist):
    rc = []
    for node in nodelist:
        if node.nodeType == node.TEXT_NODE:
            rc.append(node.data)
    return ''.join(rc)


def longest_common_substring(S, T):
    m = len(S)
    n = len(T)
    counter = [[0] * (n + 1) for x in range(m + 1)]
    longest = 0
    lcs_set = []
    lcs_indices = []
    for i in range(m):
        for j in range(n):
            if S[i] == T[j]:
                c = counter[i][j] + 1
                counter[i + 1][j + 1] = c
                if c > longest:
                    lcs_set = []
                    lcs_indices = []
                    longest = c
                    lcs_indices.append((i - c + 1, i + 1, j - c + 1, j + 1))
                    lcs_set.append(S[i - c + 1:i + 1])
                elif c == longest:
                    lcs_indices.append((i - c + 1, i + 1, j - c + 1, j + 1))
                    lcs_set.append(S[i - c + 1:i + 1])
    # print (lcs_set)
    # print (lcs_indices)
    return lcs_set, lcs_indices


def join_the_content(snippets):
    content = snippets[0]
    for i in range(1, len(snippets)):
        # print ("Snippet %d: %s",i, snippets[i])
        lcs_set, lcs_indices = longest_common_substring(content, snippets[i])
        # print (lcs_set)
        # print (lcs_indices)
        if (lcs_set == []):
            # print ("Empty lcs set")
            # print ("Content:", content)
            # print ("Snippet: ", snippets[i])
            lcs_set = ""
            lcs_indices = (len(content), len(content), 0, 0)
        else:
            lcs_set, lcs_indices = lcs_set[0], lcs_indices[0]
        joined_text = ''
        if len(content[0:lcs_indices[0]]) > len(snippets[i][0:lcs_indices[2]]):
            joined_text = content[0:lcs_indices[0]]
        else:
            joined_text = snippets[i][0:lcs_indices[2]]
        joined_text += lcs_set
        if len(content[lcs_indices[1]:]) > len(snippets[i][lcs_indices[3]:]):
            joined_text += content[lcs_indices[1]:]
        else:
            joined_text += snippets[i][lcs_indices[3]:]
        content = joined_text
        # print ("joined_content:", content)
    # print ("Final content:" , content)
    return content


def create_snippet_dataset():
    session_data = xml.dom.minidom.parse('../Session_track_2014/sessiontrack2014.xml')
    results = session_data.getElementsByTagName("result")
    document_content = {}
    for result in results:
        clueweb_id = getText(result.getElementsByTagName("clueweb12id")[0].childNodes)
        title = getText(result.getElementsByTagName("title")[0].childNodes)
        content = getText(result.getElementsByTagName("snippet")[0].childNodes)
        content = (' '.join(content.replace(".", " ").split()))
        try:
            document_content[clueweb_id]["content"] += [content]
        except:
            try:
                document_content[clueweb_id]["content"] = [content]
            except:
                document_content[clueweb_id] = {}
                document_content[clueweb_id]["title"] = title
                document_content[clueweb_id]["content"] = [content]
    # print ("Num docs: ", len(document_content.keys()))
    i = 0
    for clueweb_id in document_content:
        i += 1
        if (i % 500 == 0):
            print(i)
        # print (document_content[clueweb_id]["content"])
        document_content[clueweb_id]["content"] = join_the_content(document_content[clueweb_id]["content"])
        # break
    # making line corpus for metapy
    return document_content


def get_stopwords():
    stopwords = {}
    with open(stopwords_file, "r") as infile:
        for line in infile:
            stopwords[line.strip()] = 1
    return stopwords


utils_preprocess_parameter = 0


def set_utils_preprocess_parameter(dataset):
    global utils_preprocess_parameter
    if dataset == "clueweb":
        utils_preprocess_parameter = 1
    return


def preprocess(text, stopword_file=None, lemmatizing=False, stemming=False, get_stemmed_reverse_words=False):
    text = text.replace('.', ' ')
    # text = text.replace('-', ' ')
    text = text.lower().translate(str.maketrans('', '', string.punctuation))
    new_text = text
    if stopword_file != None:
        stopwords = {}
        with open(stopword_file, "r") as infile:
            for line in infile:
                stopwords[line.strip()] = 1
        new_text = ""
        for word in text.split():
            try:
                stopwords[word]
            except:
                new_text += word + " "

    if (lemmatizing):
        words = new_text.lower().split()
        # lemmatizer = WordNetLemmatizer()
        lemmas = []
        for word in words:
            # if word not in stopwords:
            try:
                lemmas.append(lemmatized_words[word])
            except:
                lemma = lemmatizer.lemmatize(word, pos='v')
                lemma1 = lemmatizer.lemmatize(word, pos='n')
                if (lemma == word) and (lemma1 != word):
                    lemma = lemma1
                lemmas.append(lemma)
                lemmatized_words[word] = lemma
                # print ("coming here", word, lemma)
        new_text = ' '.join(lemmas)
    return new_text
    '''
    if get_stemmed_reverse_words:
        stemmed_reverse_words_topic = {}
    if (lemmatizing):
        words = new_text.lower().split()
        #lemmatizer = WordNetLemmatizer()
        stems = []
        for word in words:
            #if word not in stopwords:
            try:
                stems.append(stemmed_words[word])
                if get_stemmed_reverse_words:
                    try:
                        stemmed_reverse_words_topic[stem] += [word]
                    except:
                        stemmed_reverse_words_topic[stem] = [word]
            except:
                stem = ks.stem(word)
                stems.append(stem)
                stemmed_words[word] = stem
                if get_stemmed_reverse_words:
                    try:
                        stemmed_reverse_words_topic[stem] += [word]
                    except:
                        stemmed_reverse_words_topic[stem] = [word]                    
                try:
                    stemmed_reverse_words[stem] += [word]
                except:
                    stemmed_reverse_words[stem] = [word]                    
                #print ("coming here", word, lemma)
        new_text = ' '.join(stems)
    if get_stemmed_reverse_words:
        return new_text,stemmed_reverse_words_topic
    else:
        return new_text
    '''


pre_lemmatization_words = {}


def preprocess_2(text, stopword_file=None, lemmatizing=False):
    text = text.lower().translate(str.maketrans('', '', string.punctuation))
    new_text = text
    if stopword_file != None:
        stopwords = {}
        with open(stopword_file, "r") as infile:
            for line in infile:
                stopwords[line.strip()] = 1
        new_text = ""
        for word in text.split():
            try:
                stopwords[word]
            except:
                new_text += word + " "
    if (lemmatizing):
        words = new_text.lower().split()
        # lemmatizer = WordNetLemmatizer()
        lemmas = []
        for word in words:
            # if word not in stopwords:
            try:
                lemmas.append(lemmatized_words[word])
                the_lemma = lemmatized_words[word]
                the_word = word
            except:
                lemma = lemmatizer.lemmatize(word, pos='v')
                lemma1 = lemmatizer.lemmatize(word, pos='n')
                if (lemma == word) and (lemma1 != word):
                    lemma = lemma1
                lemmas.append(lemma)
                lemmatized_words[word] = lemma
                the_lemma = lemma
                the_word = word
            try:
                pre_lemmatization_words[the_lemma][the_word] += 1
            except KeyError:
                try:
                    pre_lemmatization_words[the_lemma][the_word] = 1
                except KeyError:
                    pre_lemmatization_words[the_lemma] = {}
                    pre_lemmatization_words[the_lemma][the_word] = 1
        new_text = ' '.join(lemmas)
    return new_text


def get_pre_lemmatization_words():
    return pre_lemmatization_words


def write_datasets(document_content):
    with open("../Session_track_2014/clueweb_snippet/clueweb_snippet.dat", "w") as outfile1:
        with open("../Session_track_2014/clueweb_snippet/clueweb_line_corpus_id_mapping.txt", "w") as outfile2:
            idx = 1
            for clueweb_id in document_content:
                outfile2.write(clueweb_id + "\t" + str(idx) + '\n')
                content = document_content[clueweb_id]["title"] + " " + document_content[clueweb_id]["content"]
                content = content.translate(str.maketrans('', '', string.punctuation))
                outfile1.write(content + '\n')
                idx += 1
    with open("../Session_track_2014/clueweb_snippet_data.txt", "w") as outfile1:
        for clueweb_id in document_content:
            outfile1.write("# " + clueweb_id + '\n')
            outfile1.write(document_content[clueweb_id]["title"] + '\n')
            outfile1.write(document_content[clueweb_id]["content"] + '\n')


def read_robust_data_collection():
    with open("../TREC_Robust_data/Robust_data_corpus.txt", "r") as infile:
        i = 0
        robust_doc_content = {}
        for line in infile:
            if (i % 2 == 0):
                doc_id = line.strip().split(" ")[1]
            else:
                robust_doc_content[doc_id] = " ".join(line.strip().split(" ")[17:])
            i += 1
            # if (i == 20000):
            #    break
    return robust_doc_content


def read_clueweb_snippet_data():
    document_content = {}
    document_content_2 = {}
    with open("../Session_track_2014/clueweb_snippet_data.txt", "r") as infile:
        line = infile.readline()
        while (line.strip() != ""):
            clueweb_id = line.strip().split("# ")[1]
            document_content[clueweb_id] = {}
            line = infile.readline()
            document_content[clueweb_id]["title"] = line.strip()
            line = infile.readline()
            document_content[clueweb_id]["content"] = line.strip()
            line = infile.readline()
            document_content_2[clueweb_id] = document_content[clueweb_id]["title"] + " " + document_content[clueweb_id][
                "content"]
        return (document_content, document_content_2)


stopwords = get_stopwords()


def get_bigram_word_lm(data_collection):
    bigram_word_frequencies = {}
    i = 0
    for docid in data_collection:
        words = data_collection[docid].split()
        bigram_sequences = [" ".join(words[i:i + 2]) for i in range(len(words) - 1)]
        for bigram_word in bigram_sequences:
            try:
                there = stopwords[bigram_word.split()[0]]
                there = stopwords[bigram_word.split()[1]]
            except KeyError:
                try:
                    bigram_word_frequencies[bigram_word] += 1
                except:
                    bigram_word_frequencies[bigram_word] = 1
        i += 1
        if (i % 1000) == 0:
            print(i)
    return bigram_word_frequencies


def get_unigram_language_model(data_collection):
    word_frequencies = {}
    word_binary_frequencies = {}
    i = 0
    for docid in data_collection:
        words = data_collection[docid].split()
        words_dict = {}
        for word in words:
            words_dict[word] = 1
            try:
                word_frequencies[word] += 1
            except:
                word_frequencies[word] = 1
        for word in words_dict:
            try:
                word_binary_frequencies[word] += 1
            except:
                word_binary_frequencies[word] = 1
        i += 1
        if (i % 1000) == 0:
            print(i)
    return (word_frequencies, word_binary_frequencies)


def compute_NDCG(predict_rel_docs, act_rel_doc_dict, cutoff, corpus_docids):
    print("COMPUTE NDCG: ")
    dcg = 0
    print(predict_rel_docs[:cutoff])
    for i in range(min(cutoff, len(predict_rel_docs))):
        doc_id = predict_rel_docs[i][0]
        if doc_id in act_rel_doc_dict:
            rel_level = act_rel_doc_dict[doc_id]
        else:
            rel_level = 0
        print(doc_id, rel_level)
        dcg += float(math.pow(2, rel_level) - 1) / float(np.log2(i + 2))
    ideal_sorted = {}
    for docid in act_rel_doc_dict:
        try:
            there = corpus_docids[docid]
            ideal_sorted[docid] = act_rel_doc_dict[docid]
        except KeyError:
            ideal_sorted[docid] = 0
    ideal_sorted = sorted(ideal_sorted.items(), key=lambda l: l[1], reverse=True)
    print(ideal_sorted)
    idcg = 0
    for i in range(min(cutoff, len(ideal_sorted))):
        idcg += float(math.pow(2, ideal_sorted[i][1]) - 1) / float(np.log2(i + 2))
    if idcg == 0:
        idcg = 1.0

    return float(dcg) / float(idcg)


def read_bigram_topic_lm(dataset):
    topic_descs = read_topic_descs(dataset)
    topic_bigram_ct = {}
    topic_unigram_ct = {}
    topic_bigram_prob = {}
    for topic_num in topic_descs:
        topic_bigram_ct[topic_num] = {}
        topic_desc = preprocess(topic_descs[topic_num], stopwords_file, lemmatizing=True)
        topic_desc = topic_desc.split()
        for i in range(len(topic_desc) - 1):
            try:
                topic_bigram_ct[topic_num][topic_desc[i] + " " + topic_desc[i + 1]] += 1
            except KeyError:
                topic_bigram_ct[topic_num][topic_desc[i] + " " + topic_desc[i + 1]] = 1
        topic_unigram_ct[topic_num] = {}
        for i in range(len(topic_desc)):
            try:
                topic_unigram_ct[topic_num][topic_desc[i]] += 1
            except KeyError:
                topic_unigram_ct[topic_num][topic_desc[i]] = 1
        topic_bigram_prob[topic_num] = {}
        for i in range(len(topic_desc) - 1):
            topic_bigram_prob[topic_num][topic_desc[i] + " " + topic_desc[i + 1]] = float(
                topic_bigram_ct[topic_num][topic_desc[i] + " " + topic_desc[i + 1]]) / float(
                topic_unigram_ct[topic_num][topic_desc[i]])
    return topic_bigram_ct, topic_unigram_ct, topic_bigram_prob


def read_bigram_topic_lm_trec_robust():
    topic_descs = read_trec_robust_topic_descs()
    topic_bigram_lm = {}
    for topic_num in topic_descs:
        topic_bigram_lm[topic_num] = {}
        topic_desc = preprocess(topic_descs[topic_num], stopwords_file, lemmatizing=True)
        topic_desc = topic_desc.split()
        for i in range(len(topic_desc) - 1):
            try:
                topic_bigram_lm[topic_num][topic_desc[i] + " " + topic_desc[i + 1]] += 1
            except KeyError:
                topic_bigram_lm[topic_num][topic_desc[i] + " " + topic_desc[i + 1]] = 1
    return topic_bigram_lm


'''
def read_judgements():
    topic_rel_docs = {}
    with open("../Session_Track_2014/judgments.txt", "r") as infile:
        for line in infile:
            topic_num,ignore,doc_id,rel = line.strip().split()
            try:
                there = topic_rel_docs[topic_num]
            except KeyError:
                topic_rel_docs[topic_num] = {}
            if (int(rel) > 0):
                topic_rel_docs[topic_num][doc_id] = int(rel)
    return topic_rel_docs

def read_topic_descs():
    topic_descs = {}
    topics_data = xml.dom.minidom.parse('../Session_Track_2014/topictext-890.xml')
    topics = topics_data.getElementsByTagName("topic")
    for topic in topics:
        topic_num = topic.getAttribute("num")
        topic_desc = getText(topic.getElementsByTagName("desc")[0].childNodes)
        topic_descs[topic_num] = topic_desc
    return topic_descs
'''


def select_sessions(topic_rel_docs, read_full=False):
    session_data = xml.dom.minidom.parse('../Session_track_2014/sessiontrack2014.xml')
    sessions = session_data.getElementsByTagName("session")
    ideal_user_sessions = []
    precise_user_sessions = []
    recall_user_sessions = []
    all_sessions = []
    for session in sessions:
        topic_num = session.getElementsByTagName("topic")[0].getAttribute("num")
        try:
            rel_docs = topic_rel_docs[topic_num]
        except:
            rel_docs = {}
        interactions = session.getElementsByTagName("interaction")
        result_doc_ids = []
        click_doc_ids = []
        for interaction in interactions:
            results = interaction.getElementsByTagName("result")
            try:
                clicked_items = interaction.getElementsByTagName("click")
                for click in clicked_items:
                    click_doc_ids += [getText(click.getElementsByTagName("docno")[0].childNodes)]
            except:
                pass
            for result in results:
                result_doc_ids += [getText(result.getElementsByTagName("clueweb12id")[0].childNodes)]

        precision = 0
        for doc_id in click_doc_ids:
            if doc_id in rel_docs:
                precision += 1
        if click_doc_ids != []:
            precision = float(precision) / float(len(click_doc_ids))
        else:
            pass
            # print ("no clicks")
        recall = 0
        recall_total = 0
        for doc_id in result_doc_ids:
            if doc_id in rel_docs:
                recall_total += 1
                if doc_id in click_doc_ids:
                    recall += 1
        if (recall_total != 0):
            recall = float(recall) / float(recall_total)
        else:
            pass
            # print ("no rel docs in result")
        if (precision == 1) and recall == 1:
            ideal_user_sessions += [session]
        if (precision == 1):
            # print ( click_doc_ids)
            precise_user_sessions += [session]
        if (recall == 1):
            recall_user_sessions += [session]
        all_sessions += [session]
        # print (session.getAttribute("num"), precision, recall)
        if (read_full == False):
            if (int(session.getAttribute("num")) > 101):
                break
    return ideal_user_sessions, precise_user_sessions, recall_user_sessions, all_sessions


def target_document_details(doc_collection, dataset):
    topic_descs = read_topic_descs(dataset)
    target_rel_docs = read_judgements(dataset)
    topic_rel_doc_details = {}
    all_documents = []
    for topic_num in target_rel_docs:
        docids = target_rel_docs[topic_num]
        target_doc_lm = {}
        target_doc_weighted_lm = {}
        documents = []
        documents_rels = []
        for docid in docids:
            try:
                content = doc_collection[docid]
                for word in content.split():
                    try:
                        target_doc_lm[word] += 1
                        target_doc_weighted_lm[word] += docids[docid] * 1
                    except KeyError:
                        target_doc_lm[word] = 1
                        target_doc_weighted_lm[word] = docids[docid] * 1
                documents += [content]
                documents_rels += [(docid, docids[docid])]
            except KeyError:
                pass
        topic_desc = preprocess(topic_descs[topic_num], stopwords_file, lemmatizing=True)
        documents += [topic_desc]
        all_documents += documents
        topic_lda_comm_dict = LDA_topics(documents, topic_num, 10)
        topic_rel_doc_details[topic_num] = [target_doc_lm, target_doc_weighted_lm, documents, documents_rels,
                                            topic_lda_comm_dict]
    all_topics_lda_comm_dict = LDA_topics(all_documents, "all", 100)
    topic_rel_doc_details["all_topics"] = [all_documents, all_topics_lda_comm_dict]
    return topic_rel_doc_details


from gensim.models.ldamodel import LdaModel
from gensim.corpora.dictionary import Dictionary


def LDA_topics(documents, topic_num, numtopics):
    texts = [d.split() for d in documents]
    common_dictionary = Dictionary(texts)
    # print ("doing this...")
    print('doing doc2bow')
    corpus = [common_dictionary.doc2bow(text) for text in texts]
    print("done this...dng LDA")
    lda = LdaModel(corpus, num_topics=numtopics)
    print("done LDA..")
    lda.save("../supervised_models/lda_models_trec_robust/trec_robust_" + str(topic_num) + "_ldamodel.model")
    print("done this...")
    return (lda, common_dictionary)


def make_preprocess_robust_data(dataset):
    topic_descs = read_topic_descs(dataset)
    topic_rel_docs = read_judgements(dataset)
    ideal_user_sessions, precise_user_sessions, recall_user_sessions, all_sessions = select_sessions(topic_rel_docs)
    robust_data_collection = read_robust_data_collection()
    clueweb_snippet_collection, clueweb_snippet_collection_2 = read_clueweb_snippet_data()
    print("started reading done")
    print("Num docs: ", len(robust_data_collection))
    for docid in clueweb_snippet_collection_2:
        try:
            there = robust_data_collection[docid]
            print("WEIRD: ", docid)
        except:
            robust_data_collection[docid] = clueweb_snippet_collection_2[docid]
    print("started reading done")
    i = 0
    robust_data_collection_wo_stopwords = {}
    for docid in robust_data_collection:
        robust_data_collection_wo_stopwords[docid] = preprocess(robust_data_collection[docid], stopwords_file,
                                                                lemmatizing=True)
        robust_data_collection[docid] = preprocess(robust_data_collection[docid], lemmatizing=True)
        i += 1
        if (i % 100000 == 0):
            print(i)
    with open('../TREC_Robust_data/robust_data_collection_preprocessed.json', 'w') as outfile:
        json.dump(robust_data_collection, outfile)
    with open('../TREC_Robust_data/robust_data_collection_preprocessed_stopwords.json', 'w') as outfile:
        json.dump(robust_data_collection_wo_stopwords, outfile)


def load_preprocess_robust_data():
    with open('../TREC_Robust_data/robust_data_collection_preprocessed.json', 'r') as infile:
        robust_data_collection = json.load(infile)
    with open('../TREC_Robust_data/robust_data_collection_preprocessed_stopwords.json', 'r') as infile:
        robust_data_collection_wo_stopwords = json.load(infile)
    return (robust_data_collection, robust_data_collection_wo_stopwords)


def read_trec_robust_judgements():
    document_id_to_num = {}
    document_num_to_id = {}
    topic_rel_docs = {}
    with open('../Luyu/data/document_id_mapping_gen_data.txt', 'r') as infile:
        for line in infile:
            doc_num, doc_id = line.strip().split()
            document_id_to_num[doc_id] = doc_num
            document_num_to_id[doc_num] = doc_id
    with open('../Luyu/data/qrel', 'r') as infile:
        i = 0
        for line in infile:
            queryid, _, docid, rel = line.strip().split()
            try:
                there = topic_rel_docs[queryid]
            except:
                topic_rel_docs[queryid] = {}
            try:
                if (int(rel) > 0):
                    topic_rel_docs[queryid][document_id_to_num[docid]] = int(rel)
            except KeyError:
                i += 1
                pass
    print("Num gone: ", i)
    return topic_rel_docs


def read_trec_robust_queries():
    query_details = {}
    text = ''
    with open('../Luyu/data/queriesFull.txt', 'r') as inputfile:
        for line in inputfile:
            if '<top>' in line:
                pass
            elif '<num>' in line:
                queryid = line.strip().split('<num>')[1].split('Number: ')[1]
                query_details[queryid] = []
            elif '<title>' in line:
                text = line.strip().split('<title>')[1]
                query_details[queryid] += [text]
            elif '<desc>' in line:
                text = ''
            elif '<narr>' in line:
                query_details[queryid] += [text]
                text = ''
            elif '</top>' in line:
                query_details[queryid] += [text]
                text = ''
            else:
                text += ' ' + line.strip()
    topic_descs = {}
    for queryid in query_details:
        topic_descs[queryid] = query_details[queryid][0]
    print("NUM topics: ", len(topic_descs))
    return topic_descs


def read_trec_robust_topic_descs():
    query_details = {}
    text = ''
    with open('../Luyu/data/queriesFull.txt', 'r') as inputfile:
        for line in inputfile:
            if '<top>' in line:
                pass
            elif '<num>' in line:
                queryid = line.strip().split('<num>')[1].split('Number: ')[1]
                query_details[queryid] = []
            elif '<title>' in line:
                text = line.strip().split('<title>')[1]
                query_details[queryid] += [text]
            elif '<desc>' in line:
                text = ''
            elif '<narr>' in line:
                query_details[queryid] += [text]
                text = ''
            elif '</top>' in line:
                query_details[queryid] += [text]
                text = ''
            else:
                text += ' ' + line.strip()
    topic_descs = {}
    for queryid in query_details:
        topic_descs[queryid] = query_details[queryid][1]
    print("NUM topics: ", len(topic_descs))
    return topic_descs


def trec_robust_target_document_details(doc_collection, dataset):
    topic_descs = read_topic_descs(dataset)
    target_rel_docs = read_judgements(dataset)
    topic_rel_doc_details = {}
    all_documents = {}
    i = 0
    for topic_num in target_rel_docs:
        i = i + 1
        docids = target_rel_docs[topic_num]
        target_doc_lm = {}
        target_doc_weighted_lm = {}
        documents = []
        documents_rels = []
        for docid in docids:
            try:
                content = doc_collection[docid]
                for word in content.split():
                    try:
                        target_doc_lm[word] += 1
                        target_doc_weighted_lm[word] += docids[docid] * 1
                    except KeyError:
                        target_doc_lm[word] = 1
                        target_doc_weighted_lm[word] = docids[docid] * 1
                documents += [content]
                documents_rels += [(docid, docids[docid])]
            except KeyError:
                pass
        topic_desc = preprocess(topic_descs[topic_num], stopwords_file, lemmatizing=True)
        documents += [topic_desc]
        for idx, (docid, rel) in enumerate(documents_rels):
            all_documents[docid] = documents[idx]
        all_documents["topic_desc_" + str(topic_num)] = documents[-1]
        topic_rel_doc_details[topic_num] = [target_doc_lm, target_doc_weighted_lm, documents, documents_rels]
        print("TOPIC NUM {} done".format(i))
    return topic_rel_doc_details, all_documents


def make_trec_robust_LDA_model():
    with open('../TREC_Robust_data/robust_data_collection_preprocessed_stopwords.json', 'r') as infile:
        robust_data_collection_wo_stopwords = json.load(infile)
    topic_rel_doc_details, all_documents_subset = trec_robust_target_document_details(
        robust_data_collection_wo_stopwords)
    all_documents = []
    for docid in robust_data_collection_wo_stopwords:
        all_documents += [robust_data_collection_wo_stopwords[docid]]
    start_time = time.time()
    print('lda topics')
    all_topics_lda_comm_dict = LDA_topics(all_documents_subset, "all", 100)
    print("TIME TAKEN: ", time.time() - start_time)
    topic_rel_doc_details["all_topics"] = [[], all_topics_lda_comm_dict]

    pickle.dump(topic_rel_doc_details, open("../TREC_Robust_data/trec_robust_topic_rel_doc_details.pk", "wb"))


def put_vector_dict(vector_dict, idx):
    try:
        vector_dict[idx]
        return vector_dict[idx]
    except KeyError:
        return 0


def language_model_m(topic):
    IN = Counter(topic.split())
    IN = {x: float(IN[x]) / float(sum(IN.values())) for x in IN}
    # print (topic, IN)
    return IN


def language_model_b(topic):
    IN = Counter(topic.split())
    max_freq = max(list(IN.values()))
    IN = {x: float(IN[x]) / float(max_freq) for x in IN}
    # print (topic, IN)
    return IN


def read_topic_descs(dataset):
    if (dataset == "Session_track_2012") or (dataset == "Session_track_2013"):
        topic_descs = {}
        topics_data = xml.dom.minidom.parse('../' + dataset + '/topics.xml')
        topics = topics_data.getElementsByTagName("topic")
        for topic in topics:
            topic_num = topic.getAttribute("num")
            topic_desc = getText(topic.getElementsByTagName("desc")[0].childNodes)
            topic_descs[topic_num] = topic_desc
        return topic_descs
    elif (dataset == "Session_track_2014"):
        topic_descs = {}
        topics_data = xml.dom.minidom.parse('../' + dataset + '/topictext-890.xml')
        topics = topics_data.getElementsByTagName("topic")
        for topic in topics:
            topic_num = topic.getAttribute("num")
            topic_desc = getText(topic.getElementsByTagName("desc")[0].childNodes)
            topic_descs[topic_num] = topic_desc
        return topic_descs


def get_topic_desc_background_model(dataset):
    topic_descs = read_topic_descs(dataset)
    words_binary_count = {}
    for topic_num in topic_descs:
        content = preprocess(topic_descs[topic_num], stopwords_file, lemmatizing=True)
        words = Counter(content.split())
        for word in words:
            try:
                words_binary_count[word] += 1
            except:
                words_binary_count[word] = 1
    return words_binary_count


def read_judgements(dataset):
    if dataset == "Session_track_2012":
        filename = "../" + dataset + "/qrels.txt"
        filtered_rel_docs = pickle.load(open("../Session_track_2012/filtered_judgements.pk", "rb"))
        return filtered_rel_docs
    elif dataset == "Session_track_2013":
        filename = "../" + dataset + "/qrels.txt"
    elif dataset == "Session_track_2014":
        filename = "../" + dataset + "/judgments.txt"
    topic_rel_docs = {}
    with open(filename, "r") as infile:
        for line in infile:
            topic_num, ignore, doc_id, rel = line.strip().split()
            try:
                there = topic_rel_docs[topic_num]
            except KeyError:
                topic_rel_docs[topic_num] = {}
            if (int(rel) > 0):
                topic_rel_docs[topic_num][doc_id] = int(rel)
    return topic_rel_docs

def read_all_judgements(dataset):
    if dataset == "Session_track_2012":
        filename = "../" + dataset + "/qrels.txt"
    elif dataset == "Session_track_2013":
        filename = "../" + dataset + "/qrels.txt"
    elif dataset == "Session_track_2014":
        filename = "../" + dataset + "/judgments.txt"
    topic_rel_docs = {}
    with open(filename, "r") as infile:
        for line in infile:
            topic_num, ignore, doc_id, rel = line.strip().split()
            try:
                there = topic_rel_docs[topic_num]
            except KeyError:
                topic_rel_docs[topic_num] = {}
            topic_rel_docs[topic_num][doc_id] = int(rel)
    return topic_rel_docs


def jaccard_similarity(list1, list2):
    # print (list1, list2, len(set(list1).intersection(set(list2))), len(set(list1).union(set(list2))))
    return float(len(set(list1).intersection(set(list2)))) / float(len(set(list1).union(set(list2))))


def main():
    # document_content = create_snippet_dataset()
    # write_datasets(document_content)
    # test_BM25_ranker_2()
    # read_bigram_topic_lm()
    '''
    print("started reading")
    robust_data_collection = read_robust_data_collection()
    clueweb_snippet_collection,clueweb_snippet_collection_2 = read_clueweb_snippet_data()
    print("started reading done")
    print ("Num docs: ", len(robust_data_collection))
    for docid in clueweb_snippet_collection_2:
        try:
            there = robust_data_collection[docid]
            print ("WEIRD: " , docid)
        except:
            robust_data_collection[docid] = clueweb_snippet_collection_2[docid]
    print ("Num docs: ", len(robust_data_collection))
    i = 0
    for docid in robust_data_collection:
        robust_data_collection[docid] = preprocess(robust_data_collection[docid],"../Session_track_2014/clueweb_snippet/lemur-stopwords.txt" ,lemmatizing = True)
        i += 1
        if (i%100000 == 0):
            print (i)

    topic_rel_doc_details = target_document_details(robust_data_collection)
    pickle.dump(topic_rel_doc_details,open("../TREC_Robust_data/topic_rel_doc_details.pk","wb"))
    #lda_model = LdaModel.load("../supervised_models/ldamodel.model")
    '''
    # make_preprocess_robust_data()
    # with open('../TREC_Robust_data/robust_data_collection_preprocessed_stopwords.json', 'r') as infile:
    #    robust_data_collection_wo_stopwords = json.load(infile)
    print("started reading")
    clueweb_snippet_collection, clueweb_snippet_collection_2 = read_clueweb_snippet_data()
    # print ("started reading done")
    # print ("Num docs: ", len(robust_data_collection))
    print("Num docs: ", len(clueweb_snippet_collection_2))
    i = 0
    for docid in clueweb_snippet_collection_2:
        clueweb_snippet_collection_2[docid] = preprocess(clueweb_snippet_collection_2[docid], stopwords_file,
                                                         lemmatizing=True)
        i += 1
        if (i % 100000 == 0):
            print(i)

    topic_rel_doc_details, all_documents_subset = trec_robust_target_document_details(clueweb_snippet_collection_2)
    print('lda topics')
    topic_rel_doc_details = pickle.load(open("../TREC_Robust_data/topic_rel_doc_details.pk", "rb"))
    (lda, comm_doct) = topic_rel_doc_details["all_topics"][1][0], topic_rel_doc_details["all_topics"][1][1]
    texts = []
    text_ids = []
    for d in all_documents_subset:
        texts += [all_documents_subset[d].split()]
        text_ids += [d]
    print('doing doc2bow')
    corpus = [comm_doct.doc2bow(text) for text in texts]
    print("done doc2bow")
    document_vectors = lda[corpus]
    print(document_vectors[0])
    target_document_vectors = {}
    print(len(document_vectors))
    j = 0
    list_100 = range(100)
    for idx, vect in enumerate(document_vectors):
        vector_dict = dict(vect)
        vector = [0] * 100
        # vector = [put_vector_dict(vector_dict, i) for i in list_100]
        for v in vector_dict:
            vector[v] = vector_dict[v]
        target_document_vectors[text_ids[idx]] = vector
        j += 1
        if (j % 100 == 0):
            print(j)
    pickle.dump(target_document_vectors, open("../TREC_Robust_data/target_doc_topic_vectors.pk", 'wb'))


if __name__ == "__main__":
    main();
    # print ("READING DOCS")
    # read_cb9_catb_ids()



