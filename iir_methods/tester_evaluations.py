import os
import time
import random
import pickle
import pyndri
import sys
#from User_model_new import *
from evaluation_methods import Evaluation_queries,Session_operations
from search_interfaces.IIRInterface import IIRsearchInterface
from search_interfaces.IIRA2_Weights import IIR2SearchInterface
from search_interfaces.BM25vars import BM25vars
from utils import *
stopwords_file = "../lemur-stopwords.txt"
old_stdout_target = sys.stdout
cb9_corpus_trec_ids = None
#start_time = time.time()
#cb9_corpus_trec_ids = read_cb9_catb_ids()
#print ("TIME TAKEN: ", time.time()-start_time)

class Indri_engine():
    def __init__(self, index_path, output_dir):
        self.index_path = index_path
        self.output_dir = output_dir
        self.all_selected_docids_topics = pickle.load(open(os.path.join(self.output_dir,"top1000_docs_last_query_docids.pk") , "rb"))
        return
    def search(self, query, filename = "", snippets = False, count = 1000, topic_num, method = None):
        working_set_docs = self.all_selected_docids_topics[topic_num]
        queries = {}
        queries["1"] = query
        inputname = os.path.join(self.output_dir, filename + "queries_input")
        outputname = os.path.join(self.output_dir, filename + "_results_output")
        if method is not None:
            make_runquery_file2(queries, inputname, self.index_path, snippets=snippets, count=count, working_set_docs = working_set_docs, method = method)
        else:
            make_runquery_file2(queries, inputname, self.index_path, snippets=snippets, count=count, working_set_docs=working_set_docs)
        start_time = time.time()
        os.system("../../indri-5.14/runquery/IndriRunQuery " + inputname + "> " + outputname)
        print('TIME TAKEN: ', time.time() - start_time)
        all_act_formatted_results = format_batch_results_trec_format(outputname)
        return all_act_formatted_results["1"]

def format_batch_results_trec_format(result_file):
    '''
    format indri results into results for the model: [id, score, snippet]
    :param result_file:
    :return:
    '''
    with open(result_file, 'rb') as infile:
        results = {}
        for line in infile:
            line = line.decode("utf-8", errors = 'ignore')
            if ("clueweb09-en" in line) or ("clueweb12-" in line):
                #print (line)
                queryid,_,docid,rank,score,_ = line.strip().split()
                try:
                    results[queryid] += [{"docid":docid,"score":score,"snippet":""}]
                except:
                    results[queryid] = [{"docid":docid,"score":score,"snippet":""}]
                previous_queryid = queryid
            else:
                results[previous_queryid][-1]["snippet"] += line.strip()
    all_formatted_results = {}
    for queryid in results:
        formatted_results = []
        for result in results[queryid]:
            formatted_result={}
            formatted_result["docid"] = result["docid"]
            formatted_result["title"] = ""
            formatted_result["snippet"] = result["snippet"]
            formatted_result["full_text_lm"] = ""
            formatted_result["score"] = result["score"]
            formatted_results += [formatted_result]
        all_formatted_results[queryid] = formatted_results
    return all_formatted_results

def make_runquery_file2(queries, filename, index_path, snippets = True, count = 200, workingsetdocs = None, method = "okapi,k1:1.2,b:0.75"):
    '''
    this makes a complex indri query acc. to used in Session track datasets
    :param queries:
    :param filename:
    :param snippets:
    :return:
    '''
    #queries = ['prime factor']*50
    with open(filename, 'w') as outfile:
        outfile.write('<parameters>\n')
        queryids = [int(i) for i in queries.keys()]
        queryids.sort()
        queryids = [str(i) for i in queryids]
        for queryid in queryids:
            q = queries[queryid]
            outfile.write("<query>\n")
            outfile.write("<type>indri</type>\n")
            outfile.write("<number>" + str(queryid) + "</number>\n")
            outfile.write("<text>#less(spam -130)")
            outfile.write("#combine(" + q + ")\n")
            outfile.write("</text>\n")
            if workingsetdocs is not None:
                for docid in workingsetdocs:
                    outfile.write("<workingSetDocno>" + docid + "</workingSetDocno>")
            outfile.write("</query>\n")
        outfile.write('<index>' + index_path + '</index>\n')
        if snippets:
            outfile.write('<printSnippets>True</printSnippets>\n')
        else:
            outfile.write('<printSnippets>False</printSnippets>\n')
        outfile.write('<trecFormat>true</trecFormat>\n')
        outfile.write('<count>'+str(count)+'</count>\n')
        outfile.write('<baseline>' + method + '</baseline>\n')
        outfile.write('</parameters>\n')


def actual_queries_results(dataset):
    '''
    Getting actual queries from session track sessions and running the queries with the constructed indri index and evaluate the resultant ranked list, also evaluate the earch result present in the session.
    :return:
    '''
    print ("actual_queries_performance")
    #dataset = "Session_track_2012"
    session_dir = "../simulated_sessions/"
    if not os.path.exists(session_dir): os.makedirs(session_dir)
    save_dir = os.path.join(session_dir, dataset)
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    session_ops = Session_operations()
    act_sessions = session_ops.get_actual_sessions(dataset)
    sys.stdout = open(os.path.join(save_dir, "Top1000_docs_Last_queries" + '_big_idx_out.txt'), 'w')
    topic_rel_docs = read_all_judgements(dataset)
    queries_list_act_topics = session_ops.get_last_queries(act_sessions, dopreprocess = True)
    queries_list_act_topics_set = {}
    for topic_num in queries_list_act_topics:
        queries_list_act_topics_set[topic_num] = list(set([" ".join(query) for query in queries_list_act_topics[topic_num]]))

    last_query_topics = {}
    for topic_num in queries_list_act_topics:
        last_query_topics[topic_num] = random.choice(queries_list_act_topics_set[topic_num])

    query_results = {}
    act_formatted_results = {}
    method = "top1000_docs_last_query"
    index_path = ""
    all_selected_docids_topics = {}
    for topic_num in last_query_topics:
        act_formatted_results[topic_num] = {}
        actual_queries = {}
        for idx,query in enumerate(last_query_topics[topic_num]):
            actual_queries[str(idx)] = query
        inputname = os.path.join(save_dir, method + "_" + str(topic_num) +  "_queries_input")
        outputname = os.path.join(save_dir, method + "_" + str(topic_num) +  "_results_output")
        make_runquery_file2(actual_queries, inputname, index_path, snippets = False, count = 1000)
        start_time = time.time()
        os.system("../../indri-5.14/runquery/IndriRunQuery " + inputname + "> " + outputname)
        print ('TIME TAKEN: ', time.time()-start_time)
        all_act_formatted_results = format_batch_results_trec_format(outputname)

        all_selected_docids = []
        for idx,query in enumerate(last_query_topics[topic_num]):
            try:
                there = all_act_formatted_results[str(idx)]
                for result in all_act_formatted_results[str(idx)]:
                    all_selected_docids += [result["docid"]]
            except KeyError:
                print("MISSED QUERIES: ", topic_num, query)
        try:
            there = topic_rel_docs[topic_num]
            all_selected_docids += [docid for docid in topic_rel_docs]
        except KeyError:
            pass

        all_selected_docids_topics[topic_num] = all_selected_docids

    pickle.dump(all_selected_docids_topics, open(os.path.join(save_dir, method + "_docids.pk") , "wb") )

    '''
    topic_ndcgs = {}
    for topic_num in topic_rel_docs:
        act_ndcgs_2 = []
        act_ndcgs50_2 = []
        for query in queries_list_act_topics[topic_num]:
            act_results_2 = act_formatted_results[topic_num][" ".join(query)]
            act_ndcgs_2 += [ev.NDCG_eval(act_results_2, topic_num, 10, cb9_corpus_trec_ids)]
            act_ndcgs50_2 += [ev.NDCG_eval(act_results_2, topic_num, 50, cb9_corpus_trec_ids)]
        act_ndcgs = ev.QF_NDCG_act_eval(topic_num, 10, cb9_corpus_trec_ids)
        act_ndcgs50 = ev.QF_NDCG_act_eval(topic_num, 50, cb9_corpus_trec_ids)
        topic_ndcgs[topic_num] = [float(sum(act_ndcgs))/float(len(act_ndcgs)), float(sum(act_ndcgs_2))/float(len(act_ndcgs_2)), float(sum(act_ndcgs50))/float(len(act_ndcgs50)), float(sum(act_ndcgs50_2))/float(len(act_ndcgs50_2)), act_ndcgs, act_ndcgs_2]
    act_ndcgs = sum([topic_ndcgs[topic_num][0] for topic_num in topic_ndcgs])/len(list(topic_ndcgs.keys()))
    act_ndcgs_2 = sum([topic_ndcgs[topic_num][1] for topic_num in topic_ndcgs])/len(list(topic_ndcgs.keys()))
    act_ndcgs50 = sum([topic_ndcgs[topic_num][2] for topic_num in topic_ndcgs])/len(list(topic_ndcgs.keys()))
    act_ndcgs50_2 = sum([topic_ndcgs[topic_num][3] for topic_num in topic_ndcgs])/len(list(topic_ndcgs.keys()))
    print ("ACT NDCG: ", act_ndcgs)
    print ("ACT NDCG 2: ", act_ndcgs_2)
    print ("ACT NDCG50: ", act_ndcgs50)
    print ("ACT NDCG50 2: ", act_ndcgs50_2)
    '''

def BM25_QH_tester_eval(dataset, selected_session_nums_topics, sessions_topics, doc_collection_lm_dist):
    '''
        Getting actual queries from session track sessions and running the queries with the constructed indri index and evaluate the resultant ranked list, also evaluate the earch result present in the session.
        :return:
    '''
    if dataset == "Session_track_2012":
        index_path = "../../Data/work/sahitil2/cb09-catb_index"
    else:
        index_path = "../../Data/work/sahitil2/clueweb12_index"
    print("actual_queries_performance")
    # dataset = "Session_track_2012"
    session_dir = "../simulated_sessions/"
    if not os.path.exists(session_dir): os.makedirs(session_dir)
    save_dir = os.path.join(session_dir, dataset)
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    #session_ops = Session_operations()
    #act_sessions = session_ops.get_actual_sessions(dataset)
    #sessions_topics = session_ops.get_session_topics(act_sessions)
    alpha = 0.8
    indri_engine = Indri_engine(index_path)
    qh_iir = IIRsearchInterface(A1=1, alpha=alpha, engine=indri_engine, background_lm = doc_collection_lm_dist)
    topic_session_results = {}
    for topic_num in sessions_topics:
        selected_session = None
        for session in sessions_topics[topic_num]:
            if session.session_num == selected_session_nums_topics[topic_num]:
                selected_session = session
                break
        previous_queries = []
        session_results = []
        for inte in selected_session:
            if inte.type == "reformulate":
                results = qh_iir.issue_query(inte.query, previous_queries, topic_num =  topic_num, snippets = False, count = 1000)
                session_results  += [(results, inte.query)]
                previous_queries += [inte.query]
        topic_session_results[topic_num] = session_results
    pickle.dump(topic_session_results, open(os.path.join(save_dir, "QH_IIR_tester_session_rs_" + str(alpha) + ".pk") , "wb")  )


def select_sessions(dataset):
    session_ops = Session_operations()
    act_sessions = session_ops.get_actual_sessions(dataset)
    sessions_topics = session_ops.get_session_topics(act_sessions)
    selected_session_nums_topics = {}
    for topic_num in sessions_topics:
        session_intes = [list(filter(lambda l: l.type == "reformulate", session.interactions)) for session in sessions_topics[topic_num]]
        sessions_lens2 = [s for s in session_intes if len(s) == 1]
        selected_session = random.choice(sessions_lens2)
        selected_session_nums_topics[topic_num] = selected_session.session_num
    return sessions_topics,selected_session_nums_topics

def get_background_lm():
    index = pyndri.Index('../../ClueWeb09_catB_sample_index')
    token2id, id2token, id2df = index.get_dictionary()
    id2tf = index.get_term_frequencies()
    doc_collection_lm_dist = {}
    for tokenid in id2tf:
        doc_collection_lm_dist[id2token[tokenid]] = id2tf[tokenid]
    total_noof_words = float(sum(doc_collection_lm_dist.values()))
    doc_collection_lm_dist = {token: float(doc_collection_lm_dist[token]) / float(total_noof_words) for token in
                              doc_collection_lm_dist}
    return doc_collection_lm_dist


def BM25_CH_tester_eval(dataset, selected_session_nums_topics, sessions_topics, doc_collection_lm_dist):
    '''
        Getting actual queries from session track sessions and running the queries with the constructed indri index and evaluate the resultant ranked list, also evaluate the earch result present in the session.
        :return:
    '''
    if dataset == "Session_track_2012":
        index_path = "../../Data/work/sahitil2/cb09-catb_index"
    else:
        index_path = "../../Data/work/sahitil2/clueweb12_index"


    print("actual_queries_performance")
    # dataset = "Session_track_2012"
    session_dir = "../simulated_sessions/"
    if not os.path.exists(session_dir): os.makedirs(session_dir)
    save_dir = os.path.join(session_dir, dataset)
    if not os.path.exists(save_dir): os.makedirs(save_dir)

    alpha = 0.8
    indri_engine = Indri_engine(index_path)
    ch_iir = IIR2SearchInterface(A2=1, alpha=alpha, engine=indri_engine, background_lm = doc_collection_lm_dist)
    topic_session_results = {}

    for topic_num in sessions_topics:
        selected_session = None
        for session in sessions_topics[topic_num]:
            if session.session_num == selected_session_nums_topics[topic_num]:
                selected_session = session
                break
        i = 0
        all_examined_results = []
        session_results = []
        for inte in selected_session:
            if inte.type == "reformulate":
                results = ch_iir.issue_query(inte.query, all_examined_results, topic_num =  topic_num, snippets = False, count = 1000)
                session_results  += [(results, inte.query)]
            for click in inte.clicks:
                clicked_result = inte.results[click[1]]
                all_examined_results += [clicked_result]
            i = i + 1
        topic_session_results[topic_num] = session_results
    pickle.dump(topic_session_results, open(os.path.join(save_dir, "CH_IIR_tester_session_rs_" + str(alpha) + ".pk") , "wb")  )



def BM25_vars_tester_eval(dataset, selected_session_nums_topics, sessions_topics, doc_collection_lm_dist):
    '''
        Getting actual queries from session track sessions and running the queries with the constructed indri index and evaluate the resultant ranked list, also evaluate the earch result present in the session.
        :return:
    '''
    if dataset == "Session_track_2012":
        index_path = "../../Data/work/sahitil2/cb09-catb_index"
    else:
        index_path = "../../Data/work/sahitil2/clueweb12_index"

    print("actual_queries_performance")
    # dataset = "Session_track_2012"
    session_dir = "../simulated_sessions/"
    if not os.path.exists(session_dir): os.makedirs(session_dir)
    save_dir = os.path.join(session_dir, dataset)
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    indri_engine = Indri_engine(index_path)
    k1, b, keepidf = 1.2, 0.75, True
    k1, b, keepidf = 1.2, 0.75, False
    k1, b, keepidf = 0.0001, 0, True
    bm25vars = BM25vars(k1=k1, b=b, keepidf=keepidf, engine=indri_engine, background_lm = doc_collection_lm_dist)
    topic_session_results = {}
    for topic_num in sessions_topics:
        selected_session = None
        for session in sessions_topics[topic_num]:
            if session.session_num == selected_session_nums_topics[topic_num]:
                selected_session = session
                break
        session_results = []
        for inte in selected_session:
            if inte.type == "reformulate":
                results = bm25vars.issue_query(inte.query, topic_num =  topic_num, snippets = False, count = 1000)
                session_results  += [(results, inte.query)]
        topic_session_results[topic_num] = session_results
    pickle.dump(topic_session_results, open(os.path.join(save_dir, "TF_K1_"+ str(k1) + "_keepidf_" + str(keepidf) + "_tester_session_rs_" + str(alpha) + ".pk") , "wb")  )


if __name__ == "__main__":
    dataset = "Session_track_2012"
    actual_queries_results(dataset)
    #select_sessions(dataset)
    #doc_collection_lm_dist = get_background_lm()
