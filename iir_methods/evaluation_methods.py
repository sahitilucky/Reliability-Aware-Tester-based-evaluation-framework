from utils import *
import json


class Session_operations():
    '''
    This class is to read Session track data sessions into session objects and store more info about the sessions .
    '''

    def __init__(self):
        self.dataset = None
        self.act_sessions = None
        self.act_session_topics = None
        return

    def get_actual_sessions(self, dataset):
        self.dataset = dataset
        xml_filename = ''
        if dataset == 'Session_track_2012':
            xml_filename = '../' + dataset + '/sessiontrack2012.xml'
        elif dataset == 'Session_track_2013':
            xml_filename = '../' + dataset + '/sessiontrack2013.xml'
        elif dataset == 'Session_track_2014':
            xml_filename = '../' + dataset + '/sessiontrack2014.xml'
        print(xml_filename)
        '''
        tree = ET.parse(xml_filename, ET.XMLParser(encoding='utf-8'))
        xml_data = tree.getroot()
        #here you can change the encoding type to be able to set it to the one you need
        xmlstr = ET.tostring(xml_data, encoding='utf-8', method='xml')

        data_dict = dict(xmltodict.parse(xmlstr))
        '''
        session_data = xml.dom.minidom.parse(xml_filename)
        all_sessions = session_data.getElementsByTagName("session")
        act_sessions = []
        for session in all_sessions:
            # print ('coming here')
            act_session = Session(session_xml=session, session_xml_dataset=self.dataset)
            # print (len(act_session.interactions))
            # print ([l.type for l in act_session.interactions])
            act_sessions += [act_session]
        self.act_sessions = act_sessions
        return self.act_sessions

    def get_reformulation_tuples(self, session_topics, dopreprocess=False):
        reformulated_queries = {}
        for topic_num in session_topics:
            reformulated_queries[topic_num] = []
            for session in session_topics[topic_num]:
                for idx, inte in enumerate(session.interactions):
                    if (inte.type == "reformulate") and (idx != 0):
                        if dopreprocess:
                            q1 = preprocess(" ".join(session.interactions[idx - 1].query), lemmatizing=True).split()
                            q2 = preprocess(" ".join(inte.query), lemmatizing=True).split()
                            if (q1 != q2):
                                reformulated_queries[topic_num] += [(q1, q2)]
                        else:
                            reformulated_queries[topic_num] += [(session.interactions[idx - 1].query, inte.query)]
        return reformulated_queries

    def get_session_topics(self, sessions):
        sessions_topics = {}
        for session in sessions:
            try:
                sessions_topics[session.topic_num] += [session]
            except:
                sessions_topics[session.topic_num] = [session]
        return sessions_topics

    def get_all_queries(self, sessions, dopreprocess=False):
        sessions_topics = self.get_session_topics(sessions)
        queries_list_topics = {}
        for topic_num in sessions_topics:
            queries_list_act = [list(filter(lambda l: l.type == "reformulate", session.interactions)) for session in
                                sessions_topics[topic_num]]
            queries_list_act = [interaction.query for session in queries_list_act for interaction in session]
            if dopreprocess:
                queries_list_act = [preprocess(" ".join(query), lemmatizing=True).split() for query in queries_list_act]
            queries_list_topics[topic_num] = queries_list_act
            # print (queries_list_topics[topic_num])
        return queries_list_topics

    def get_last_queries(self, sessions, dopreprocess=False):
        sessions_topics = self.get_session_topics(sessions)
        queries_list_topics = {}
        for topic_num in sessions_topics:
            queries_list_act = [list(filter(lambda l: l.type == "reformulate", session.interactions)) for session in
                                sessions_topics[topic_num]]
            all_last_queries = [session[-1].last_query for session in queries_list_act if len(session)>1]
            if dopreprocess:
                all_last_queries = [preprocess(" ".join(query), lemmatizing=True) for query in all_last_queries]
            queries_list_topics[topic_num] = all_last_queries
            # print (queries_list_topics[topic_num])
        return queries_list_topics

    def get_initial_queries(self):
        return

    def get_validation_set(self, sessions):
        queries_list_topics = self.get_all_queries(sessions, dopreprocess=True)
        validation_queries = {}
        for topic_num in queries_list_topics:
            queries_list = list(set([' '.join(query) for query in queries_list_topics[topic_num]]))
            if int(len(queries_list) / 5) > 0:
                validation_queries[topic_num] = random.sample(queries_list, int(len(queries_list) / 5))
                validation_queries[topic_num] = [query.split() for query in validation_queries[topic_num]]
        return validation_queries

    def get_validation_set_divisions(self, sessions):
        '''
        Making a training and vaidation queries set from all actual queries of Session track dataset
        :param sessions:
        :return:
        '''
        queries_list_topics = self.get_all_queries(sessions, dopreprocess=True)
        validation_set1 = {}
        validation_set2 = {}
        validation_set3 = {}
        validation_set4 = {}
        validation_set1_test = {}
        validation_set2_test = {}
        validation_set3_test = {}
        validation_set4_test = {}
        for topic_num in queries_list_topics:
            queries_list = list(set([' '.join(query) for query in queries_list_topics[topic_num]]))
            queries_list = np.random.permutation(queries_list)
            fold1 = list(queries_list[:int(float(len(queries_list)) / float(4))])
            fold2 = list(
                queries_list[int(float(len(queries_list)) / float(4)): 2 * int(float(len(queries_list)) / float(4))])
            fold3 = list(queries_list[
                         2 * int(float(len(queries_list)) / float(4)): 3 * int(float(len(queries_list)) / float(4))])
            fold4 = list(queries_list[3 * int(float(len(queries_list)) / float(4)):])
            validation_set1_test[topic_num] = []
            validation_set2_test[topic_num] = []
            validation_set3_test[topic_num] = []
            validation_set4_test[topic_num] = []
            if fold1 != []:
                validation_set1[topic_num] = [query.split() for query in fold1]
                validation_set2_test[topic_num] = [query.split() for query in fold1]
                validation_set3_test[topic_num] = [query.split() for query in fold1]
                validation_set4_test[topic_num] = [query.split() for query in fold1]
            if fold2 != []:
                validation_set2[topic_num] = [query.split() for query in fold2]
                validation_set1_test[topic_num] = [query.split() for query in fold2]
                validation_set3_test[topic_num] += [query.split() for query in fold2]
                validation_set4_test[topic_num] += [query.split() for query in fold2]
            if fold3 != []:
                validation_set3[topic_num] = [query.split() for query in fold3]
                validation_set1_test[topic_num] += [query.split() for query in fold3]
                validation_set2_test[topic_num] += [query.split() for query in fold3]
                validation_set4_test[topic_num] += [query.split() for query in fold3]
            if fold4 != []:
                validation_set4[topic_num] = [query.split() for query in fold4]
                validation_set1_test[topic_num] += [query.split() for query in fold4]
                validation_set2_test[topic_num] += [query.split() for query in fold4]
                validation_set3_test[topic_num] += [query.split() for query in fold4]

        return [(validation_set1, validation_set1_test), (validation_set2, validation_set2_test),
                (validation_set3, validation_set3_test), (validation_set4, validation_set4_test)]


class Evaluation_queries():
    def __init__(self, act_sessions=None, sim_sessions=None, candidate_queries=None, dataset=None):
        '''
        Takes actual session, simulated sessions , or only initial queries as input, this class computed different evaluation measures
        :param act_sessions:
        :param sim_sessions:
        :param candidate_queries:
        :param dataset:
        '''
        self.session_ops = Session_operations()
        if act_sessions != None:
            self.act_sessions = act_sessions
            self.act_session_topics = self.session_ops.get_session_topics(act_sessions)
        if sim_sessions != None:
            self.sim_sessions = sim_sessions
            self.sim_session_topics = self.session_ops.get_session_topics(sim_sessions)
        if candidate_queries != None:
            self.candidate_queries_topics = candidate_queries
        if (dataset != None):
            self.topic_rel_docs = read_judgements(dataset)
        return

    def query_similarity_evaluation(self, onlyqueries=False, top_k=None, validation_set=False,
                                    act_session_queries=None):
        '''
        jaccard similarity two ways: 1. for each simulated query find the best match in actual queries, for each actual query find the best match in simulated queries.
        :param onlyqueries:
        :param top_k:
        :param validation_set:
        :param act_session_queries:
        :return:
        '''
        if validation_set:
            queries_list_act_topics = act_session_queries
        else:
            queries_list_act_topics = self.session_ops.get_all_queries(self.act_sessions, dopreprocess=True)
        if (not onlyqueries):
            queries_list_sim_topics = self.session_ops.get_all_queries(self.sim_sessions)
        else:
            queries_list_sim_topics = {}
            for topic_num in self.candidate_queries_topics:
                queries_list_sim_topics[topic_num] = [query[0] for query in
                                                      self.candidate_queries_topics[topic_num][:int(top_k)]]
        # print (queries_list_act_topics.keys())
        # print (queries_list_sim_topics.keys())
        query_similarities = []
        query_similarities_2 = []
        similarities_list = {}
        for topic_num in queries_list_act_topics:
            if topic_num in queries_list_sim_topics:
                if (queries_list_sim_topics[topic_num] == []) or (queries_list_act_topics[topic_num] == []):
                    continue
                # print ('topic number: ', topic_num)
                # print ('actual queries: ', queries_list_act_topics[topic_num])
                # print ('simulated queries: ', queries_list_sim_topics[topic_num])
                simulated_to_act_sim, act_to_simulated_sim, simulated_to_act_sim_list, act_to_simulated_sim_list = self.jaccard_similarity(
                    queries_list_act_topics[topic_num], queries_list_sim_topics[topic_num])
                similarities_list[topic_num] = simulated_to_act_sim_list
                query_similarities += [float(simulated_to_act_sim + act_to_simulated_sim) / float(2)]
                query_similarities_2 += [simulated_to_act_sim]
        # print ("QUERY Similarities: ", float(sum(query_similarities))/float(len(query_similarities)))
        # print ("QUERY Similarities_2 : ", float(sum(query_similarities_2))/float(len(query_similarities_2)))
        avg_jaccard_similarity = float(sum(query_similarities)) / float(len(query_similarities))
        avg_jaccard_similarity_2 = float(sum(query_similarities_2)) / float(len(query_similarities_2))
        return avg_jaccard_similarity, avg_jaccard_similarity_2, similarities_list

    def jaccard_similarity(self, queries_list_act, queries_list_sim):
        simulated_to_act_sim = [[] for q1 in queries_list_sim]
        act_to_simulated_sim = []
        for q2 in queries_list_act:
            list1 = []
            for idx, q1 in enumerate(queries_list_sim):
                sim = jaccard_similarity(q1, q2)
                simulated_to_act_sim[idx] += [sim]
                list1 += [sim]
            act_to_simulated_sim += [list1]
        # simulated_to_act_sim = [[jaccard_similarity(q1, q2) for q2 in queries_list_act]for q1 in queries_list_sim]
        # act_to_simulated_sim = [[jaccard_similarity(q1, q2) for q1 in queries_list_sim]for q2 in queries_list_act]
        # print (simulated_to_act_sim, act_to_simulated_sim)
        simulated_to_act_sim_list = [max(x) for x in simulated_to_act_sim]
        act_to_simulated_sim_list = [max(x) for x in act_to_simulated_sim]
        simulated_to_act_sim = float(sum(simulated_to_act_sim_list)) / float(len(simulated_to_act_sim_list))
        act_to_simulated_sim = float(sum(act_to_simulated_sim_list)) / float(len(act_to_simulated_sim_list))
        return (simulated_to_act_sim, act_to_simulated_sim, simulated_to_act_sim_list, act_to_simulated_sim_list)

    def reformulation_similarity(self, top_k):
        act_reformulated_queries = self.session_ops.get_reformulation_tuples(self.act_session_topics, dopreprocess=True)
        sim_reformulated_queries = self.session_ops.get_reformulation_tuples(self.sim_session_topics)
        act_reformulated_queries_list = {}
        for topic_num in act_reformulated_queries:
            act_reformulated_queries_list[topic_num] = {}
            for (query1, query2) in act_reformulated_queries[topic_num]:
                try:
                    act_reformulated_queries_list[topic_num][" ".join(query1)] += [query2]
                except:
                    act_reformulated_queries_list[topic_num][" ".join(query1)] = [query2]
        wt_avg_similarities = []
        for topic_num in sim_reformulated_queries:
            act_query1s = act_reformulated_queries_list[topic_num].keys()
            for (query1, query2) in sim_reformulated_queries[topic_num]:
                similarities = [(q1, jaccard_similarity(q1.split(), query1)) for q1 in act_query1s]
                similarities = sorted(similarities, key=lambda l: l[1], reverse=True)[:top_k]
                if act_query1s == []:
                    continue
                wt_den = 0
                wt_avg_similarity = 0
                # print ("Query tuple: ")
                # print (query1,query2)
                for q1, sim_score in similarities:
                    # print (q1,sim_score)
                    # print (act_reformulated_queries_list[topic_num][q1])
                    similarities = [jaccard_similarity(query2, act_query2) for act_query2 in
                                    act_reformulated_queries_list[topic_num][q1]]
                    avg_similarity = float(sum(similarities)) / float(len(similarities))
                    wt_avg_similarity += sim_score * avg_similarity
                    wt_den += sim_score
                if wt_den == 0:
                    wt_avg_similarity = 0
                else:
                    wt_avg_similarity = float(wt_avg_similarity) / float(wt_den)
                # print (wt_avg_similarity)
                wt_avg_similarities += [wt_avg_similarity]
        avg_wt_avg_similarity = float(sum(wt_avg_similarities)) / float(len(wt_avg_similarities))
        # print ("avg_wt_avg_similarity: ", avg_wt_avg_similarity)
        return avg_wt_avg_similarity

    def NDCG_eval(self, formatted_results, topic_num, cutoff, corpus_ids=None):
        '''
        computing NDCG of the given ranked list with the relevant ranked list.
        :param formatted_results:
        :param topic_num:
        :param cutoff:
        :param corpus_ids:
        :return:
        '''
        predict_rel_docs = []
        for idx, result in enumerate(formatted_results):
            predict_rel_docs += [(result["docid"], idx)]
        act_rel_doc_dict = self.topic_rel_docs[topic_num]
        if corpus_ids == None:
            corpus_ids = act_rel_doc_dict
        return compute_NDCG(predict_rel_docs, act_rel_doc_dict, cutoff, corpus_ids)

    def QF_NDCG_act_eval(self, topic_num, cutoff, corpus_ids=None):
        '''
        Evaluate the search results present in the Session dataset sessions.
        :param topic_num:
        :param cutoff:
        :param corpus_ids:
        :return:
        '''
        ndcgs = []
        for session in self.act_session_topics[topic_num]:
            for interaction in session.interactions:
                ndcgs += [self.NDCG_eval(interaction.results, topic_num, cutoff, corpus_ids)]
        return ndcgs


if __name__ == "__main__":
    session_ops = Session_operations()
    act_sessions = session_ops.get_actual_sessions("Session_track_2012")
    save_dir = "../simulated_sessions/Session_track_2012"
    sim_sessions = None
    candidate_queries_topics = None

    methods = ['CCQF_0.1_0.765_0.135_2_noef_0.9_0.98_big_idx',
               'CCQF_0.1_0.765_0.135_2_noef_type2_nowsu_0.9_0.99_big_idx',
               'CCQF_0.1_0.765_0.135_2_noef_reform_0.9_0.9_big_idx',
               'CCQF_0.1_0.765_0.135_2_noef_reform_0.9_0.9_nowsu_big_idx',
               'CCQF_0.1_0.765_0.135_2_noef_reform_type2_0.9_0.9_big_idx',
               'CCQF_0.1_0.765_0.135_2_noef_no_reform_big_idx', 'CCQF_0.1_0.765_0.135_2_noef_0.9_0.95_big_idx',
               'CCQF_0.1_0.765_0.135_2_noef_noqf_0.9_0.95_big_idx', 'CCQF_0.1_0.765_0.135_2_noef_noqf_0.9_0.9_big_idx',
               'CCQF_0.1_0.765_0.135_2_noef_type2_nowsu_0.9_0.9_big_idx']
    # method = "QS3plus_big_idx"
    for method in methods:
        sim_sessions = pickle.load(open(os.path.join(save_dir, method + "_sessions.pk"), "rb"))
        candidate_queries_topics = pickle.load(open(os.path.join(save_dir, method + "_queries.pk"), "rb"))
        print("Method: ", method)
        ev = Evaluation_queries(act_sessions, sim_sessions, candidate_queries_topics)
        print("QR performance:")
        print(ev.query_similarity_evaluation(onlyqueries=False))
        print("QF performance top-1:")
        print(ev.query_similarity_evaluation(onlyqueries=True, top_k=1))
        print("QF performance top-5:")
        print(ev.query_similarity_evaluation(onlyqueries=True, top_k=5))
        print("QF performance top-10:")
        print(ev.query_similarity_evaluation(onlyqueries=True, top_k=10))
        print("Reformulation similarity: ")
        print(ev.reformulation_similarity(3))





