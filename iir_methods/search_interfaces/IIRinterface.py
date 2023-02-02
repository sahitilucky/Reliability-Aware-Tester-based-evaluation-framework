import os
from base_interface import BaseSearchInterface
import logging
import math
from collections import Counter


class IIRsearchInterface(BaseSearchInterface):
    def __init__(self, A1=0, A2 = 0, alpha = 0.5, engine = None, background_lm = None):
        super(IIRsearchInterface, self).__init__()
        self.A1 = A1
        self.A2 = A2
        self.alpha = alpha
        self.background_lm = background_lm
        self.engine = engine
    def issue_query(self, query, previous_queries, topic_num, snippets = False, count = 1000):
        #Based on the qrgument Ai, use the specific session interactions or don't
        if self.A1 == 1:
            response = self.issue_query_useA1_inte(query, previous_queries, self.alpha, topic_num, snippets, count)
        
        self.response = response
        return response

    def issue_query_useA1_inte(self, query, previous_queries, alpha, topic_num, snippets, count):
        #response = self.engine._search(query)
        #docids = self.get_docids(response)
        num_queries = len(previous_queries)
        k = num_queries + 1
        query_lm = {}
        for query_idx,q in enumerate(previous_queries): 
            q_len = len(q.split(" "))
            term_dict = dict(Counter(q.split()))
            for word in term_dict:
                try:    
                    there = query_lm[word]
                except KeyError:
                    query_lm[word] = 0
                query_lm[word] += (float(1)/float(k))*math.pow((1-alpha),k-1-query_idx)*alpha*(float(term_dict[word])/float(q_len))
        
        q_len = len(query.split(" "))
        term_dict = dict(Counter(query.split()))
        for word in term_dict:
            try:
                there = query_lm[word]
            except KeyError:
                query_lm[word] = 0
            query_lm[word] += (float(1)/float(k))*math.pow((1-alpha),0)*alpha*(float(term_dict[word])/float(q_len))

        total_sum = sum(query_lm.values())
        query_lm = {word:float(query_lm[word])/float(total_sum) for word in query_lm}
        print("Query weighting using previous queries: {}".format(query_lm))
        term_weights, term_responses, terms = self.term_wise_responses(query_lm, snippets, count, topic_num)

        response = self.combine_term_responses_BM25(term_responses, term_weights, count = count)

        self._last_response = response
        return response

    def get_docids(self, response):
        docnums = []
        for result in response.results:
            docnums += [result.whooshid]
        return set(docnums)

    def term_wise_responses(self, query_lm, snippets, count, topic_num):
        term_weights = []
        term_responses = []
        terms = []
        filename = str(topic_num) + "_queries_"
        for word in query_lm:
            term_responses.append(self.engine.search(word, filename = filename, snippets = snippets, count = count, topic_num = topic_num)  )
            term_weights.append(query_lm[word])
            terms.append(word)
        return term_weights, term_responses,terms

    def combine_term_responses_BM25(self, term_responses, term_weights, count, k3 = 1):
        docid_scores = {}
        for idx, term_response in enumerate(term_responses):
            q = term_weights[idx]
            for result in term_response:
                docid = result["docid"]
                try:
                    docid_scores[docid] += (float((k3+1)*q)/float(k3+q))*result["score"]
                except KeyError:
                    docid_scores[docid] = (float((k3+1)*q)/float(k3+q))*result["score"]
        reranked_results = sorted(docid_scores.items(), key=lambda l: l["score"], reverse=True)
        formatted_results = []
        for (docid,score) in reranked_results:
            formatted_result = {}
            formatted_result["docid"] = docid
            formatted_result["score"] = score
            formatted_results += [formatted_result]
        return formatted_results[:count]
        return response
