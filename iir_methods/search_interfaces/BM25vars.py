import os
from base_interface import BaseSearchInterface
import logging
import math
from collections import Counter

#log = logging.getLogger('simuser.search_interfaces.IIRinterface')


class BM25vars(BaseSearchInterface):
    def __init__(self, k1= 1.2,b= 0.75, keepidf = True, engine=None):
        super(BM25vars, self).__init__()
        self.K1 = k1
        self.b = b
        self.keepidf = keepidf
        self.engine = engine

    def issue_query(self, query, topic_num, snippets=False, count=1000):
        # Based on the qrgument Ai, use the specific session interactions or don't
        if self.keepidf:
            filename = str(topic_num) + "_queries_"
            method = "okapi,k1=" + str(self.K1) + ",b=" + str(self.b)
            response = self.engine.search(query, filename=filename, snippets=snippets, count=count, topic_num=topic_num, method = method)
            self.response = response
        else:
            q_len = len(query.split(" "))
            term_dict = dict(Counter(query.split()))
            term_weights, term_responses, terms = self.term_wise_responses(term_dict, snippets, count, topic_num, "okapi,k1=1.2,b=0.75")
            term_weightsidf, term_responsesidf, termsidf = self.term_wise_responses(term_dict, snippets, count, topic_num, "okapi,k1=0.0001,b=0")
            docid_scores = {}
            for idx, term_response in enumerate(term_responses):
                q = term_weights[idx]
                docid_scores[terms[idx]] = {}
                for result in term_response:
                    docid_scores[terms[idx]][result["docid"]] = result["score"]
            docid_scoresidf = {}
            for idx, term_response in enumerate(term_responsesidf):
                q = term_weights[idx]
                docid_scoresidf[terms[idx]] = {}
                for result in term_response:
                    docid_scoresidf[terms[idx]][result["docid"]] = result["score"]

            docid_scoresnoidf = {}
            for term in docid_scores:
                docid_scoresnoidf[term]= {}
                for docid in docid_scores[term]:
                    docid_scoresnoidf[term][docid] = float(docid_scores[term][docid])/float(docid_scoresidf[term][docid])

            response = self.combine_term_responses_BM25(self, docid_scoresnoidf, term_dict, count, k3=1)

        return response

    def term_wise_responses(self, query_lm, snippets, count, topic_num, method ):
        term_weights = []
        term_responses = []
        terms = []
        filename = str(topic_num) + "_queries_"
        for word in query_lm:
            term_responses.append(
                self.engine.search(word, filename=filename, snippets=snippets, count=count, topic_num=topic_num, method = method))
            term_weights.append(query_lm[word])
            terms.append(word)
        return term_weights, term_responses, terms

    def combine_term_responses_BM25(self, docid_scoresnoidf, query_lm, count, k3=1):
        docid_scores = {}
        for term in docid_scoresnoidf:
            q = query_lm[term]
            for docid in docid_scoresnoidf[term]:
                try:
                    docid_scores[docid] += (float((k3 + 1) * q) / float(k3 + q)) * docid_scoresnoidf[term][docid]
                except KeyError:
                    docid_scores[docid] = (float((k3 + 1) * q) / float(k3 + q)) * docid_scoresnoidf[term][docid]
        reranked_results = sorted(docid_scores.items(), key=lambda l: l["score"], reverse=True)
        formatted_results = []
        for (docid, score) in reranked_results:
            formatted_result = {}
            formatted_result["docid"] = docid
            formatted_result["score"] = score
            formatted_results += [formatted_result]
        return formatted_results[:count]
        return response
