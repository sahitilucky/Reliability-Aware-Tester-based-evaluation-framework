import os
import logging
from IIRinterface import IIRsearchInterface
import math
from collections import Counter


# = logging.getLogger('simuser.search_interfaces.IIR2interface')

class IIR2SearchInterface(IIRsearchInterface):
    def __init__(self, whoosh_index_dir, A1=0, A2=0, alpha = 0.8, engine = None, background_lm = None):
        super(IIR2SearchInterface, self).__init__(A1, A2, alpha = alpha, engine = engine, background_lm = background_lm)
        self.background_lm = background_lm


    def issue_query(self, query, all_examined_results, topic_num, snippets = False, count = 1000):
        original_query = query 
        if (self.A2 == 1 and self.A1==0):
            #response, docids = self.get_baseline_results(query)
            relevant_terms = self.clicked_documents_terms(all_examined_results)
            query_lm = self.expand_query_weights_useA2_inte(query, relevant_terms, n = 5, current_query_weight=self.alpha)
            print("QUERY EXPANSION USING RELEVANCE FEEDBACK: {}".format(query_lm))
            term_weights, term_responses, terms = self.term_wise_responses(query_lm, snippets, count, topic_num)
            term_weights_string = ", ".join([terms[idx]+" "+str(term_weights[idx]) for idx in range(len(term_weights))])
            print("CLICK RELEVANCE FEEDBACK IIR2 interface: {}".format(term_weights_string))
            response = self.combine_term_responses_BM25(term_responses, term_weights, count = count)
        self.response = response
        return response

    def clicked_documents_terms(self, all_examined_results):
        clicked_documents = all_examined_results
        meta_document = ""
        for result in clicked_documents:
            content = result["title"] + " " + result["snippet"]
            meta_document +=  " " + content
        reldoc_term_dict = dict(Counter(meta_document.split(" ")))
        total_len = len(meta_document.split(" "))
        reldoc_language_model = {reldoc_term_dict[term]/len(total_len) for term in reldoc_term_dict}
        KLdiv_terms = {}
        for word in reldoc_term_dict:
            try:
                there = self.background_lm[word]
            except KeyError:
                KLdiv_terms[word] = reldoc_language_model[word]*math.log(float(reldoc_language_model[word])/float(self.background_lm[word]), 2)
        relevant_terms = sorted(KLdiv_terms.items(), key = lambda l :l[1], reverse=True)
        relevant_terms = [term[0] for term in relevant_terms]
        return relevant_terms

    def expand_query_weights_useA2_inte(self, query, relevant_terms, n = 5, current_query_weight = 0.8): 
        alpha = current_query_weight
        #alpha = 0.5
        query_lm = {}
        for term in relevant_terms[:n]:
            try:
                query_lm[term] += (1-alpha)
            except:
                query_lm[term] = (1-alpha)

        for term in query.terms.split(" "):
            try:
                query_lm[term] += alpha
            except:
                query_lm[term] = alpha

        total_sum = sum(query_lm.values())
        query_lm = {word:float(query_lm[word])/float(total_sum) for word in query_lm}
        
        return query_lm





    