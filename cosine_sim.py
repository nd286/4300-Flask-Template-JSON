# cosine similarity of the query with all of the descriptions in our database
# make a dictionary that only has each flavor and description once to avoid dupicates of flavors/desc
# make the dictionary on keys in our json

# for key in item in json
# if key not in dictionary
# add key and description to dictionary


def accumulate_dot_scores(query_word_counts: dict, index: dict, idf: dict) -> dict:
    """Perform a term-at-a-time iteration to efficiently compute the numerator term of cosine similarity across multiple documents.
    query_word_counts: dict,
        A dictionary containing all words that appear in the query;
        Each word is mapped to a count of how many times it appears in the query.
        In other words, query_word_counts[w] = the term frequency of w in the query.
        You may safely assume all words in the dict have been already lowercased.

    index: the inverted index as above,

    idf: dict,
        Precomputed idf values for the terms.


    Returns
    =======

    doc_scores: dict
        Dictionary mapping from doc ID to the final accumulated score for that doc
    """
    # TODO-7.1
    doc_scores = {}
    for term, q_doc_freq in query_word_counts.items():
        if term in index and term in idf:
            idf_term = idf[term]
            query_weight = q_doc_freq * idf_term

            for doc_id, doc_freq in index[term]:
                doc_weight = doc_freq * idf_term
                if doc_id not in doc_scores:
                    doc_scores[doc_id] = 0.0
                doc_scores[doc_id] += query_weight * doc_weight
    return doc_scores


# compute final result from a4
def index_search(
    query: str,
    index: dict,
    idf,
    doc_norms,
    score_func=accumulate_dot_scores,
    tokenizer=helpers.treebank_tokenizer,
) -> List[Tuple[int, int]]:
    """Search the collection of documents for the given query

    Arguments
    =========

    query: string,
        The query we are looking for.

    index: an inverted index as above

    idf: idf values precomputed as above

    doc_norms: document norms as computed above

    score_func: function,
        A function that computes the numerator term of cosine similarity (the dot product) for all documents.
        Takes as input a dictionary of query word counts, the inverted index, and precomputed idf values.
        (See Q7)

    tokenizer: a TreebankWordTokenizer

    Returns
    =======

    results, list of tuples (score, doc_id)
        Sorted list of results such that the first element has
        the highest score, and `doc_id` points to the document
        with the highest score.

    Note:

    """

    # TODO-8.1
    query_lower = query.lower()
    query_tokenized = tokenizer.tokenize(query_lower)

    query_word_counts = {}
    for token in query_tokenized:
        if token in query_word_counts:
            query_word_counts[token] += 1
        else:
            query_word_counts[token] = 1

    doc_scores = score_func(query_word_counts, index, idf)

    query_norm_sq = 0.0
    for word, freq in query_word_counts.items():
        if word in idf:
            query_norm_sq += (freq * idf[word]) ** 2
    query_norm = math.sqrt(query_norm_sq)

    results = []

    for doc_id, dot_product in doc_scores.items():
        if doc_norms[doc_id] > 0 and query_norm > 0:
            cosine_similarity = dot_product / (doc_norms[doc_id] * query_norm)
        else:
            cosine_similarity = 0
        results.append((cosine_similarity, doc_id))

    results.sort(reverse=True, key=lambda x: x[0])
    return results
