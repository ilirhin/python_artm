import numpy as np


def get_theta(corpus, model):
    doc_topic = list()
    for doc in corpus:
        doc_topic.append(model.__getitem__(doc, eps=0))
    return np.array(doc_topic)[:, :, 1]


def get_phi(model):
    p = list()
    for topicid in range(model.num_topics):
        topic = model.state.get_lambda()[topicid]
        topic = topic / topic.sum()
        p.append(topic)
    return np.array(p)