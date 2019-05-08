import os
from sklearn.metrics.pairwise import pairwise_distances_argmin
from chatterbot import ChatBot
from chatterbot.trainers import ChatterBotCorpusTrainer
from utils import *
from extra import predict
import pandas as pd
sample_size=100



class ThreadRanker(object):
    def __init__(self, paths):
  
        self.category_embeddings_folder = paths['CATEGORY_EMBEDDINGS_FOLDER']
        self.scheme_embeddings_folder = paths['SCHEME_EMBEDDINGS_FOLDER']
        self.word_embeddings,self.embeddings_dim=load_embeddings(paths['WORD_EMBEDDINGS'])
        

    def __load_embeddings_by_category(self, category_name):
        
        embeddings_path = os.path.join(self.category_embeddings_folder, category_name + ".pkl")
        category_ids, category_embeddings = unpickle_file(embeddings_path)
        return category_ids, category_embeddings

    def __load_embeddings_by_scheme(self, scheme_name):

        embeddings_path = os.path.join(self.scheme_embeddings_folder, scheme_name + ".pkl")
        scheme_ids, scheme_embeddings = unpickle_file(embeddings_path)
        return scheme_ids, scheme_embeddings

    def get_best_category(self, question, category_name):
        
        category_ids, category_embeddings = self.__load_embeddings_by_category(category_name)

        
        question_vec = question_to_vec(question, self.word_embeddings, self.embeddings_dim).reshape(1,-1)
      
        best_category = pairwise_distances_argmin(question_vec, category_embeddings)[0]
        
        return category_ids[best_category]
    def get_best_scheme(self, question, scheme_name):
       
        scheme_ids, scheme_embeddings = self.__load_embeddings_by_scheme(scheme_name)

        
        question_vec = question_to_vec(question, self.word_embeddings, self.embeddings_dim).reshape(1,-1)
        best_scheme = pairwise_distances_argmin(question_vec, scheme_embeddings)[0]
        
        return scheme_ids[best_scheme]


class DialogueManager(object):
    def __init__(self, paths):
        print("Loading resources...")

        # Intent recognition:
        self.intent_recognizer = unpickle_file(paths['INTENT_RECOGNIZER'])
        self.tfidf_vectorizer = unpickle_file(paths['TFIDF_VECTORIZER'])
        self.ANSWER_TEMPLATE = 'Is it the scheme you are talking about %s, may be this helps you:  \n %s: \n%s \n Thanks for using chatbot'
        # Goal-oriented part:
        
        self.thread_ranker = ThreadRanker(paths)
        

    def create_chitchat_bot(self):
        """Initializes self.chitchat_bot with some conversational model."""

      
        self.chitchat_bot= ChatBot("Training Example")
        self.chitchat_bot.set_trainer(ChatterBotCorpusTrainer)

        self.chitchat_bot.train("chatterbot.corpus.english")
        
    

    def generate_answer(self, question):
    
        prepared_question = text_prepare(question)
        
        features = self.tfidf_vectorizer.transform([prepared_question])
        intent = self.intent_recognizer.predict(features)
         
        if (intent == 'dialogue'):       
            response = self.chitchat_bot.get_response(question)
            return response
        else:        
            scheme = predict(prepared_question)
            questions_df = pd.read_csv('data/questions.csv', sep=',')
            thread_id=0
            if scheme in questions_df['scheme'].unique():
                thread_id = self.thread_ranker.get_best_scheme(prepared_question, scheme)
            elif scheme in questions_df['category'].unique():
                    thread_id = self.thread_ranker.get_best_category(prepared_question, scheme)
            else:
                thread_id = self.thread_ranker.get_best_scheme(prepared_question, scheme)

            return self.ANSWER_TEMPLATE % (scheme,questions_df.iloc[thread_id,3],questions_df.iloc[thread_id,4])

