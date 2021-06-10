
# coding: utf-8

import pandas as pd
import numpy as np 
import os

import re
from nltk.tokenize import word_tokenize 
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from sklearn.metrics import f1_score, accuracy_score

#### BASE CLASS
  
class Base:
    """Base class that houses common utilities for reading in test data
    and calculating model accuracy and F1 scores.
    """
    def __init__(self,train_path,test_path,mix ):
        print("parent class")
        self.train_path = train_path
        self.test_path = test_path
        self.mix = mix
        
    #### Get Training Data
    
    def get_train_data(self, train_path, mix)-> pd.DataFrame:
        '''Function to fetch train data and create a DataFrame for Training a Model'''
        print("Get Training Data")
    
        text = []
        rating = []
        try:
            ## Get Positive Label Train Data
            for filename in os.listdir(train_path+"pos"):
                pos_data_train = open(train_path+"pos/"+filename, 'r' , encoding="ISO-8859-1").read()
                text.append(pos_data_train)
                rating.append("1")
            ## Get Negative Label Train Data
            for filename in os.listdir(train_path+"neg"):
                neg_data_train = open(train_path+"neg/"+filename, 'r' , encoding="ISO-8859-1").read()
                text.append(neg_data_train)
                rating.append("0")
            
            train_dataset = list(zip(text,rating))    
        
            ## Shuffle Data
            if mix:
                np.random.shuffle(train_dataset)
        
            ## Create a Datafrane
            df_train = pd.DataFrame(data = train_dataset, columns=['Review', 'Rating'])
            return(df_train)
    
        except Exception as e:
            print("There is an eror in get_train_data: ", e)
            pass
   
    #### Get Test Data
    
    def get_test_data(self,test_path) -> pd.DataFrame:
        '''Function to fetch Test data and create a DataFrame Testing Accracy of the Model'''
        print("Get Test Data")

        text = []
        rating = []
        try:
            ## Get Positive Label Train Data
            for filename in os.listdir(test_path+"pos"):
                pos_data = open(test_path+"pos/"+filename, 'r' , encoding="ISO-8859-1").read()
                text.append(pos_data)
                rating.append("1")
            ## Get Negative Label Train Data
            for filename in os.listdir(test_path+"neg"):
                neg_data = open(test_path+"neg/"+filename, 'r' , encoding="ISO-8859-1").read()
                text.append(neg_data)
                rating.append("0")
            
            test_dataset = list(zip(text,rating)) 
            
            ## Create a Datafrane
            df_test  = pd.DataFrame(data = test_dataset, columns=['Review', 'Rating'])
            return(df_test)
        except Exception as ex:
            print("There is an eror in get_test_data: ", ex)
       
     #### Clean the Text - Data Preprocessing
    
    def clean_data(self,text) -> pd.DataFrame:
        '''Function to clean Review Text'''
        print("Data Preprocessing")
        stemmer= PorterStemmer()
        try:
            # Preprocessing
            # convert to lower case
            clean_text =  text.str.lower()
            # Remove Numbers
            clean_text = clean_text.str.replace('\d+', '')
            # Remove trailing spaces
            clean_text = clean_text.str.strip()
            # Remove Punctuations
            clean_text = clean_text.str.replace('[^\w\s]',' ')
            # Remove <br>
            clean_text = clean_text.str.replace('br', '')
            # Remove extra space in between words
            clean_text = clean_text.str.replace(' +', ' ')
            # Remove Numbers
            clean_text = clean_text.str.replace('\d+', '')
            # remove stop words
            stop = stopwords.words('english')
            stop.extend(["movie","movies","film","one"])
            clean_text =   clean_text.apply(lambda x: " ".join(x for x in x.split() if x not in stop ))
            # Stemming
            #clean_text =   clean_text.apply(lambda x: " ".join(stemmer.stem(x) for x in x.split() ))


            return clean_text
        except Exception as e:
            print("In Exception of clean_data: ", e)
            return None  
        
        #### Tokenize the Reviews
   
    def tokenization(df_reviews):
        '''Tokenize the Review Text'''
        print(" Tokenize the Reviews")
        # Tokenize 
        df_reviews["Clean_Review"] = df_reviews["Clean_Review"].astype(str).str.strip().str.split('[\W_]+')
        # Initialize a CountVectorizer object: count_vect
        count_vec = CountVectorizer(analyzer='word',tokenizer=lambda doc: doc, lowercase=False, max_df = 0.70, min_df = 100)
        words_vec = count_vec.fit(df_reviews["Clean_Review"])
        bag_of_words = words_vec.transform(df_reviews["Clean_Review"])
        tokens = count_vec.get_feature_names()
        df_words = pd.DataFrame(data=bag_of_words.toarray(),columns=tokens)
        return df_words

#### CHILD CLASS
     
    
class LogisticRegressionSentiment(Base):
    """Predict fine-grained sentiment scores using a sklearn Logistic Regression pipeline."""
    def __init__(self, train_path, test_path, mix):
        super().__init__(train_path, test_path, mix)
        print("Starting LogisticRegressionSentiment Model ")
        
        from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        self.pipeline = Pipeline(
            [
                ('vect', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
                ('clf', LogisticRegression(solver='liblinear', multi_class='auto')),
            ]
        )
     
        #### Get Accuracy Score
    
    def accuracy_score(self,Rating,Pred_Rating)-> pd.DataFrame:
        '''Get Accuracy Score'''
        print("Get accuracy score")
        score = accuracy_score(Rating, Pred_Rating)
        return score
    
    def predict(self,train_path,mix) -> pd.DataFrame:
        "Train model using sklearn pipeline"
        print("Predict Sentiment ")
        train_df = self.get_train_data(train_path, mix)
        train_df["Clean_Review"] = self.clean_data(train_df["Review"])
        learner = self.pipeline.fit(train_df["Clean_Review"], train_df["Rating"])
        # Predict class labels using the learner and output DataFrame
        test_df = self.get_test_data(test_path)
        #test_df["Clean_Review"] = self.clean_data(test_df["Review"])
        test_df['Pred_Rating'] = learner.predict(test_df['Review'])
        score = self.accuracy_score(test_df['Rating'],test_df['Pred_Rating'])
        print("Accuracy of the Logictic Regression Model is:  ",score)
        return learner

#### Main

    
if __name__ == "__main__":
        print("...Start Building Logistic Regression Model...")
        train_path = "test/aclImdb/train/"
        test_path = "test/aclImdb/test/"
        mod_lr = LogisticRegressionSentiment(train_path,test_path,True)
        mod = mod_lr.predict(train_path,True)
        v = 0
        while True:
           
            print(" ")
            input_sen = str(input("Text: "))
            input_sen = [input_sen]
            print(" ")
            val = mod.predict(input_sen)
            print(" ")
            #print(val)
            if val == '1':
                print("The emotion of the above statement is: Positive")
            else:
                print("The emotion of the above statement is: Negative")
            print(" ")
            key_in = input("Do You want to try another sentence? (yes to try again or no to stop): ").lower()
            print(" ")
            if(key_in == "no"):
                print("Thank You!")
                break







