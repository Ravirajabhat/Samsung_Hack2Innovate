#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import chardet
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.externals import joblib

from sklearn.svm import LinearSVC
#from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import f1_score

class TrainMessageSubClassifer():
	def __init__(self,trainfile_name_path='TRAIN_info_SMS.csv'):
		self.pre_process(trainfile_name_path)

	def pre_process(self,trainfile_name_path):
		with open(trainfile_name_path, 'rb') as f:
			self.encode_result = chardet.detect(f.read()) 
			f.close()
		csvreader = pd.read_csv(trainfile_name_path,encoding=self.encode_result['encoding'])
		self.messages=np.array(csvreader['Message'])
		self.lables=np.array(csvreader['Label'])
	
	def train_SVMClassifier(self):
		#feature extraction and transformation
		classifier = Pipeline([
		('vectorizer', CountVectorizer(ngram_range=(1,4),min_df=1,max_df=0.5,analyzer='char')),
		('tfidf', TfidfTransformer()),
		('clf', OneVsRestClassifier(LinearSVC(random_state=101,loss='hinge')))])
		
		#training
		classifier.fit(self.messages, self.lables)
		
		#Validation
		self.cross_val_score(classifier)
		
		self.save(classifier,"message_sub_classifier.model")
	
	def f_score(self,actual_lable,predicted_lable):
		print "FScore : "+ f1_score(actual_lable, predicted_lable, average='micro')
	
	def cross_val_score(self,classifier):
		scores = cross_val_score(classifier, self.messages, self.lables, cv=5)
		print scores
	
	def save(self,model,model_file_path):
		joblib.dump(model, model_file_path)
		
if __name__ == '__main__':
	messagesubclassifier= TrainMessageSubClassifer()
	messagesubclassifier.train_SVMClassifier()

