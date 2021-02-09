# "Semantic Analysis for IMDB Movie Reviews" project by:
# Ahmet Sergen Boncuk
# Murat Mert Yurdakul

import numpy as np
import pandas as pd
import os
import copy
import re
from spellchecker import SpellChecker
from nltk.stem import PorterStemmer
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics 
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from timeit import default_timer as timer   
import seaborn as sn
import matplotlib.pyplot as plt; plt.rcdefaults()
import matplotlib.pyplot as plt
#from nltk.tokenize import sent_tokenize, word_tokenize

targetPositiveFilesPath = "./pos100"	#"<path/file_name>" #Targeted file path which includes all positive comments. (root path is current directory)
targetNegativeFilesPath = "./neg100"	#"<path/file_name>"	#Targeted file path which includes all negative comments. (root path is current directory)
# Using lesser amount of comment is required to process and get the results faster

stop_words = {"ourselves", "hers", "between", "yourself", "but", "again", "there", "about", "once", "during", 
"out", "very", "having", "with", "they", "own", "an", "be", "some", "for", "do", "its", "yours", "such", "into", "of", 
"most", "itself", "other", "off", "is", "s", "am", "or", "who", "as", "from", "him", "each", "the", "themselves", 
"until", "below", "are", "we", "these", "your", "his", "through", "don", "nor", "me", "were", "her", "more", 
"himself", "this", "down", "should", "our", "their", "while", "above", "both", "up", "to", "ours", "had", "she", 
"all", "when", "at", "any", "before", "them", "same", "and", "been", "have", "in", "will", "on", "does", 
"yourselves", "then", "that", "because", "what", "over", "why", "so", "can", "did", "now", "under", "he", 
"you", "herself", "has", "just", "where", "too", "only", "myself", "which", "those", "i", "after", "few", "whom", 
"t", "being", "if", "theirs", "my", "against", "a", "by", "doing", "it", "how", "further", "was", "here", "than","also","movie","watch",""} 
negative_words = {"not","didnt","didn't","doenst","doesn't","dont","don't","isnt","isn't"}


def sorted_aphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(data, key=alphanum_key)

																						#***# READ POSİTİVE COMMENTS
																							# REMOVE HYPERTEXT TEGS, LOWERCASE TEXT, REMOVE SPECİFİC EXTENDED CHARACTERS
								


spell = SpellChecker(distance=1)														#***# SpellChecker(distance=1) is multiple times faster than the default one
ps = PorterStemmer()
bag_of_words = {}
neg_word = ""
final_time = 0


#==================================================VERSION 3 CHANGES========================================================================================#
#Python arrays changed to np.arrays for faster processing
splitted_pos_comment = np.array([])
splitted_neg_comment = np.array([])
splitted_pos_comment = list(splitted_pos_comment)
splitted_neg_comment = list(splitted_neg_comment)
filtered_pos_comment = np.array([])
filtered_neg_comment = np.array([])
filtered_pos_comment = list(filtered_pos_comment)
filtered_neg_comment = list(filtered_neg_comment)
#===========================================================================================================================================================#

file_names = sorted_aphanumeric(os.listdir(targetPositiveFilesPath))						# os.listdir("<path/file_name>") ---> list names of files in specified directory (list type)
pos_comment_number = len(file_names)
for x in range(pos_comment_number):															# index every text file in specified directory, append it to positive comments list	
    text_file = open(targetPositiveFilesPath+"/"+file_names[x],"r",encoding="utf8")			# open ("<file name>","<mode : r(read), w(write), a(append), r+(special read and write)>")
    text = text_file.read()																	# text_file.read(<number of chars to be printed>)
    cleanr = re.compile('<.*?>')
    text = re.sub(cleanr, ' ', text)														# remove hypertext tags
    text = text.lower()																		# lowercase text
    text = re.sub(r'[ş|ç|ö|ü|ğ|ı|.|,|!|?|-]',r'',text)										# remove specific extended characters
    splitted_text = text.split(" ")
                                                              
    filtered_pos_comment.clear()
    start = timer()																			# Timer starts
    for y in range (len(splitted_text)):
        word = splitted_text[y]
        word = spell.correction(word)
        word = ps.stem(word)
        if word not in stop_words:
            if word in negative_words:
                neg_word = word + " "
            else:
                word = neg_word + word
                filtered_pos_comment.append(word)
                neg_word = ""
                
                if word in bag_of_words:
                    value = bag_of_words.get(word)
                    value[0] += 1
                    bag_of_words[word] = value
                else:
                    bag_of_words[word] = [1,0]
    splitted_pos_comment.append(filtered_pos_comment.copy())
    print("[TEST]pos comment loop = ",x+1,"/",pos_comment_number)							#TEST


#==================================================VERSION 3 CHANGES========================================================================================#
#Timer added
    final_time = final_time + (timer()-start) 
#    print("Total time              :",final_time,"\n")
#===========================================================================================================================================================#


text_file.close()            

file_names = sorted_aphanumeric(os.listdir(targetNegativeFilesPath))  

neg_comment_number = len(file_names)
for x in range(neg_comment_number):															# index every text file in specified directory, append it to positive comments list
    text_file = open(targetNegativeFilesPath+"/"+file_names[x],"r",encoding="utf8")			# open ("<file name>","<mode : r(read), w(write), a(append), r+(special read and write)>")
    text = text_file.read()																	# text_file.read(<number of chars to be printed>)
    cleanr = re.compile('<.*?>')
    text = re.sub(cleanr, ' ', text)														# remove hypertext tags
    text = text.lower() 																	# lowercase text
    text = re.sub(r'[ş|ç|ö|ü|ğ|ı|.|,|!|?|-]',r'',text)										# remove specific extended characters
    splitted_text = text.split(" ")
    
    filtered_neg_comment.clear()
    start = timer()																			# Timer starts
    for y in range (len(splitted_text)):
        word = splitted_text[y]
        word = spell.correction(word)
        word = ps.stem(word)
        if word not in stop_words:
            if word in negative_words:
                neg_word = word + " "
            else:
                word = neg_word + word
                filtered_neg_comment.append(word)
                neg_word = ""
                    
                if word in bag_of_words:
                    value = bag_of_words.get(word)
                    value[1] += 1
                    bag_of_words[word] = value
                else:
                    bag_of_words[word] = [0,1]
    splitted_neg_comment.append(filtered_neg_comment.copy())
    print("[TEST]neg comment loop = ",x+1,"/",neg_comment_number)							#TEST


#==================================================VERSION 3 CHANGES========================================================================================#
#Timer added
    final_time = final_time + (timer()-start)
#    print("Total time              :",final_time,"\n")
#===========================================================================================================================================================#
print("Total time              :",final_time,"\n")

text_file.close()

#print("\n[TEST]splitted_pos_comment=",splitted_pos_comment)								# TEST
#print("\n[TEST]splitted_neg_comment=",splitted_neg_comment)								# TEST
#print("\n",bag_of_words)


																						#***# ADD EVERY WORD TO BAG OF WORDS DİCTİONARY, COUNT USAGE TİMES

																			
#print("\n[TEST]bag_of_words=",bag_of_words)

																						#***# REMOVE RARE AND USELESS WORDS
final_table = {}
for key in bag_of_words:
    final_table[key] = []
final_table["CLASS"] = []


print("\nbag_of_words process started")
start = timer()

for key in (bag_of_words):
	value = bag_of_words[key]
	pos_value = value[0]
	neg_value = value[1]
	
	value_difference = min(pos_value,neg_value)/max(pos_value,neg_value)         			# (0~1) Data becomes meaningless as it approaches 1
	limit = 0.8		
	if pos_value == neg_value:
		final_table.pop(key)
	elif (abs(pos_value-neg_value) == 1):													# Adjust lower cut off limit 
		final_table.pop(key)
	elif (value_difference >= limit):
		final_table.pop(key)


final_time = final_time + (timer()-start)
#    print("Bag of words process")
print("bag_of_words process ended in: ",final_time,"secs\n")


																						#***# CREATE FİNAL TABLE FOR MODEL BUİLDİNG





print("Term Frequency process started")

start = timer()
text_count = 0
for c in range (pos_comment_number):   
    for key in final_table:
        count=0
        for w in range(len(splitted_pos_comment[c])):
            word = splitted_pos_comment[c][w]
            if word == key:
                count += 1
        if key != "CLASS":
            final_table[key].append(count)
    final_table["CLASS"].append(1)													 		# 1 = POSITIVE
    text_count += 1
    print("Term Frequency process - pos: ",text_count,"/",pos_comment_number)
final_time = final_time + (timer()-start)
print("Total time              :",final_time,"\n")


start = timer()
text_count = 0
for c in range (neg_comment_number):
    for key in final_table:
        count=0
        for w in range(len(splitted_neg_comment[c])):
            word = splitted_neg_comment[c][w]
            if word == key:
                count += 1
        if key != "CLASS":
            final_table[key].append(count)
    final_table["CLASS"].append(0)															# 0 = NEGATIVE
    text_count += 1
    print("Term Frequency process - neg: ",text_count,"/",neg_comment_number)
final_time = final_time + (timer()-start)
print("Total time              :",final_time,"\n")

print("Term Frequency process ended")

#print("\n[TEST]final_table=",final_table)

																						#***# CREATE DATAFRAME

data_frame = pd.DataFrame(final_table)
data_frame.index = data_frame.index + 1

#print("\n[TEST]data_frame=",data_frame)

#print(data_frame.columns)

X = data_frame.iloc[0:,0:-1]																# features
y = data_frame["CLASS"]																		# target variable/label (CLASS)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)	# 80% training and 20% test samples selected randomly

																				
																				
																				
																				
																				
																						#***# DECİSİON TREE
																				
print("\n========== DECISION TREE ==========\n")
																				
clf = DecisionTreeClassifier()																# Create Decision Tree classifer object
clf = clf.fit(X_train,y_train)																# Train Decision Tree Classifer
y_pred = clf.predict(X_test)																# Predict the response for test dataset
confusionMatrix = metrics.confusion_matrix(y_test, y_pred, labels=None, sample_weight=None)	# Confusion Matrix ( matrix[0][0]=a:TP, matrix[0][1]=b:FN, matrix[1][0]=c:FP, matrix[1][1]=d:TN )
print("Confusion matrix: \n",confusionMatrix)
DT_accuracy = metrics.accuracy_score(y_test, y_pred)										# Model metrics for performance evaluation:


if (confusionMatrix[0][0]==0):
	DT_precision = 0
	DT_recall = 0
	DT_fMeasure = 0
else:
	DT_precision = confusionMatrix[0][0]/(confusionMatrix[0][0]+confusionMatrix[1][0])
	DT_recall = confusionMatrix[0][0]/(confusionMatrix[0][0]+confusionMatrix[0][1])
	DT_fMeasure = (2*confusionMatrix[0][0])/(2*confusionMatrix[0][0]+confusionMatrix[0][1]+confusionMatrix[1][0])

print("Accuracy:",DT_accuracy)
print("Precision:",DT_precision)
print("Recall:",DT_recall)
print("F-Measure:",DT_fMeasure)

																						#***# SVM
																			
print("\n========== SVM ==========\n")

clf = svm.SVC(kernel='linear')
clf = clf.fit(X_train,y_train)																# Train Classifer
y_pred = clf.predict(X_test)																# Predict the response for test dataset
confusionMatrix = metrics.confusion_matrix(y_test, y_pred, labels=None, sample_weight=None)	# Confusion Matrix ( matrix[0][0]=a:TP, matrix[0][1]=b:FN, matrix[1][0]=c:FP, matrix[1][1]=d:TN )
print("Confusion matrix: \n",confusionMatrix)
SVM_accuracy = metrics.accuracy_score(y_test, y_pred)										# Model metrics for performance evaluation:

if (confusionMatrix[0][0]==0):
	SVM_precision = 0
	SVM_recall = 0
	SVM_fMeasure = 0
else:
	SVM_precision = confusionMatrix[0][0]/(confusionMatrix[0][0]+confusionMatrix[1][0])
	SVM_recall = confusionMatrix[0][0]/(confusionMatrix[0][0]+confusionMatrix[0][1])
	SVM_fMeasure = (2*confusionMatrix[0][0])/(2*confusionMatrix[0][0]+confusionMatrix[0][1]+confusionMatrix[1][0])

print("Accuracy:",SVM_accuracy)	
print("Precision:",SVM_precision)
print("Recall:",SVM_recall)
print("F-Measure:",SVM_fMeasure)

																						#***# NAIVE BAYES
																			
print("\n========== NAIVE BAYES ==========\n")

clf = GaussianNB()
clf = clf.fit(X_train,y_train)																# Train Decision Tree Classifer
y_pred = clf.predict(X_test)																# Predict the response for test dataset
confusionMatrix = metrics.confusion_matrix(y_test, y_pred, labels=None, sample_weight=None)	# Confusion Matrix ( matrix[0][0]=a:TP, matrix[0][1]=b:FN, matrix[1][0]=c:FP, matrix[1][1]=d:TN )
print("Confusion matrix: \n",confusionMatrix)
NB_accuracy = metrics.accuracy_score(y_test, y_pred)										# Model metrics for performance evaluation:

if (confusionMatrix[0][0]==0):
	NB_precision = 0
	NB_recall = 0
	NB_fMeasure = 0
else:
	NB_precision = confusionMatrix[0][0]/(confusionMatrix[0][0]+confusionMatrix[1][0])
	NB_recall = confusionMatrix[0][0]/(confusionMatrix[0][0]+confusionMatrix[0][1])
	NB_fMeasure = (2*confusionMatrix[0][0])/(2*confusionMatrix[0][0]+confusionMatrix[0][1]+confusionMatrix[1][0])

print("Accuracy:",NB_accuracy)
print("Precision:",NB_precision)
print("Recall:",NB_recall)
print("F-Measure:",NB_fMeasure)

																						#***# KNN
																			
print("\n========== KNN ==========\n")


clf = KNeighborsClassifier(n_neighbors=5)
clf = clf.fit(X_train,y_train)																# Train Decision Tree Classifer
y_pred = clf.predict(X_test)																# Predict the response for test dataset


confusionMatrix = metrics.confusion_matrix(y_test, y_pred, labels=None, sample_weight=None)	# Confusion Matrix ( matrix[0][0]=a:TP, matrix[0][1]=b:FN, matrix[1][0]=c:FP, matrix[1][1]=d:TN )
print("Confusion matrix: \n",confusionMatrix)
																							# Model metrics for performance evaluation:
KNN_accuracy = metrics.accuracy_score(y_test, y_pred)
print("Accuracy:",KNN_accuracy)
if (confusionMatrix[0][0]==0):
	KNN_precision = 0
	KNN_recall = 0
	KNN_fMeasure = 0
else:
	KNN_precision = confusionMatrix[0][0]/(confusionMatrix[0][0]+confusionMatrix[1][0])
	KNN_recall = confusionMatrix[0][0]/(confusionMatrix[0][0]+confusionMatrix[0][1])
	KNN_fMeasure = (2*confusionMatrix[0][0])/(2*confusionMatrix[0][0]+confusionMatrix[0][1]+confusionMatrix[1][0])

print("Precision:",KNN_precision)
print("Recall:",KNN_recall)
print("F-Measure:",KNN_fMeasure)

																						#***# RANDOM FOREST
																		
print("\n========== RANDOM FOREST ==========\n")

clf = RandomForestRegressor()																# Create Decision Tree classifer object
clf = clf.fit(X_train,y_train)																# Train Decision Tree Classifer
y_pred = clf.predict(X_test)																# Predict the response for test dataset
confusion_matrix = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])	# Confusion Matrix ( matrix[0][0]=a:TP, matrix[0][1]=b:FN, matrix[1][0]=c:FP, matrix[1][1]=d:TN )
print("Confusion matrix: \n",confusionMatrix)
#sn.heatmap(confusion_matrix, annot=True)
																							# Model metrics for performance evaluation:
if (confusionMatrix[0][0]==0):
	RF_accuracy = 0
	RF_precision = 0
	RF_recall= 0
	RF_fMeasure = 0
else:
	RF_accuracy = (confusionMatrix[0][0]+confusionMatrix[1][1])/(confusionMatrix[0][0]+confusionMatrix[0][1]+confusionMatrix[1][0]+confusionMatrix[1][1])
	RF_precision = confusionMatrix[0][0]/(confusionMatrix[0][0]+confusionMatrix[1][0])
	RF_recall = confusionMatrix[0][0]/(confusionMatrix[0][0]+confusionMatrix[0][1])
	RF_fMeasure = (2*confusionMatrix[0][0])/(2*confusionMatrix[0][0]+confusionMatrix[0][1]+confusionMatrix[1][0])

print("Accuracy:",RF_accuracy)
print("Precision:",RF_precision)
print("Recall:",RF_recall)
print("F-Measure:",RF_fMeasure)

																						#***# K-MEANS
																		
print("\n========== K-MEANS ==========\n")

clf = KMeans(init='k-means++')
clf = clf.fit(X_train,y_train)																# Train Decision Tree Classifer
y_pred = clf.predict(X_test)																# Predict the response for test dataset
confusionMatrix = metrics.confusion_matrix(y_test, y_pred, labels=np.unique(y), sample_weight=None)		#labels=np.unique(y)
print("Confusion matrix: \n",confusionMatrix)

# Model metrics for performance evaluation:
KM_accuracy = metrics.accuracy_score(y_test, y_pred)
if (confusionMatrix[0][0]==0):
	KM_precision = 0
	KM_recall = 0
	KM_fMeasure = 0
else:
	KM_precision = confusionMatrix[0][0]/(confusionMatrix[0][0]+confusionMatrix[1][0])
	KM_recall = confusionMatrix[0][0]/(confusionMatrix[0][0]+confusionMatrix[0][1])
	KM_fMeasure = (2*confusionMatrix[0][0])/(2*confusionMatrix[0][0]+confusionMatrix[0][1]+confusionMatrix[1][0])
	
print("Accuracy:",KM_accuracy)
print("Precision:",KM_precision)
print("Recall:",KM_recall)
print("F-Measure:",KM_fMeasure)																		
																		
																							# PRİNT RESULTS AND GRAPHS
																		
print("\n========== FINAL RESULTS COMPARISON ==========\n")
print("Accuracy:")
print("DT :",DT_accuracy)
print("SVM:",SVM_accuracy)
print("NB :",NB_accuracy)
print("KNN:",KNN_accuracy)
print("RF :",RF_accuracy)
print("KM :",KM_accuracy)

axes = plt.gca()
axes.set_ylim([0,1])
plt.figure(1)
objects = ('Accuracy', 'Precision', 'Recall', 'F-Measure')
y_pos = np.arange(len(objects))
performance = [DT_accuracy,DT_precision,DT_recall,DT_fMeasure]

plt.bar(y_pos, performance, align='center', alpha=1)
plt.xticks(y_pos, objects)
plt.ylabel('Rate')
plt.title('Decision Tree')


axes = plt.gca()
axes.set_ylim([0,1])
f1 = plt.figure(2)
objects = ('Accuracy', 'Precision', 'Recall', 'F-Measure')
y_pos = np.arange(len(objects))
performance = [SVM_accuracy,SVM_precision,SVM_recall,SVM_fMeasure]

plt.bar(y_pos, performance, align='center', alpha=1)
plt.xticks(y_pos, objects)
plt.ylabel('Rate')
plt.title('SVM')


axes = plt.gca()
axes.set_ylim([0,1])
f1 = plt.figure(3)
objects = ('Accuracy', 'Precision', 'Recall', 'F-Measure')
y_pos = np.arange(len(objects))
performance = [NB_accuracy,NB_precision,NB_recall,NB_fMeasure]

plt.bar(y_pos, performance, align='center', alpha=1)
plt.xticks(y_pos, objects)
plt.ylabel('Rate')
plt.title('Naive Bayes')


axes = plt.gca()
axes.set_ylim([0,1])
f1 = plt.figure(4)
objects = ('Accuracy', 'Precision', 'Recall', 'F-Measure')
y_pos = np.arange(len(objects))
performance = [KNN_accuracy,KNN_precision,KNN_recall,KNN_fMeasure]

plt.bar(y_pos, performance, align='center', alpha=1)
plt.xticks(y_pos, objects)
plt.ylabel('Rate')
plt.title('KNN')


axes = plt.gca()
axes.set_ylim([0,1])
f1 = plt.figure(5)
objects = ('Accuracy', 'Precision', 'Recall', 'F-Measure')
y_pos = np.arange(len(objects))
performance = [RF_accuracy,RF_precision,RF_recall,RF_fMeasure]

plt.bar(y_pos, performance, align='center', alpha=1)
plt.xticks(y_pos, objects)
plt.ylabel('Rate')
plt.title('Random Forest')


axes = plt.gca()
axes.set_ylim([0,1])
f1 = plt.figure(6)
objects = ('Accuracy', 'Precision', 'Recall', 'F-Measure')
y_pos = np.arange(len(objects))
performance = [KM_accuracy,KM_precision,KM_recall,KM_fMeasure]

plt.bar(y_pos, performance, align='center', alpha=1)
plt.xticks(y_pos, objects)
plt.ylabel('Rate')
plt.title('K-Means')


axes = plt.gca()
axes.set_ylim([0,1])
#f1 = plt.figure(7)
plt.figure(figsize=(8, 5))
objects = ('Decision Tree', 'SVM', 'Naive Bayes', 'KNN', 'Random Forest', 'K-Means')
y_pos = np.arange(len(objects))
performance = [DT_accuracy,SVM_accuracy,NB_accuracy,KNN_accuracy,RF_accuracy,KM_accuracy]

plt.bar(y_pos, performance, align='center', alpha=1)
plt.xticks(y_pos, objects)
plt.ylabel('Accuracy')
plt.title('Learning Algorithm')

plt.show()