#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 13 01:06:13 2020

@author: marc
"""

import numpy as np
import nltk
import nltk.data
import csv
from nltk.corpus import stopwords
import spacy
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import sys

sys.path.insert(1, '/'.join(os.path.realpath(__file__).split('/')[:-1]))

print("Loading Spacy fr_core_news_md...")
nlp = spacy.load('fr_core_news_md')

#===========================================================================================
#Functions for extraction of APEC Job Offer from Txt File
#===========================================================================================

def morphosyntaxic_labelling(shortened_description):
    
    '''
    Pretreats and tokenise a text to prepare for lemmatization.
    
    Returns a list of list of [phrase's tokens, tokens' position].
    
    shortened_description is the text to be treated.
    '''
    
    descriptif_syntaxe=[]
    for line in shortened_description:
        treated_line=' '.join(line)
        treated_line=nlp(treated_line)
        tokens=[]
        for token in treated_line:
            tokens.append([token,token.pos_])
        descriptif_syntaxe.append(tokens)
    return descriptif_syntaxe

def pretreatmentOfJobOfferText(bloc_de_texte,language="french"):
    
    '''
    Prepare a textblock to be analysed, by tokenizing, lemmatizing and removing punctuation.
    Returns the treated textblock.
    
    bloc_de_text is the textblock to be processed
    language is the language used to process nltk.corpus.stopwords, by default set to "french".
    '''
    
    stop_words = set(stopwords.words(language))
    
    #Tokenise Sentences
    organized_description=[]
    for line in bloc_de_texte:
        sentences = nltk.sent_tokenize(line)
        organized_description.append(sentences)
        
    #Lemmatisation
    lemmatized_description=[]
    for line in organized_description:
        treated_line=nlp(line[0])
        tokens=[]
        for token in treated_line:
            tokens.append(token.lemma_)
        line_lemmatise=' '.join(tokens)
        lemmatized_description.append(line_lemmatise)
    
    #Tokenise Words
    shortened_description=[]
    for sentence in lemmatized_description:
        words = nltk.word_tokenize(sentence)
        without_stop_words = [word for word in words if not word in stop_words]
        without_commas = [word for word in without_stop_words if word!=',']
        shortened_description.append(without_commas)
    shortened_description = list(filter((',').__ne__, shortened_description))
    
    #Remove Punctuation
    MS_shortened_description=morphosyntaxic_labelling(shortened_description)
    treated_description=[]
    for i in range(len(MS_shortened_description)):
        newline=[]
        for j in range(len(shortened_description[i])):
            if MS_shortened_description[i][j][1] != 'PUNCT':
                if MS_shortened_description[i][j][1] != 'X':
                    newline.append(shortened_description[i][j])
        treated_description.append(newline)
    
    #Final concatenation
    return [' '.join(line) for line in treated_description]

def extractJobOffer(path):

    '''
    Extract an APEC Type Job Offer in Txt form.
    Returns a list of three texts, corresponding to each section of the APEC Job Offer. Respectively : Description, Profile wanted, Company.
    
    path is the path to the APEC Job Offer in Txt form.
    '''
 
    #Extraction
    with open(path) as f:
        content = f.readlines()
    
    #Separating of paragraphs in sentences by detecting "."
    new_content=[]
    i=0
    for i in range(len(content)):
        contents=content[i].split(".")
        for e in contents:
            new_content.append(e)
    content=new_content
    
    #Remove whitespaces and empty strings
    content = [x.strip() for x in content]
    content = list(filter(None, content))
    
    #Sorting
    index_descriptif=content.index("Descriptif du poste")
    index_profil=content.index("Profil recherché")
    index_entreprise=content.index("Entreprise")
    description=content[index_descriptif+1:index_profil]
    profile=content[index_profil+1:index_entreprise]
    company=content[index_entreprise+1:]
    
    return [description, profile, company]

#===========================================================================================
#Functions for preparation of Imported APEC Job Offer for Learning
#===========================================================================================

def sentence_features_mission_autre(phrase_lemma):

    '''
    Extract features from lemmatized sentences with mission context.
    Returns the features of the given lemmatized sentence.
    
    phrase_lemma is the lemmatized sentence to be processed.
    '''
    
    features=dict()
    features['taille']=len(phrase_lemma)
    
    features['premier mot']=str(phrase_lemma[0])
    features['dernier mot']=str(phrase_lemma[-1])
    MS_lemmatized_sentence=morphosyntaxic_labelling([phrase_lemma])
    morphosyntaxes=[]
    for line in MS_lemmatized_sentence:
        for e in line:
            morphosyntaxes.append(e[1])
    features['fonction du premier mot']=str(morphosyntaxes[0])
    features['fonction du mot à 25%']=str(morphosyntaxes[int(len(morphosyntaxes)*0.25)])
    features['fonction du mot du milieu']=str(morphosyntaxes[len(morphosyntaxes)//2])
    features['fonction du mot à 75%']=str(morphosyntaxes[int(len(morphosyntaxes)*0.75)])
    features['fonction du dernier mot']=str(morphosyntaxes[-1])
    
    return features

def sentence_features_competence_autre(phrase_lemma):
    
    '''
    Extract features from lemmatized sentences with skills context.
    Returns the features of the given lemmatized sentence.
    
    phrase_lemma is the lemmatized sentence to be processed.
    '''
    
    features=dict()
    features['taille']=len(phrase_lemma)
    
    often_recognized_verbs=['être','aimer','avoir','maîtriser','justifier','issu','faire']
    for verb_parsed in often_recognized_verbs:
        if verb_parsed in phrase_lemma:
            features["nb. de mots "+verb_parsed]=phrase_lemma.count(verb_parsed)
            features["index de première occurence du mot "+verb_parsed]=phrase_lemma.index(verb_parsed)
    
    features['premier mot']=str(phrase_lemma[0])
    features['dernier mot']=str(phrase_lemma[-1])
    MS_lemmatized_sentence=morphosyntaxic_labelling([phrase_lemma])
    morphosyntaxes=[]
    for line in MS_lemmatized_sentence:
        for e in line:
            morphosyntaxes.append(e[1])
    features['fonction du premier mot']=str(morphosyntaxes[0])
    if len(morphosyntaxes)>1:
        features['fonction du deuxième mot']=str(morphosyntaxes[1])
        if len(morphosyntaxes)>2:
            features['fonction du troisième mot']=str(morphosyntaxes[1])    
    features['fonction du mot à 25%']=str(morphosyntaxes[int(len(morphosyntaxes)*0.25)])
    features['fonction du mot du milieu']=str(morphosyntaxes[len(morphosyntaxes)//2])
    features['fonction du mot à 75%']=str(morphosyntaxes[int(len(morphosyntaxes)*0.75)])
    features['fonction du dernier mot']=str(morphosyntaxes[-1])

    for word in phrase_lemma:
        features['contains({})'.format(word)] = (word in set(phrase_lemma))
    
    return features

def labelingOfDatabase(path):
    
    '''
    Labellise the database found at the given path.
    
    path is the path used to get to the wanted database.
    '''
    
    with open(path,'r') as f:
        lines=f.readlines()
        bdd=[]
        for line in lines:
            e = line.split('\t')
            bdd.append((e[1].split(),e[0]))
    return bdd

def createFeaturesetList(fonction,bdd):
    
    '''
    Create a list of features corresponding to each entry of the database.
    Returns a list of tuples (features, label)
    
    fonction is the function used to extract features from a lemmatized sentence.
    bdd is the database of lemmatized sentences used, organized as a list of tuples (line,label).
    '''
    
    return [(fonction(line), label) for (line, label) in bdd]

def buildDataframe(fonction_features,bdd):
    
    '''
    Builds a dataframe for training and learning purposes from the database with the given
    
    fonction is the function used to extract features from a lemmatized sentence.
    bdd is the database of lemmatized sentences used, organized as a list of tuples (line,label).
    '''
    
    df = pd.DataFrame(columns=['Id', 'Label', 'Phrase','Phrase_lemmatisee','data'])
    for i in range(len(bdd)):
        lemmatized_sentence=pretreatmentOfJobOfferText([' '.join(bdd[i][0])])
        line_to_add={'Id':i,'Label':bdd[i][1],'Phrase':' '.join(bdd[i][0]),'Phrase_lemmatisee':lemmatized_sentence[0],'data':fonction_features(lemmatized_sentence[0].split())}
        df=df.append(line_to_add,ignore_index=True)
    return df

#===========================================================================================
#Functions for evaluation of Training/Learning
#===========================================================================================

def evaluation(df_positif,df_negatif,test_size=0.1,classifier=nltk.NaiveBayesClassifier):

    '''
    Evaluates the precision of a given classifier according to positive and negative 
    
    Returns nothing, print results.
    
    df_positif is a dataframe containing entries with labels corresponding as positives in your evaluation.
    df_negatif is a dataframe containing entries with labels corresponding as negatives in your evaluation.
    test_size is the percentage of entries used for test, as a float [0;1]
    classifier is the classifier used.
    '''
    
    train_positif, test_positif = train_test_split(df_positif, test_size=test_size)
    train_negatif, test_negatif = train_test_split(df_negatif, test_size=test_size)
    
    df_train, df_test = pd.concat([train_positif,train_negatif]), pd.concat([test_positif,test_negatif])
    
    X_train = np.stack(df_train['data'])
    X_test = np.stack(df_test['data'])
    
    X_train = [(X_train[i], df_train.iloc[i]['Label']) for i in range(len(X_train))]
    X_test = [(X_test[i], df_test.iloc[i]['Label']) for i in range(len(X_test))]
    
    #y_train = np.where(np.array(df_train['Label'] == 'autre'),0,1)
    #y_test = np.where(np.array(df_test['Label'] == 'autre'),0,1)
    
    #print(train_set)
    #print('===')
    #print(test_set)
    
    classifier_trained = classifier.train(X_train)
    print(nltk.classify.accuracy(classifier, X_test))
    classifier_trained.show_most_informative_features(5)
    
    errors = []
    for i in range(len(X_test)):
        (features, label)=X_test[i]
        guess = classifier_trained.classify(features)
        proba = classifier_trained.prob_classify(features)
        if guess != label:
            errors.append( [label, guess, df_test.iloc[i], proba] )
    for error in errors:
        labels=df_test['Label'].unique()
        print("Label : {0} ; Prediction : {1}".format(error[0],error[1]))
        print("Probabilities({} ; {}) : {} ; {} ".format(labels[0],labels[1],error[3].prob(labels[0]),error[3].prob(labels[1])))
        print("Line : {}".format(error[2]['Id']))
        print("Sentence : {}".format(error[2]['Phrase']))
        print('---')

#===========================================================================================
#Training/Learning functions
#===========================================================================================
    
def createLearnerMissionSkillsOthers(path_bdd,features_function_used,mission_label='Mission',competence_label='Competence',autre_label='Autre'):

    '''
    Trains two classifiers with a given database. One is used to spot entries relative to missions, the other to skills.
    
    path_bdd is the path to the file used as a database.
    features_function_used is the function used to extract features from a lemmatized sentence.
    
    mission_label is the label describing a mission when classifying.
    competence_label is the label describing a skill when classifying.
    autre_label is the label describing what is not a skill or mission when classifying.
    '''    

    bdd=labelingOfDatabase(path_bdd)
    bdd_dataframe=buildDataframe(features_function_used,bdd)
        
    missions_df=bdd_dataframe[bdd_dataframe['Label']==mission_label]
    skills_df=bdd_dataframe[bdd_dataframe['Label']==competence_label]
    autres_dataframe=bdd_dataframe[bdd_dataframe['Label']==autre_label]
    
    #Preparation des Bases de données pour exploitation
    #OVR Classifier, j'aurais pu améliorer l'algorithme. whatever ?
    #Utile pour deviner automatiquement dans un texte ce qui est une mission ou une compétence
    
    missions_autres_dataframe=pd.concat([missions_df,autres_dataframe])
    missions_autres_stack=np.stack(missions_autres_dataframe['data'])
    missions_autres_train=[(missions_autres_stack[i], missions_autres_dataframe.iloc[i]['Label']) for i in range(len(missions_autres_stack))]
    
    competences_autres_dataframe=pd.concat([skills_df,autres_dataframe])
    competences_autres_stack=np.stack(competences_autres_dataframe['data'])
    competences_autres_train=[(competences_autres_stack[i], competences_autres_dataframe.iloc[i]['Label']) for i in range(len(competences_autres_stack))]
    
    missions_autres_classifier = nltk.NaiveBayesClassifier.train(missions_autres_train)
    competences_autres_classifier = nltk.NaiveBayesClassifier.train(competences_autres_train)
    
    return missions_autres_classifier, competences_autres_classifier

#===========================================================================================
#Entries-related functions
#===========================================================================================

def classifyEntry(path_to_file_to_classify, features_function_used, missions_autres_classifier, competences_autres_classifier,mission_label='Mission',competence_label='Competence',autre_label='Autre'):
    
    '''
    Classify a given APEC Txt file following its trained classifiers,
    and creates a "treated file" at the same folder the file given is.
    
    path_to_file_to_classify is the path to where the APEC Txt file is.
    missions_autres_classifier is the classifier used to differentiate missions from other entries.
    competences_autres_classifier is the classifier used to differentiate skills from other entries.
    
    mission_label is the label describing a mission when classifying.
    competence_label is the label describing a skill when classifying.
    autre_label is the label describing what is not a skill or mission when classifying.
    '''
    
    #Fichier importe
    texte_separe=extractJobOffer(path_to_file_to_classify)
    df_texte_separe=[]
    
    #Preparation du fichier importe
    for j in range(len(texte_separe)):
        bloc=texte_separe[j]
        df_bloc=pd.DataFrame(columns=['Id', 'Label', 'Phrase','Phrase_lemmatisee','data'])
        for i in range(len(bloc)):
            if (len(bloc[i].split())>1):
                lemmatized_sentence=pretreatmentOfJobOfferText([bloc[i]])
                line_to_add={'Id':i,'Phrase':' '.join(bloc[i]),'Phrase_lemmatisee':lemmatized_sentence[0],'data':features_function_used(lemmatized_sentence[0].split())}
                df_bloc=df_bloc.append(line_to_add,ignore_index=True)
        df_texte_separe.append(df_bloc)
        
    
    #Classification
    for index, row in df_texte_separe[0].iterrows():
        #Missions
        row['Label']=missions_autres_classifier.classify(row['data'])
    missions_df_to_write=df_texte_separe[0][df_texte_separe[0]['Label'] == mission_label]
    
    for index, row in df_texte_separe[1].iterrows():
        #Competences
        row['Label']=competences_autres_classifier.classify(row['data'])
    skills_df_to_write=df_texte_separe[1][df_texte_separe[1]['Label'] == competence_label]
    
    #Détection des thèmes principaux d'une phrase
    #Ceci sert à déterminer quelle réponse ou phrase préconstruite est à utiliser.
    #On peut le déterminer à partir de quelques verbes et des mots particuliers
    
    all_needed_info=pd.concat([missions_df_to_write,skills_df_to_write])
    all_sentences=np.stack(all_needed_info['Phrase_lemmatisee'])
    
    with open('/'.join(path_to_file_to_classify.split('/')[:-1])+'/'+path_to_file_to_classify.split('/')[-1].split('.txt')[0]+'_treated_text.txt','w') as f:
        for sentence in all_sentences.tolist():
            f.write('['+sentence+']'+'\n')
            f.write('\n')

def addNewEntryToDatabase(path_to_apec_txt_file, path_to_database_file, mission_label='Mission', competence_label='Competence', entreprise_label='Entreprise', autre_label='Autre'):
    
    '''
    Import a new APEC offer (formatted into .txt) and adds it to the database to be used later in learning.
    
    path_to_apec_txt_file is the path to the APEC Txt file (.txt) the user want to import.
    path_to_database is the path to the Database fiile (.tsv) used for learning purpose.
    
    mission_label is the label describing a mission when classifying.
    competence_label is the label describing a skill when classifying.
    autre_label is the label describing what is not a skill or mission when classifying.
    '''
    
    [treated_description,treated_profile,treated_company]=extractJobOffer(path_to_apec_txt_file)
        
    with open(path_to_database_file,'a') as f:
        
        spamwriter=csv.writer(f,delimiter='\t',quotechar='|',quoting=csv.QUOTE_MINIMAL)
        
        for i in range(len(treated_description)):
            line=treated_description[i]
            reponse=input(line+'\n'+"Does this line describe a mission ? [y]/n")
            if len(line)>1:
                if reponse=='n':
                    spamwriter.writerow([autre_label, line])
                else:
                    spamwriter.writerow([mission_label, line])
        for i in range(len(treated_profile)):
            line=treated_profile[i]
            reponse=input(line+'\n'+"Does this line describe a skill ? [y]/n")
            if len(line)>1:
                if reponse=='n':
                    spamwriter.writerow([autre_label, line])
                else:
                    spamwriter.writerow([competence_label, line])
        for i in range(len(treated_company)):
            line=treated_company[i]
            if len(line)>1:
                reponse=input(line+'\n'+"Does this line describe a core aspect of the company ? [y]/n")
                if reponse=='n':
                    spamwriter.writerow([autre_label, line])
                else:
                    spamwriter.writerow([entreprise_label, line])