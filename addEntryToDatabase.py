#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 12:59:29 2020

@author: marc
"""

import easygui
import spacy
import sys
import os

# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '/'.join(os.path.realpath(__file__).split('/')[:-1]))
import functionsFile

#Preparation à la mise en bases de données
#Prémice à l'entraînement

print("Loading Spacy fr_core_news_md...")
nlp = spacy.load('fr_core_news_md')
#Chargement du tokenizer

#Si chemin du fichier non connu, allez le chercher

#Sinon
#path_to_apec_txt_file ="/home/marc/Bureau/test_apec.txt"

#===========

filename=os.path.realpath(__file__)
path_to_file=filename.split('/')
path_to_file=path_to_file[:len(path_to_file)]
path_to_folder='/'.join(path_to_file[:len(path_to_file)-1])+'/'
path_to_database_file=path_to_folder+"bdd_phrases_non_lemmatisees.tsv" #to change depending on database location and name

functionsFile.addNewEntryToDatabase(easygui.fileopenbox(), path_to_database_file, mission_label='Mission', competence_label='Competence', entreprise_label='Entreprise', autre_label='Autre')
