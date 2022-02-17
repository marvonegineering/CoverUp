#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 22:08:12 2020

@author: marc
"""

import os
import sys
import easygui

sys.path.insert(1, '/'.join(os.path.realpath(__file__).split('/')[:-1]))
import functionsFile

#======================
#importing
#======================

#filename = easygui.fileopenbox() #for easy search of your wanted database. by default, will look for original database.
filename=os.path.realpath(__file__)
path_to_file=filename.split('/')
path_to_file=path_to_file[:len(path_to_file)]
path_to_folder='/'.join(path_to_file[:len(path_to_file)-1])+'/'

#to change depending on database location and name, and function used to extract features from targeted APEC job offer
path_bdd=path_to_folder+"bdd_phrases_non_lemmatisees.tsv"
features_function_used=functionsFile.sentence_features_competence_autre

#===============================
#learning on database
#===============================

learner1, learner2 = functionsFile.createLearnerMissionSkillsOthers(path_bdd, features_function_used)

#===============================
#classifying and exporting
#===============================

functionsFile.classifyEntry(easygui.fileopenbox(), features_function_used, learner1, learner2)
