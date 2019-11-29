#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json

def json2data(file):
    notes_Dict = {}
    try:
        with open(file, 'r') as json_file:
            notes_Dict = json.load(json_file)
        return (notes_Dict)
    except FileNotFoundError:
        print("Invalid File Name")

def create_octaves(notes, file, octaves):
    notes_Dict = json2data(file)
    octaves_list = []
    for i in range(octaves):
        aux_Dict = {}
        for note in notes:
            fc =  notes_Dict[note]
            aux_Dict[note] = fc * (2**i)
        octaves_list.append(aux_Dict)
    return(octaves_list)

def get_thresholds(notes, file, octaves, th_min):
    octaves_list = create_octaves(notes, file, octaves)
    for i in range(len(octaves_list)):
        for note in notes:
            fc = octaves_list[i][note]
            if i == 0 and note == 'Do':
                th = th_min
            else:
                th = round((prev_fc + ((fc - prev_fc)/2)), 2)
            octaves_list[i][note] = (th, octaves_list[i][note])
            prev_fc = fc
    return(octaves_list)
