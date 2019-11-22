#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json

FILE = "src/notes.json"
NOTES = ['Do', 'Do#/ReB', 'Re', 'Re#/MiB', 'Mi', 'Fa', 'Fa#/SolB', 'Sol',
         'Sol#/LaB', 'La', 'La#/SiB', 'Si']
OCTAVES_NUM = 5
TH_MIN = 25

def json2data():
    notes_Dict = {}
    try:
        with open(FILE, 'r') as json_file:
            notes_Dict = json.load(json_file)
        return (notes_Dict)
    except FileNotFoundError:
        print("Invalid File Name")

def create_octaves():
    notes_Dict = json2data()
    octaves_list = []
    for i in range(OCTAVES_NUM):
        aux_Dict = {}
        for note in NOTES:
            fc =  notes_Dict[note]
            aux_Dict[note] = fc * (2**i)
        octaves_list.append(aux_Dict)
    return(octaves_list)

def get_thresholds():
    octaves_list = create_octaves()
    for i in range(len(octaves_list)):
        for note in NOTES:
            fc = octaves_list[i][note]
            if i == 0 and note == 'Do':
                th = TH_MIN
            else:
                th = round((prev_fc + ((fc - prev_fc)/2)), 2)
            octaves_list[i][note] = (th, octaves_list[i][note])
            prev_fc = fc
    return(octaves_list)

if __name__ == '__main__':
    thresholds_list = get_thresholds()
    print(thresholds_list)
