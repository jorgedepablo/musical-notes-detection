#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json

FILE = "src/notes.json"
NOTES = ['Do', 'Do#/ReB', 'Re', 'Re#/MiB', 'Mi', 'Fa', 'Fa#/SolB', 'Sol',
         'Sol#/LaB', 'La', 'La#/SiB', 'Si']
OCTAVES_NUM = 5

def json2data(file):
    notes_Dict = {}
    try:
        with open(FILE, 'r') as json_file:
            notes_Dict = json.load(json_file)
        return (notes_Dict)
    except FileNotFoundError:
        print("Invalid File Name")

def create_octaves(notes_Dict):
    octaves_list = []
    i = 1
    for i <= OCTAVES_NUM :
        aux_Dict = {}
        aux_Dict = notes_Dict
        for note in NOTES:
            aux_Dict[note] = notes_Dict[note]*(2**(i-1))
        octaves_list.append(aux_Dict)
    print(octaves_list)


if __name__ == '__main__':
    notes_Dict = json2data(FILE)
    create_octaves(notes_Dict)
