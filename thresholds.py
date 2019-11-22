#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json

file = "src/notes.json"
notes_list = ['Do', 'Do#/ReB', 'Re', 'Re#/MiB', 'Mi', 'Fa', 'Fa#/SolB', 'Sol',
              'Sol#/LaB', 'La', 'La#/SiB', 'Si']
def json2data(file):
    notes_Dict = {}
    try:
        with open(file, 'r') as json_file:
            notes_Dict = json.load(json_file)
        return (notes_Dict)
    except FileNotFoundError:
        print("Invalid File Name")

if __name__ == '__main__':
    notes_Dict = json2data(file)
    for note in notes_list:
        notes_Dict[note] = (notes_Dict[note], 5)
        print(notes_Dict[note])
