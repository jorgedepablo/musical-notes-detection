#!/usr/bin/env python3
# -*- coding: utf-8 -*-

def get_matrix():
    test = [ ]
    for i in range(7):
        if i == 0:
            name = 'Do octava 5'
            note_test = 'do.wav'
        if i == 1:
            name = 'Re octava 5'
            note_test = 're.wav'
        if i == 2:
            name = 'Mi octava 5'
            note_test = 'mi.wav'
        if i == 3:
            name = 'Fa octava 5'
            note_test = 'fa.wav'
        if i == 4:
            name = 'Sol octava 5'
            note_test = 'sol.wav'
        if i == 5:
            name = 'La octava 4'
            note_test = 'la.wav'
        if i == 6:
            name = 'Si octava 4'
            note_test = 'si.wav'
        test.append([name, note_test])
    return(test)
