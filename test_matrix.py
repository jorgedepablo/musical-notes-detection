#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wf
import scipy.signal as ss

import thresholds
import tds_utils

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
            note_test = 're.wav'
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


def run_test(test):
    results = []
    for j in range(len(test)):
        print('Hpola')
        file = (path + test[j][1])
        fs,y = wf.read(file)
        print(fs)
        print(file)
        s = y[100:1000]
        psd,f = tds_utils.my_spectra(s,fs)
        peaks, propieties = ss.find_peaks(psd, height=10000)
        peak = peaks[0]
        fc = abs(f[peak])
        print(fc)

        i = 0
        while i <= len(thresholds) - 1:
            if i == OCTAVES_NUM - 1:
                if fc > thresholds[i]['Do'][0] and fc < TH_MAX:
                    octave = i + 1
                    break
            else:
                if fc > thresholds[i]['Do'][0] and fc < thresholds[i+1]['Do'][0]:
                    octave = i + 1
                    break
                else:
                    i += 1

        for note in NOTES:
            th_inf = thresholds[i][note][0]
            if fc > th_inf and note != 'Si':
                note_ant = note
            elif fc > th_inf and note == 'Si':
                result = (note + ' octava ' + str(octave))
                break
            else:
                result = (note_ant + ' octava ' + str(octave))
                break
        results.append(result)
    print(result)

if __name__ == "__main__":

    test = get_matrix()
    run_test(test)
