import matplotlib as mpl
mpl.use('Agg')
import pynbody
import matplotlib.pyplot as plt
import numpy as np
import pynbody.plot as pp
import pynbody.filt as filt
import pickle
import pandas as pd
import logging
from pynbody import array,filt,units,config,transformation
from pynbody.analysis import halo
import os

# set the config to prioritize the AHF catalog
pynbody.config['halo-class-priority'] =  [pynbody.halo.ahf.AHFCatalogue,
                                          pynbody.halo.GrpCatalogue,
                                          pynbody.halo.AmigaGrpCatalogue,
                                          pynbody.halo.legacy.RockstarIntermediateCatalogue,
                                          pynbody.halo.rockstar.RockstarCatalogue,
                                          pynbody.halo.subfind.SubfindCatalogue, pynbody.halo.hop.HOPCatalogue]
                                          



snapnums = ['004096', '004032', '003936', '003840', '003744', '003648', '003606', '003552', '003456', '003360', '003264', '003195', '003168', '003072','002976', '002880', '002784', '002688', '002592', '002554', '002496', '002400', '002304','002208', '002112', '002088', '002016', '001920', '001824','001740','001728','001632', '001536', '001475', '001440', '001344', '001269', '001248','001152', '001106', '001056', '000974', '000960','000864', '000776', '000768', '000672', '000637', '000576', '000480', '000456', '000384', '000347', '000288', '000275', '000225', '000192', '000188', '000139', '000107', '000096', '000071']

# haloids dictionary defines the major progenitor branch back through all snapshots for each z=0 halo we are 
# interest in ... read this as haloids[1] is the list containing the haloid of the major progenitors of halo 1
# so if we have three snapshots, snapshots = [4096, 2048, 1024] then we would have haloids[1] = [1, 2, 5] 
# --- i.e. in snapshot 2048 halo 1 was actually called halo 2, and in 1024 it was called halo 5
h329_haloids = {
    1: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 5, 19, 18, 11],
    7: [7, 6, 6, 6, 6, 5, 5, 4, 4, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 6, 7, 7, 8, 7, 6, 6, 6, 6, 5, 5, 5, 5, 6, 5, 6, 
        6, 8, 8, 9, 10, 8, 9, 9, 10, 11, 10, 14, 14, 14, 16, 17, 23, 22, 22, 24, 30, 25, 24],
    29: [29, 27, 23, 21, 18, 17, 15, 14, 13, 12, 12, 12, 12, 12, 12, 13, 12, 12, 12, 11, 13, 12, 14, 13, 12, 12, 11, 
         11, 11, 11, 11, 10, 11, 10, 9, 9, 20, 22, 26, 27, 25, 39, 39, 39, 42, 42, 48, 49, 50, 50, 49, 66, 68, 74, 
         77, 97, 89],
    31: [31, 31, 27, 25, 25, 25, 24, 25, 25, 24, 25, 25, 26, 28, 26, 27, 27, 28, 29, 27, 29, 28, 29, 28, 29, 30, 29, 
         28, 28, 29, 29, 28, 28, 30, 29, 30, 30, 28, 30, 31, 27, 23, 23, 25, 27, 27, 42, 34, 40, 45, 43, 39, 80, 90, 
         89, 78, 66, 65],
    32: [32, 32, 32, 32, 32, 34, 34, 35, 34, 33, 32, 30, 31, 31, 27, 19, 17, 16, 25, 29, 15, 13, 13, 12, 11, 11, 12, 
         13, 12, 12, 12, 11, 13, 13, 13, 14, 14, 15, 14, 13, 14, 15, 15, 14, 17, 16, 19, 22, 55],
    55: [55, 56, 54, 54, 55, 55, 55, 55, 51, 48, 49, 48, 49, 48, 46, 43, 44, 44, 41, 41, 42, 42, 41, 42, 40, 41, 39, 
         41, 39, 40, 40, 44, 44, 43, 41, 41, 46, 46, 48, 46, 43, 42, 41, 45, 53, 34, 53, 42, 64, 159, 158, 171, 169,
         191, 193, 148, 106, 105],
    94: [94, 96, 97, 99, 99, 101, 101, 100, 101, 104, 102, 98, 101, 99, 92, 103, 120, 118, 101, 97, 90, 85, 85, 110, 
         84, 84, 69, 71, 70, 73, 74, 73, 72, 70, 69, 62, 57, 50, 42, 39, 36, 30, 31, 33, 36, 35, 41, 38, 44, 52, 52, 
         56, 49, 54, 54, 52, 71, 73],
    116: [116, 118, 119, 123, 121, 122, 119, 116, 114, 113, 111, 101, 105, 102, 100, 96, 92, 92, 91, 84, 86, 87, 84, 
          82, 80, 81, 79, 70, 62, 56, 55, 51, 43, 37, 34, 34, 34, 30, 32, 29, 28, 25, 26, 21, 28, 26, 30, 30, 33, 
          35],
    119: [119, 123, 124, 129, 126, 126, 126, 127, 130, 131, 133, 133, 135, 136, 131, 130, 122, 104, 76, 66],
    131: [131, 136, 133, 138, 138, 137, 138, 137, 139, 142, 143, 143, 146, 149, 147, 151, 150, 150, 151, 150, 149, 
          148, 152, 152, 155, 158, 159, 158, 157, 169, 170, 168, 168, 163, 160, 167, 171, 172, 172, 175, 170, 178, 
          176, 170, 173, 47, 169, 177, 180, 176, 171, 164, 163, 165, 163, 136, 130, 123, 71, 42, 34, 9],
    154: [154, 154, 152, 156, 151, 162, 149, 150, 149, 150, 146, 145, 149, 150, 146, 147, 144, 139, 135, 131, 126, 
          112, 108, 96, 85, 88, 89, 88, 80, 80, 79, 79, 70, 67, 66, 58, 49, 49, 51, 58, 53, 51, 50, 55, 60, 62, 
          67, 67, 69, 68, 71, 70, 70, 76, 74, 73, 72, 69],
    443: [443, 435, 416, 375, 336, 309, 455, 336, 156, 73, 48]
}

h229_haloids = {
    1: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4, 15, 18],
    2: [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 5, 5, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 7, 16, 22, 23, 22, 24, 25, 53, 85],
    5: [5, 5, 5, 5, 5, 5, 4, 4, 5, 5, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 5, 6, 6, 6, 6, 6, 5, 5, 5, 5, 5, 6, 6, 7, 8, 10, 9, 10, 11, 10, 10, 9, 9, 9, 14, 17, 16, 17, 19, 19, 23, 27, 28, 24, 32, 49, 29, 83, 93],
    6: [6, 6, 6, 6, 7, 6, 6, 6, 7, 7, 7, 8, 8, 8, 7, 7, 8, 8, 6, 6, 7, 7, 7, 8, 8, 8, 7, 7, 6, 7, 7, 9, 10, 10, 10, 10, 11, 13, 12, 13, 14, 13, 12, 12, 18, 20, 23, 26, 30, 34, 38, 50, 51, 79, 88, 227, 258, 275],
    10: [10, 10],
    14: [14, 14, 14, 15, 15, 15, 16, 16, 15, 14, 14, 13, 13, 12, 13, 12, 14, 14, 14, 13, 13, 12, 12, 13, 13, 13, 14, 14, 14, 13, 13, 13, 13, 13, 14, 15, 19, 21, 28, 31, 31, 36, 37, 42, 44, 46, 48, 49, 51, 51, 52, 73, 75, 71, 69, 68, 68, 67, 51, 37, 35, 28],
    16: [16, 17, 16, 17, 17, 18, 19, 20, 19, 19, 19, 20, 21, 21, 23, 24, 24, 24, 24, 22, 23, 25, 25, 26, 27, 27, 27, 28, 29, 29, 28, 28, 28, 28, 28, 34, 40, 44, 41, 44, 42, 44, 45, 39, 46, 50, 58, 65, 68, 72, 76, 79, 73, 91, 92, 145, 177, 186, 220],
    18: [18, 19, 18, 18, 18, 16, 17, 17, 17, 17, 17, 17, 18, 17, 16, 15, 15, 11, 10],
    21: [21, 22, 20, 21, 20, 20, 21, 22, 21, 23, 23, 23, 23, 22, 22, 21, 21, 18, 18, 16, 17, 16, 15, 16, 15, 15, 15, 15, 15, 15, 14, 14, 16, 16, 16, 21, 22, 26, 25, 26, 26, 29, 29, 26, 34, 34, 40, 39, 44, 57, 57, 59, 64, 75, 89, 119, 163, 159, 203],
    24: [24, 24, 23, 25, 26, 26, 26, 27, 27, 27, 27, 27, 28, 28, 30, 30, 31, 31, 26, 23, 29, 22, 22, 23, 30, 26, 20, 20, 20, 21, 21, 22, 21, 20, 20, 22, 21, 23, 21, 22, 21, 23, 23, 24, 23, 23, 22, 23, 24, 30, 34, 35, 32, 36, 37, 63, 66, 69, 118],
    27: [27, 27, 27, 29, 30, 31, 31, 32, 32, 31, 33, 33, 33, 33, 34, 35, 35, 35, 35, 34, 35, 36, 37, 38, 37, 38, 37, 39, 39, 38, 37, 34, 35, 36, 37, 38, 39, 42, 37, 39, 40, 43, 44, 43, 45, 47, 52, 54, 56, 61, 59, 63, 66, 63, 61, 61, 52, 53, 38, 28, 20, 13],
    32: [32, 30, 25, 23, 23],
    45: [45, 43, 40, 38, 40, 36, 36, 37, 35, 34, 31, 29, 27, 26, 24, 22, 20, 17, 17, 17, 18, 18, 18, 17, 17, 17, 16, 16, 16, 16, 16, 18, 18, 17, 17, 18, 17, 19, 18, 19, 18, 18, 18, 18, 19, 19, 20, 21, 22, 20, 24, 31, 78, 78, 77, 74, 71, 70, 140],
    46: [46, 48, 47],
    48: [48, 47, 46, 44, 45, 42, 42, 43, 44, 40, 42, 43, 42, 43, 44, 45, 44, 43, 41, 39, 41, 41, 41, 42, 40, 40, 38, 40, 41, 40, 40, 38, 40, 39, 41, 41, 41, 45, 43, 43, 43, 46, 47, 48, 51, 52, 57, 61, 59, 63, 67, 67, 65, 57, 58, 81, 74, 71, 34, 30, 47],
    52: [52, 53, 53, 54, 56, 54, 54, 57, 58, 57, 59, 59, 59, 59, 61, 58, 60, 60, 62, 62, 63, 64, 65, 64, 65, 66, 67, 68, 65, 62, 62, 62, 56, 55, 52, 45, 44, 49, 47, 48, 54, 62, 62, 66, 72, 74, 77, 81, 79, 81, 79, 93, 100, 95, 94, 130, 408, 404],
    58: [58, 59, 60, 59, 61, 59, 59, 61, 62, 61, 62, 62, 62, 62, 65, 65, 67, 66, 67, 66, 65, 66, 68, 67, 69, 69, 69, 71, 71, 73, 74, 76, 81, 80, 81, 83, 82, 87, 85, 87, 85, 89, 92, 96, 103, 103, 111, 115, 121, 113, 119, 135, 153, 159, 151, 179, 164, 163, 128, 98, 76],
    59: [59, 46, 32],
    63: [63, 62, 62, 61, 60, 53, 53, 51, 43, 38, 38, 40, 54, 38, 39, 39, 40, 40, 39, 41, 44, 45, 47, 47, 50, 49, 49, 49, 50, 52, 53, 52, 52, 52, 54, 55, 54, 60, 57, 62, 61, 67, 70, 75, 87, 87, 92, 101, 98, 96, 100, 121, 135, 195, 191, 188, 172, 170, 133, 118],
    67: [67, 67, 66, 65, 66, 67, 67, 68, 70, 66, 54, 42, 38, 34, 28, 27, 27, 26, 27, 27, 26, 27, 27, 28, 28, 29, 28, 30, 30, 31, 31, 26, 25, 25, 25, 26, 27, 27, 27, 28, 27, 30, 31, 28, 31, 31, 32, 33, 39, 39, 39, 40, 42, 41, 41, 43, 46, 43, 36, 113],
    89: [89, 91, 96, 97, 101, 99, 100, 99, 98, 98, 99, 98, 97, 92, 92, 86, 82, 81, 83, 82, 82, 80, 80, 78, 80, 79, 80, 81, 78, 79, 80, 83, 84, 82, 82, 81, 75, 79, 72, 70, 66, 52, 49, 49, 56, 57, 75, 78, 74, 79, 82, 99, 108, 119, 126, 116, 141, 138, 207],
    91: [91, 88, 84, 82, 78, 73, 72, 74, 74, 73, 72, 70, 68, 61, 57, 55, 54, 50, 40, 38, 39, 39, 40, 40, 38, 37, 36, 38, 38, 33, 38, 37, 36, 33, 30, 32, 24, 33, 26, 27, 25, 31, 32, 29, 32, 32, 36, 37, 41, 44, 42, 39, 39, 37, 38, 59, 97, 99, 229],
    128: [128, 131, 131, 127, 129, 128, 127, 121, 110, 104, 104, 100, 98, 95, 97, 94, 95, 98, 98, 96, 94, 93, 93, 94, 97, 96, 105, 105, 102, 106, 107, 111, 112, 110, 109, 110, 110, 115, 113, 119, 121, 118, 118, 121, 128, 127, 124, 130, 133, 131, 134, 150, 158, 154, 145, 142, 224, 224, 175],
    191: [191, 186, 186, 181, 164, 136, 128, 110, 95, 76, 78, 78, 79, 75, 74, 73, 71, 65, 56, 52, 43, 42, 39, 34, 36, 36, 34, 35, 34, 37, 36, 33, 34, 35, 36, 37, 37, 41, 35, 38, 39, 42, 43, 37, 42, 44, 49, 48, 52, 53, 54, 54, 50, 60, 62, 90, 85, 84, 62, 60, 59],
    257: [257, 263, 265, 263, 264, 256, 258, 259, 264, 269, 280, 285, 281, 289, 296, 302, 313, 325, 317, 320, 312, 303, 278, 245, 227, 224, 312, 227, 233, 238, 238, 248, 272, 275, 278, 279, 286, 290, 288, 291, 288, 275, 279, 238, 217, 216, 301, 319, 328, 359, 350, 376, 377, 400, 390, 394, 422, 421],
    538: [538, 540, 520, 501, 486, 471, 471]
}

h242_haloids = {
    1: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 4, 4, 3, 2],
    4: [4],
    10: [10, 9, 8, 9, 9, 6, 6, 7, 7, 4],
    21: [21, 20, 21, 20, 20, 20, 21, 21, 21, 21, 19, 21, 22, 22, 21, 22, 21, 19, 18, 19, 19, 18, 19, 17, 18, 17, 18, 20, 18, 19, 19, 19, 19, 18, 18, 17, 15, 17, 17, 20, 17, 18, 19, 19, 20, 20, 21, 22, 26, 28, 31, 27, 24, 32, 31, 24, 21, 21, 14, 12, 16, 17],
    27: [27, 25, 24, 24, 26, 27, 27, 27, 26, 28, 25, 26, 25, 24, 22, 20, 19, 18, 19, 18, 18, 19, 20, 19, 19, 18, 19, 18, 16, 17, 17, 17, 16, 15, 15, 13, 13, 15, 13, 14, 13, 13, 14, 13, 15, 15, 15, 15, 16, 18, 18, 18, 18, 14, 14, 10, 9, 9, 8, 6, 5, 4],
    31: [31, 30, 32, 31, 31, 32, 32, 33, 32, 33, 30, 31, 31, 32, 32, 32, 32, 33, 35, 35, 33, 32, 30, 28, 29, 27, 28, 27, 24, 24, 24, 23, 23, 23, 24, 25, 20, 22, 21, 24, 20, 23, 23, 21, 22, 22, 23, 24, 28, 32, 32, 37, 37, 39, 40, 38, 25, 25, 17, 11, 15, 28],
    35: [35, 32, 27, 27, 34, 21, 20, 18, 18, 18, 17, 19, 19, 20, 20, 19, 20, 20, 21, 21, 20, 20, 21, 20, 21, 20, 21, 26, 26, 28, 28, 28, 29, 29, 29, 30, 29, 29, 31, 39, 37, 31, 31, 25, 24, 24, 33, 40, 62, 73, 73, 78, 78, 77, 73, 73, 64, 62, 41, 28, 27],
    37: [37, 35, 38, 38, 38, 39, 39, 39, 38, 38, 36, 38, 39, 39, 39, 39, 40, 40, 40, 41, 41, 40, 40, 37, 41, 38, 39, 41, 39, 39, 39, 38, 35, 37, 35, 37, 34, 35, 39, 46, 43, 44, 45, 43, 39, 39, 39, 43, 46, 54, 52, 55, 55, 59, 57, 51, 44, 42, 31, 26, 26],
    45: [45, 44, 45, 44, 46, 44, 45, 45, 44, 45, 45, 46, 46, 47, 47, 46, 47, 48, 51, 49, 48, 48, 47, 47, 52, 51, 51, 53, 52, 51, 50, 45, 44, 42, 43, 45, 44, 45, 44, 51, 49, 51, 52, 44, 42, 41, 42, 47, 49, 55, 53, 52, 51, 54, 54, 54, 42, 41, 36, 31, 28, 26],
    47: [47, 47, 46, 49, 52, 52, 51, 51, 50, 50, 49, 51, 52, 53, 53, 53, 53, 53, 55, 55, 55, 57, 54, 52, 54, 53, 53, 54, 56, 58, 58, 59, 58, 60, 60, 62, 65, 67, 71, 75, 72, 70, 71, 71, 67, 67, 58, 62, 67, 69, 67, 64, 63, 71, 77, 92, 104, 105, 96, 81],
    48: [48, 39, 37, 32, 28, 29, 29, 29, 27, 27, 21, 18, 17, 16, 16, 14, 14, 14, 13, 13, 13, 14, 13, 14, 13, 11, 11, 14, 14, 14, 14, 13, 12, 12, 11, 11, 11, 13, 12, 13, 14, 14, 15, 15, 16, 16, 16, 17, 19, 22, 22, 28, 48, 57, 59, 63, 74, 73, 129],
    49: [49, 48, 49, 50, 53, 53, 53, 53, 53, 53, 54, 55, 63, 56, 57, 55, 55, 57, 58, 58, 58, 60, 59, 58, 61, 61, 61, 67, 66, 70, 69, 73, 74, 80, 80, 85, 85, 91, 89, 94, 91, 87, 85, 85, 79, 78, 73, 78, 87, 103, 104, 104, 97, 98, 97, 102, 94, 98, 166],
    67: [67, 66, 66, 66, 68, 69, 68, 68, 69, 70, 68, 70, 70, 70, 64, 57, 56, 56, 61, 56, 56, 54, 51, 50, 50, 50, 49, 51, 51, 53, 52, 50, 50, 51, 51, 49, 48, 49, 49, 56, 53, 53, 54, 54, 54, 49, 49, 52, 57, 66, 65, 66, 64, 64, 61, 60, 58, 59, 51, 68],
    71: [71, 67, 65, 63, 63, 62, 60, 60, 59, 58, 59, 61, 59, 59, 58, 58, 59, 59, 60, 60, 59, 61, 60, 57, 59, 59, 59, 62, 59, 63, 62, 65, 65, 68, 65, 65, 66, 70, 67, 71, 68, 66, 66, 60, 44, 44, 44, 49, 53, 56, 50, 53, 59, 87, 87, 127, 331, 333],
    81: [81, 79, 76, 72, 70, 63, 59, 55, 55, 51, 44, 45, 45, 45, 45, 44, 45, 45, 47, 47, 47, 46, 46, 42, 42, 41, 38, 39, 25, 29, 32, 39, 36, 36, 36, 42, 39, 37, 40, 47, 44, 42, 40, 40, 40, 40, 55, 56, 59, 61, 59, 58, 58, 79, 80, 83, 93, 97, 113],
    102: [102, 90],
    131: [131, 128, 126, 123, 121, 118, 119, 122, 120, 121, 119, 120, 120, 122, 121, 124, 125, 128, 132, 135, 136, 136, 136, 130, 116, 110, 97, 97, 156, 139, 127, 100, 98, 97, 99, 104, 102, 107, 109, 116, 112, 110, 112, 109, 102, 105, 108, 113, 116, 107, 101, 97, 98, 93, 92, 71, 70, 69, 70],
    407: [407, 407, 408, 404, 409, 410, 412, 409, 403, 396, 374, 357, 348, 287, 231, 204],
    418: [418, 420, 414, 411, 414, 420, 419, 420, 428, 428, 422, 419, 419, 417, 423, 418, 415, 403, 399, 401, 401, 402, 405, 405, 411, 406, 410, 413, 415, 423, 422, 423, 418, 399, 398, 399, 381, 387, 372, 359, 349, 342, 345, 298, 186, 174, 114, 95, 83, 63, 58, 48, 57, 58, 56, 55, 53, 50, 40, 39, 45],
}


def bulk_processing(tstep, haloids, rvirs, snapshots, path, savepath):
    snapshot = snapshots[tstep]
    # load the relevant pynbody data
    s = pynbody.load(path+snapshot)
    s.physical_units()
    t = s.properties['time'].in_units('Gyr')
    print(f'Loaded snapshot {snapshot}, {13.800797497330507 - t} Gyr ago')
    h = s.halos()
    hd = s.halos(dummy=True)
    
    # get current halo ids
    current_haloids = np.array([])
    z0_haloids = np.array([])
    for key in list(haloids.keys())[1:]:
        if not haloids[key][tstep] == 0:
            z0_haloids = np.append(z0_haloids, key)
            current_haloids = np.append(current_haloids, haloids[key][tstep])

    # get current rvirs
    current_rvirs = np.array([])
    for key in list(rvirs.keys())[1:]:
        if not haloids[key][tstep] == 0:
            current_rvirs = np.append(current_rvirs, rvirs[key][tstep])
    
    print(f'Gathered {len(current_haloids)} haloids')
    
    h1id = haloids[1][tstep]
    # get h1 properties
    h1x = hd[h1id].properties['Xc']
    h1y = hd[h1id].properties['Yc']
    h1z = hd[h1id].properties['Zc']
    
    h1d = np.array([h1x, h1y, h1z]) # halo 1 position
    h1d = h1d / s.properties['h'] * s.properties['a']
    h1r = hd[h1id].properties['Rvir'] # halo 1 virial radius
    h1r = h1r / s.properties['h'] * s.properties['a'] # put halo 1 virial radius in physical units
    h1v = np.array([hd[h1id].properties['VXc'], hd[h1id].properties['VYc'], hd[h1id].properties['VZc']]) # halo 1 velocity, already in physical units
    
    pynbody.analysis.angmom.faceon(h[h1id])
    pg = pynbody.analysis.profile.Profile(s.g, min=0.01, max=10*h1r, ndim=3) # make gas density profile
    # pall = pynbody.analysis.profile.Profile(s, min=0.01, max=6*h1r, ndim=3, nbins=200) # make total density profile
    print('\t Made gas density profile for halo 1 (technically halo %s)' % h1id)
    rbins = pg['rbins']
    density = pg['density']
    # density_enc = pall['density_enc']
    # rbins_all = pall['rbins']
    # try:
    #     r200b = np.min(rbins_all[density_enc < 4416.922400090694])
    #     print(f'\t Halo 1 R_200,c = {h1r:.2f} kpc, R_200,b = {r200b:.2f} kpc')
    # except:
    #     r200b = None


    
    for i, rvir, z0haloid in zip(current_haloids, current_rvirs, z0_haloids):
        print('Major progenitor halod ID:', i)
        halo = h.load_copy(i)        
        properties = hd[i].properties

        x = (properties['Xc'] - h1x) / s.properties['h'] * s.properties['a'] 
        y = (properties['Yc'] - h1y) / s.properties['h'] * s.properties['a']
        z = (properties['Zc'] - h1z) / s.properties['h'] * s.properties['a']

        old_rvir = properties['Rvir'] / s.properties['h'] * s.properties['a'] # put rvir in physical units
        print(f'\t Adjusted virial radius {rvir:.2f} kpc, old virial radius {old_rvir:.2f} kpc.')

        # compute ram pressure on halo from halo 1
        # first compute distance to halo 1

        v_halo = np.array([properties['VXc'],properties['VYc'],properties['VZc']]) # halo i velocity
        v = v_halo - h1v

        d = np.array([properties['Xc'],properties['Yc'],properties['Zc']]) # halo i position
        d = d / s.properties['h'] * s.properties['a'] # halo i position in physical units
        d = np.sqrt(np.sum((d - h1d)**2)) # now distance from halo i to halo 1
                
        print('\t Distance from halo 1 (technically halo %s): %.2f kpc or %.2f Rvir' % (h1id,d,d/h1r))
                
        # now use distance and velocity to calculate ram pressure
        pcgm = density[np.argmin(abs(rbins-d))]
        Pram = pcgm * np.sum(v**2)
        print('\t Ram pressure %.2e Msol kpc^-3 km^2 s^-2' % Pram)
                
        try:
            pynbody.analysis.angmom.faceon(h[i])
            pynbody.analysis.angmom.faceon(halo)
            calc_rest = True
            calc_outflows = True
            if len(h[i].gas) < 30:
                raise Exception
        except:
            print('\t Not enough gas (%s), skipping restoring force and inflow/outflow calculations' % len(h[i].gas))
            calc_rest = False
            calc_outflows = False
                        
        # calculate restoring force pressure
        if not calc_rest:
            Prest = None
            ratio = None
            env_vel = None
            env_rho = None
        else:
            print('\t # of gas particles:',len(h[i].gas))
            try:
                p = pynbody.analysis.profile.Profile(h[i].g,min=.01,max=rvir)
                print('\t Made gas density profile for satellite halo %s' % i)
                Mgas = np.sum(h[i].g['mass'])
                percent_enc = p['mass_enc']/Mgas

                rhalf = np.min(p['rbins'][percent_enc > 0.5])
                SigmaGas = Mgas / (2*np.pi*rhalf**2)

                dphidz = properties['Vmax']**2 / properties['Rmax']
                Prest = dphidz * SigmaGas

                print('\t Restoring pressure %.2e Msol kpc^-3 km^2 s^-2' % Prest)
                ratio = Pram/Prest
                print(f'\t P_ram / P_rest {ratio:.2f}')
            except:
                print('\t ! Error in calculating restoring force...')
                Prest = None
                Pram = None

            # calculate nearby region density
            try:
                r_inner = rvir
                r_outer = 3*rvir
                inner_sphere = pynbody.filt.Sphere(str(r_inner)+' kpc', [0,0,0])
                outer_sphere = pynbody.filt.Sphere(str(r_outer)+' kpc', [0,0,0])
                env = s[outer_sphere & ~inner_sphere].gas

                env_volume = 4/3 * np.pi * (r_outer**3 - r_inner**3)
                env_mass = np.sum(env['mass'].in_units('Msol'))
                env_vel = np.mean(np.array(env['mass'].in_units('Msol'))[np.newaxis].T*np.array(env['vel'].in_units('kpc s**-1')), axis=0) / env_mass
                env_rho = env_mass / env_volume
                print(f'\t Environmental density {env_rho:.2f} Msol kpc^-3')
                print(f'\t Environmental wind velocity {env_vel} kpc s^-1')
            except:
                print('\t Could not calculate environmental density')
                env_rho, env_vel = None, None


        age = np.array(h[i].star['age'].in_units('Myr'),dtype=float)
        sfr = np.sum(np.array(h[i].star['mass'].in_units('Msol'))[age < 100]) / 100e6
        print(f'\t Star formation rate {sfr:.2e} Msol yr^-1')

        # calculate gas fraction

        mstar = np.sum(h[i].star['mass'].in_units('Msol'), dtype=float)
        mgas = np.sum(h[i].gas['mass'].in_units('Msol'), dtype=float)
        mass = np.sum(h[i]['mass'].in_units('Msol'),dtype=float)
        
        gas_density = np.array(h[i].g['rho'], dtype=float)
        gas_temp = np.array(h[i].g['temp'], dtype=float)
        gas_mass = np.array(h[i].g['mass'], dtype=float)
        gas_r = np.array(h[i].g['r'], dtype=float)
        hi = np.array(h[i].g['HI'], dtype=float)
        print(f'\t Number of particles in halo {len(gas_density):.2e}')
        
        gas_sphere = h[i][pynbody.filt.Sphere(str(rvir)+' kpc', [0,0,0])].gas
        gas_density_sphere = np.array(gas_sphere['rho'],dtype=float)
        gas_temp_sphere = np.array(gas_sphere['temp'],dtype=float)
        gas_mass_sphere = np.array(gas_sphere['mass'],dtype=float)
        gas_r_sphere = np.array(gas_sphere['r'], dtype=float)
        print(f'\t Number of particles in sphere {len(gas_density_sphere):.2e}')
            
        if mgas == 0 and mstar == 0:
            gasfrac = None
        else:
            gasfrac = mgas / (mstar + mgas)
            print('\t Gas fraction %.2f' % gasfrac)
        
        # calculate gas temperature
        gtemp = np.sum(h[i].gas['temp']*h[i].gas['mass'].in_units('Msol'))/mgas
        print('\t Gas temperature %.2f K' % gtemp)
        # atomic hydrogen gas fraction
        mHI = np.sum(h[i].gas['HI']*h[i].gas['mass'])
        mHII = np.sum(h[i].gas['HII']*h[i].gas['mass'])
        if mHI == 0 and mstar == 0:
            HIGasFrac = None
        else:
            HIGasFrac = mHI/(mstar+mHI)
            if mstar == 0:
                HIratio = None
            else:
                HIratio = mHI/mstar
            print('\t HI gas fraction %.2f' % HIGasFrac)
                
        # get gas coolontime and supernovae heated fraction
        if not mgas == 0:
            coolontime = np.array(h[i].gas['coolontime'].in_units('Gyr'), dtype=float)
            gM = np.array(h[i].gas['mass'].in_units('Msol'), dtype=float) 
            SNHfrac = np.sum(gM[coolontime > t]) / mgas
            print('\t Supernova heated gas fraction %.2f' % SNHfrac)	
        else:
            SNHfrac = None
                
                
        if not calc_outflows:
            GIN2,GOUT2,GIN_T_25,GOUT_T_25,GINL,GOUTL,GIN_T_L,GOUT_T_L = None,None,None,None,None,None,None,None
        else:
            # gas outflow rate
            dL = .1*rvir

            # select the particles in a shell 0.25*Rvir
            inner_sphere2 = pynbody.filt.Sphere(str(.2*rvir) + ' kpc', [0,0,0])
            outer_sphere2 = pynbody.filt.Sphere(str(.3*rvir) + ' kpc', [0,0,0])
            shell_part2 = halo[outer_sphere2 & ~inner_sphere2].gas

            print("\t Shell 0.2-0.3 Rvir")

            #Perform calculations
            velocity2 = shell_part2['vel'].in_units('kpc yr**-1')
            r2 = shell_part2['pos'].in_units('kpc')
            Mg2 = shell_part2['mass'].in_units('Msol')
            r_mag2 = shell_part2['r'].in_units('kpc')
            temp2 = shell_part2['temp']

            VR2 = np.sum((velocity2*r2), axis=1)

            #### first, the mass flux within the shell ####

            gin2 = []
            gout2 = []

            for y in range(len(VR2)):
                if VR2[y] < 0: 
                    gflowin2 = np.array(((VR2[y]/r_mag2[y])*Mg2[y])/dL)
                    gin2 = np.append(gin2, gflowin2)
                else: 
                    gflowout2 = np.array(((VR2[y]/r_mag2[y])*Mg2[y])/dL)
                    gout2 = np.append(gout2, gflowout2)
            GIN2 = np.sum(gin2)
            GOUT2 = np.sum(gout2)

            print("\t Flux in %.2f, Flux out %.2f" % (GIN2,GOUT2))

            ##### now, calculate temperatures of the mass fluxes ####

            tin2 = []
            min_2 = []
            tout2 = []
            mout_2 = []

            for y in range(len(VR2)):
                if VR2[y] < 0: 
                    intemp2 = np.array(temp2[y]*Mg2[y])
                    tin2 = np.append(tin2, intemp2)
                    min2 = np.array(Mg2[y])
                    min_2 = np.append(min_2, min2)
                else: 
                    outemp2 = np.array(temp2[y]*Mg2[y])
                    tout2 = np.append(tout2, outemp2)
                    mout2 = np.array(Mg2[y])
                    mout_2 = np.append(mout_2, mout2)

            in_2T = np.sum(tin2)/np.sum(min_2)
            out_2T = np.sum(tout2)/np.sum(mout_2)    
            GIN_T_25 = np.sum(in_2T)
            GOUT_T_25 = np.sum(out_2T)

            print("\t Flux in temp %.2f, Flux out temp %.2f" % (GIN_T_25,GOUT_T_25))

            print('\t Shell 0.9-1.0 Rvir')
            #select the particles in a shell
            inner_sphereL = pynbody.filt.Sphere(str(.9*rvir) + ' kpc', [0,0,0])
            outer_sphereL = pynbody.filt.Sphere(str(rvir) + ' kpc', [0,0,0])
            shell_partL = halo[outer_sphereL & ~inner_sphereL].gas

            #Perform calculations
            DD = .1*rvir
            velocityL = shell_partL['vel'].in_units('kpc yr**-1')
            rL = shell_partL['pos'].in_units('kpc')
            MgL = shell_partL['mass'].in_units('Msol')
            r_magL = shell_partL['r'].in_units('kpc')
            tempL = shell_partL['temp']

            VRL = np.sum((velocityL*rL), axis=1)

            #### First, the Mas Flux within a Shell ####

            ginL = []
            goutL = []

            for y in range(len(VRL)):
                if VRL[y] < 0: 
                    gflowinL = np.array(((VRL[y]/r_magL[y])*MgL[y])/DD)
                    ginL = np.append(ginL, gflowinL)
                else: 
                    gflowoutL = np.array(((VRL[y]/r_magL[y])*MgL[y])/DD)
                    goutL = np.append(goutL, gflowoutL)   
            GINL = np.sum(ginL)
            GOUTL = np.sum(goutL)

            print('\t Flux in %.2f, Flux out %.2f' % (GINL,GOUTL))

            ##### now calculate the temperature of the flux ####

            tinL = []
            min_L = []
            toutL = []
            mout_L = []

            for y in range(len(VRL)):
                if VRL[y] < 0: 
                    intempL = np.array(tempL[y]*MgL[y])
                    tinL = np.append(tinL, intempL)
                    minL = np.array(MgL[y])
                    min_L = np.append(min_L, minL)
                else: 
                    outempL = np.array(tempL[y]*MgL[y])
                    toutL = np.append(toutL, outempL)
                    moutL = np.array(MgL[y])
                    mout_L = np.append(mout_L, moutL)

            in_LT = np.sum(tinL)/np.sum(min_L)
            out_LT = np.sum(toutL)/np.sum(mout_L)    
            GIN_T_L = np.sum(in_LT)
            GOUT_T_L = np.sum(out_LT)

            print("\t Flux in temp %.2f, Flux out temp %.2f" % (GIN_T_L,GOUT_T_L))


        f = open(savepath, 'ab')
        pickle.dump({
                'time': t, 
                't':t,
                'z':s.properties['z'],
                'a':s.properties['a'],
                'haloid': i,
                'z0haloid':z0haloid,
                'mstar': mstar,
                'mgas': mgas,
                'mass':mass,
                'Rvir': rvir,
                #'r200b':r200b,
                'gas_rho': gas_density,
                'gas_temp':gas_temp,
                'gas_mass':gas_mass,
                'gas_r':gas_r,
                'gas_hi':hi,
                'gas_rho_sphere': gas_density_sphere,
                'gas_temp_sphere':gas_temp_sphere,
                'gas_mass_sphere':gas_mass_sphere,
                'gas_r_sphere':gas_r_sphere,
                'x':x,
                'y':y,
                'z':z,
                'sfr':sfr,
                'Pram': Pram, 
                'Prest': Prest, 
                'v_halo':v_halo,
                'v_halo1':h1v,
                'v_env':env_vel,
                'env_rho':env_rho,
                'ratio': ratio, 
                'h1dist': d/h1r, 
                'h1dist_kpc': d,
                'h1rvir':h1r,
                'gasfrac': gasfrac,
                'SNHfrac': SNHfrac,
                'mHI':mHI,
                'fHI': HIGasFrac, 
                'HIratio': HIratio,
                'gtemp': gtemp,
                'inflow_23':GIN2,
                'outflow_23':GOUT2,
                'inflow_temp_23':GIN_T_25,
                'outflow_temp_23':GOUT_T_25,
                'inflow_91':GINL,
                'outflow_91':GOUTL,
                'inflow_temp_91':GIN_T_L,
                'outflow_temp_91':GOUT_T_L
        },f,protocol=pickle.HIGHEST_PROTOCOL)
        f.close()




if __name__ == '__main__':

    for name, haloids in zip(['h329','h242','h229'],[h329_haloids, h242_haloids, h229_haloids]):
        path= '/home/christenc/Data/Sims/'+name+'.cosmo50PLK.3072g/'+name+'.cosmo50PLK.3072gst5HbwK1BH/snapshots/'
        snapshots = [name+'.cosmo50PLK.3072gst5HbwK1BH.'+snapnum for snapnum in snapnums]

        for key in list(haloids.keys()):
            if len(haloids[key]) != len(snapshots):
                for i in range(len(snapshots)-len(haloids[key])):
                    haloids[key].append(0)
                                
        print(haloids)

        savepath = f'/home/akinshol/Data/Timescales/DataFiles2/'+name+'.data'
        if os.path.exists(savepath):
            os.remove(savepath)
            print('Removed previous .data file')
        print(f'Saving to {savepath}')

        for tstep in range(len(snapshots)):
            bulk_processing(tstep, haloids, snapshots, path, savepath)

