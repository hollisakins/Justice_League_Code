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


from timescales_bulk import bulk_processing


snapshots = ['h148.cosmo50PLK.3072g3HbwK1BH.004096', 
             'h148.cosmo50PLK.3072g3HbwK1BH.003968', 
             'h148.cosmo50PLK.3072g3HbwK1BH.003840', 
             'h148.cosmo50PLK.3072g3HbwK1BH.003712', 
             'h148.cosmo50PLK.3072g3HbwK1BH.003606', 
             'h148.cosmo50PLK.3072g3HbwK1BH.003584', 
             'h148.cosmo50PLK.3072g3HbwK1BH.003456', 
             'h148.cosmo50PLK.3072g3HbwK1BH.003328', 
             'h148.cosmo50PLK.3072g3HbwK1BH.003200', 
             'h148.cosmo50PLK.3072g3HbwK1BH.003195', 
             'h148.cosmo50PLK.3072g3HbwK1BH.003072', 
             'h148.cosmo50PLK.3072g3HbwK1BH.002944', 
             'h148.cosmo50PLK.3072g3HbwK1BH.002816', 
             'h148.cosmo50PLK.3072g3HbwK1BH.002688', 
             'h148.cosmo50PLK.3072g3HbwK1BH.002554', 
             'h148.cosmo50PLK.3072g3HbwK1BH.002432', 
             'h148.cosmo50PLK.3072g3HbwK1BH.002304', 
             'h148.cosmo50PLK.3072g3HbwK1BH.002176', 
             'h148.cosmo50PLK.3072g3HbwK1BH.002088', 
             'h148.cosmo50PLK.3072g3HbwK1BH.002048', 
             'h148.cosmo50PLK.3072g3HbwK1BH.001920', 
             'h148.cosmo50PLK.3072g3HbwK1BH.001740', 
             'h148.cosmo50PLK.3072g3HbwK1BH.001536', 
             'h148.cosmo50PLK.3072g3HbwK1BH.001408', 
             'h148.cosmo50PLK.3072g3HbwK1BH.001280', 
             'h148.cosmo50PLK.3072g3HbwK1BH.001269', 
             'h148.cosmo50PLK.3072g3HbwK1BH.001152', 
             'h148.cosmo50PLK.3072g3HbwK1BH.001106', 
             'h148.cosmo50PLK.3072g3HbwK1BH.001024', 
             'h148.cosmo50PLK.3072g3HbwK1BH.000974', 
             'h148.cosmo50PLK.3072g3HbwK1BH.000896', 
             'h148.cosmo50PLK.3072g3HbwK1BH.000866', 
             'h148.cosmo50PLK.3072g3HbwK1BH.000768', 
             'h148.cosmo50PLK.3072g3HbwK1BH.000701', 
             'h148.cosmo50PLK.3072g3HbwK1BH.000640', 
             'h148.cosmo50PLK.3072g3HbwK1BH.000512', 
             'h148.cosmo50PLK.3072g3HbwK1BH.000456', 
             'h148.cosmo50PLK.3072g3HbwK1BH.000384', 
             'h148.cosmo50PLK.3072g3HbwK1BH.000347', 
             'h148.cosmo50PLK.3072g3HbwK1BH.000275', 
             'h148.cosmo50PLK.3072g3HbwK1BH.000225', 
             'h148.cosmo50PLK.3072g3HbwK1BH.000188', 
             'h148.cosmo50PLK.3072g3HbwK1BH.000139']

haloids = {
    1: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 10, 13, 40, 45, 54],
    2: [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 6, 5, 5, 5, 5, 6, 4, 5, 6, 18, 25, 33, 15, 19, 24],
    3: [3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 3, 3, 4, 4, 4, 4, 4, 6, 11, 12, 11, 13, 14, 17, 16, 14, 18, 10, 8, 5, 4],
    5: [5, 5, 5, 4, 4, 4, 4, 4, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 5, 5, 5, 6, 6, 6, 6, 6, 7, 7, 7, 6, 7, 10, 14, 10, 8, 10, 7, 5, 3, 3, 4, 5],
    6: [6, 6, 6, 6, 6, 6, 6, 5, 6, 6, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 7, 7, 8, 8, 8, 8, 10, 10, 8, 8, 8, 12, 15, 18, 17, 17, 16, 54, 47, 46, 44],
    9: [9, 9, 9, 9, 7, 7, 7, 7, 8, 8, 8, 8, 9, 9, 9, 9, 8, 8, 8, 8, 8, 8, 9, 9, 11, 11, 9, 9, 12, 12, 9, 9, 7, 10, 23, 20, 19, 19, 23, 22, 11, 9, 55],
    10: [10, 10, 11, 11, 11, 11, 12, 12, 13, 13, 13, 13, 13, 13, 13, 14, 14, 13, 13, 13, 13, 13, 18, 18, 20, 20, 20, 20, 22, 21, 29, 29, 34, 36, 42, 51, 72, 112, 123, 211, 613],
    11: [11, 11, 10, 10, 10, 10, 11, 11, 12, 12, 10, 11, 10, 19, 10, 10, 10, 10, 10, 10, 10, 10, 11, 12, 15, 15, 18, 18, 16, 16, 20, 20, 32, 34, 39, 39, 39, 40, 42, 29, 14, 12, 16],
    13: [13, 13, 13, 13, 13, 13, 14, 14, 14, 14, 14, 14, 15, 12, 12, 11, 11, 11, 11, 11, 11, 11, 12, 13, 13, 13, 14, 14, 15, 15, 18, 18, 17, 18, 16, 16, 25, 23, 20, 12, 9, 11, 9],
    14: [14, 14, 14, 12, 12, 12, 13, 13, 15, 15, 15, 15, 14, 14, 14, 15, 15, 15, 15, 15, 16, 16, 19, 24, 42, 42, 47, 44, 46, 49, 54, 57, 69, 74, 71, 71, 73, 109, 113, 121, 193, 232],
    21: [21, 19, 18, 16, 16, 17, 17, 17, 17, 17, 17, 17, 17, 18, 19, 19, 19, 18, 17, 17, 18, 17, 22, 22, 27, 28, 35, 36, 42, 65, 85, 84, 82, 88, 91, 173, 194, 263, 347, 401, 450, 445, 320],
    24: [24, 24, 24, 23, 23, 24, 24, 24, 25, 25, 25, 24, 24, 26, 26, 26, 27, 28, 29, 30, 32, 33, 35, 35, 36, 36, 38, 37, 38, 41, 42, 41, 45, 54, 74, 118, 111, 132, 214, 313, 518],
    27: [27, 28, 28, 28, 29, 29, 25, 23, 23, 23, 18, 19, 18, 16, 18, 17, 17, 14, 14, 14, 14, 15, 17, 17, 19, 19, 22, 22, 24, 24, 27, 27, 25, 26, 29, 32, 33, 41, 48, 125, 243, 343],
    28: [28, 29, 29, 29, 27, 26, 22, 19, 19, 19, 19, 18, 19, 17, 17, 16, 16, 17, 19, 19, 12, 12, 13, 15, 17, 18, 17, 16, 17, 18, 22, 24, 23, 24, 22, 45, 61, 118, 129, 110, 108, 101, 182],
    30: [30, 31, 31, 31, 32, 33, 33, 32, 31, 31, 32, 30, 30, 31, 32, 31, 35, 46, 51, 51, 54, 60, 64, 66, 72, 71, 69, 70, 68, 66, 77, 78, 94, 101, 108, 123, 141, 163, 174, 199, 163, 116, 107],
    32: [32, 27, 27, 26, 26, 27, 27, 26, 27, 27, 27, 27, 27, 28, 27, 27, 28, 26, 28, 29, 30, 29, 32, 38, 45, 47, 52, 54, 53, 53, 59, 59, 64, 63, 64, 92, 233, 229, 248, 271, 303, 375, 369],
    36: [36, 35, 33, 40, 34, 34, 31, 29, 28, 28, 26, 26, 26, 25, 24, 25, 24, 25, 26, 26, 27, 28, 33, 33, 33, 33, 36, 35, 37, 42, 44, 44, 46, 48, 51, 53, 54, 56, 66, 83, 85, 102, 269],
    37: [37, 37, 36, 34, 35, 36, 35, 34, 34, 34, 34, 32, 28, 23, 23, 24, 23, 24, 23, 25, 26, 26, 28, 28, 28, 29, 33, 34, 36, 39, 45, 45, 48, 52, 59, 108, 128, 206, 209, 224, 277],
    41: [41, 40, 40, 42, 43, 44, 45, 46, 46, 46, 50, 48, 49, 48, 45, 47, 46, 47, 48, 49, 52, 54, 60, 61, 61, 61, 58, 60, 63, 70, 93, 93, 112, 123, 129, 126, 124, 146, 188, 302, 269, 216],
    45: [45, 43, 43, 39, 40, 41, 39, 38, 36, 36, 36, 35, 32, 30, 28, 22, 20, 19, 18, 20, 22, 19, 23, 21, 24, 24, 27, 27, 32, 32, 30, 30, 31, 37, 50, 63, 65, 75, 154, 172, 285, 351, 297],
    47: [47, 47, 46, 45, 47, 47, 46, 45, 45, 45, 48, 45, 45, 47, 44, 46, 45, 45, 45, 47, 50, 52, 83, 81, 88, 91, 107, 281, 251, 230, 191, 181, 187, 226, 235, 262, 288, 470, 424, 595, 620, 471],
    48: [48, 48, 38, 32, 31, 32, 30, 31, 32, 32, 33, 34, 34, 34, 34, 34, 34, 34, 34, 36, 40, 39, 43, 47, 54, 56, 63, 65, 66, 62, 66, 75, 91, 97, 112, 195, 360, 483, 523, 830],
    58: [58, 60, 60, 60, 60, 61, 63, 64, 67, 66, 67, 67, 69, 72, 73, 72, 70, 75, 77, 76, 79, 82, 96, 101, 107, 105, 103, 105, 108, 114, 121, 122, 123, 132, 135, 152, 173, 171, 176, 230, 262, 201],
    61: [61, 62, 62, 57, 59, 60, 61, 60, 59, 59, 55, 55, 47, 42, 40, 40, 41, 42, 43, 46, 49, 49, 57, 59, 64, 64, 64, 63, 67, 63, 67, 68, 74, 78, 85, 98, 98, 96, 98, 91, 68, 48, 46],
    65: [65, 64, 65, 64, 57, 57, 52, 48, 47, 47, 49, 47, 48, 50, 54, 54, 52, 56, 59, 58, 60, 61, 67, 70, 122, 132, 162, 189, 204, 212, 212, 206, 186, 187, 188, 199, 209, 214, 202, 194, 212, 263, 277],
    68: [68, 51, 49, 46, 45, 45, 44, 43, 42, 42, 41, 37, 39, 27, 21, 21, 21, 21, 22, 23, 24, 21, 25, 25, 25, 25, 28, 30, 34, 38, 40, 46, 126, 152, 160, 203, 238, 254, 283, 662, 756],
    80: [80, 69, 66, 55, 51, 51, 49, 44, 41, 41, 39, 33, 31, 32, 30, 29, 30, 23, 21, 21, 21, 27, 21, 20, 22, 22, 24, 24, 25, 25, 28, 28, 28, 31, 30, 28, 26, 22, 29, 32, 28, 33, 25],    
    81: [81, 91, 77, 73, 70, 70, 69, 69, 63, 62, 58, 54, 60, 63, 74, 76, 74, 80, 79, 79, 81, 84, 117, 137, 155, 153, 164, 169, 187, 187, 205, 203, 215, 232, 255, 299, 299, 329, 303, 256, 295, 282, 195],
    96: [96, 97, 99, 99, 95, 93, 92, 90, 86, 86, 87, 99, 59, 54, 55, 53, 49, 49, 53, 53, 53, 56, 61, 64, 67, 67, 67, 71, 69, 68, 73, 70, 73, 75, 83, 103, 109, 121, 118, 92, 65, 47, 28],
    105: [105, 105, 107, 108, 107, 109, 113, 111, 113, 112, 113, 121, 118, 121, 122, 124, 125, 129, 132, 131, 136, 137, 147, 159, 175, 174, 174, 177, 179, 177, 188, 184, 194, 207, 216, 246, 263, 251, 266, 294, 266, 200, 143],
    119: [119, 120, 128, 129, 129, 130, 132, 132, 130, 129, 132, 128, 124, 119, 110, 110, 106, 106, 101, 102, 96, 90, 89, 102, 116, 115, 120, 126, 130, 133, 135, 137, 137, 135, 142, 186, 190, 202, 193, 212, 881],
    127: [127, 124, 118, 117, 116, 117, 85, 37, 37, 37, 37, 38, 37, 39, 37, 38, 38, 36, 36, 38, 41, 42, 48, 53, 53, 53, 55, 56, 57, 56, 69, 74, 81, 93, 128, 148, 155, 200, 197, 167, 151, 174, 246],
    128: [128, 130, 132, 131, 127, 128, 123, 113, 109, 109, 104, 100, 100, 98, 79, 68, 66, 66, 68, 69, 72, 73, 75, 73, 78, 77, 74, 79, 77, 76, 82, 82, 85, 89, 89, 85, 106, 211, 236, 259, 237, 328, 346],
    136: [136, 136, 141, 143, 144, 145, 146, 151, 153, 153, 155, 158, 159, 159, 154, 151, 199, 125, 122, 119, 108, 108, 98, 114, 82, 81, 83, 82, 80, 84, 89, 88, 111, 116, 122, 155, 164, 167, 165, 153, 120, 99, 122],
    163: [163, 163, 171, 172, 171, 174, 176, 184, 183, 183, 186, 189, 195, 203, 204, 206, 211, 216, 222, 222, 229, 233, 210, 218, 87, 86, 85, 83, 81, 83, 86, 85, 87, 92, 93, 101, 95, 91, 86, 78, 67, 115, 83],
    212: [212, 211, 211, 203, 188, 185, 179, 183, 178, 178, 175, 171, 168, 164, 126, 121, 117, 116, 112, 111, 103, 57, 58, 58, 65, 65, 75, 80, 86, 90, 95, 91, 97, 105, 120, 188, 188, 197, 179, 128, 156, 184, 317],
    265: [265, 264, 258, 255, 249, 250, 246, 247, 184, 184, 176, 179, 174, 180, 174, 171, 167, 145, 133, 136, 139, 92, 90, 90, 92, 92, 89, 89, 89, 93, 97, 94, 105, 109, 111, 112, 118, 129, 133, 100, 76, 96, 342],
    278: [278, 280, 286, 309, 289, 268, 189, 158, 119, 119, 116, 119, 113, 110, 101, 93, 87, 50, 47, 33, 36, 44, 49, 50, 50, 50, 51, 48, 50, 51, 56, 56, 65, 67, 66, 75, 92, 151, 157, 152, 205, 214, 232],
    283: [283, 279, 267, 229, 213, 207, 180, 173, 173, 175, 196, 74, 68, 46, 48, 36, 32, 33, 32, 35, 38, 36, 41, 42, 44, 44, 46, 46, 49, 52, 60, 63, 72, 86, 94, 139, 178, 184, 167, 208, 337, 393],
    329: [329, 324, 304, 279, 262, 264, 260, 261, 251, 252, 253, 247, 229, 214, 207, 190, 200, 169, 147, 137, 128, 130, 112, 93, 132, 79, 43, 42, 43, 45, 51, 52, 67, 71, 77, 77, 80, 73, 71, 47, 63, 87, 88],
    372: [372, 370, 360, 342, 272, 260, 229, 213, 204, 204, 204, 195, 190, 182, 165, 154, 150, 156, 154, 155, 161, 151, 154, 167, 179, 179, 191, 190, 196, 198, 201, 199, 223, 244, 246, 219, 255, 304, 315, 280, 289, 303, 307],
    377: [377, 368, 353, 334, 318, 311],
    384: [384, 335, 331, 311, 303, 301, 281, 273, 272, 272, 277, 267, 263, 256, 233, 222, 224, 225, 234, 226, 221, 202, 166, 141, 145, 143, 141, 138, 136, 142, 138, 138, 138, 143, 145, 149, 145, 155, 153, 164, 109, 83, 234],
    386: [386, 304, 222, 218, 217, 217, 216, 217, 215, 215, 212, 210, 166, 168, 169, 169, 164, 166, 168, 167, 162, 133, 119, 122, 130, 126, 118, 117, 111, 111, 116, 112, 115, 118, 117, 113, 115, 130, 132, 116, 96, 139, 80],
    442: [442, 403, 438, 84, 52, 53, 55, 53, 54, 54, 56, 58, 57, 58, 56, 57, 56, 59, 60, 60, 59, 64, 78, 91, 102, 102, 106, 108, 113, 119, 128, 131, 141, 146, 143, 146, 148, 168, 168, 175, 203, 204, 163],
    491: [491, 478, 483, 481, 485, 487, 489, 492, 486, 486, 491, 491, 429, 418, 420, 427, 431, 443, 449, 457, 459, 452, 397, 230, 243, 241, 242, 237, 235, 236, 234, 230, 247, 255, 258, 268, 265, 216, 190, 213, 176, 105, 50],
    620: [620, 313, 264, 263, 260, 259, 258, 263, 252, 253, 247, 232, 227, 237, 192, 174, 172, 177, 183, 183, 186, 191, 205, 215, 240, 238, 248, 253, 265, 274, 283, 287, 295, 325, 325, 357, 359, 379, 413, 402, 315, 257, 149],
    678: [678, 660, 548, 519, 521, 520, 514, 499, 482, 482, 509, 332, 317, 317, 322, 327, 329, 336, 328, 314, 277, 502, 168, 116, 97, 97, 96, 100, 100, 103, 115, 115, 131, 151, 153, 143, 180, 327, 339, 343, 302, 249, 197],
    699: [699, 707, 714, 725, 738, 744, 761, 770, 777, 777, 784, 769, 737, 745, 781, 513, 486, 484, 471, 466, 422, 353, 224, 227, 239, 234, 213, 199, 178, 233, 123, 121, 118, 121, 130, 132, 127, 139, 160, 249, 329, 312, 235],
    759: [760, 767, 777, 789, 800, 802, 817, 821, 823, 826, 826, 837, 848, 864, 835, 827, 831, 851, 860, 864, 896, 891, 804, 693, 580, 496, 449, 448, 473, 722, 471, 481, 510, 524, 497, 553, 820, 882, 1044],
    914: [915, 912, 906, 910, 910, 910, 903, 881, 890, 893, 887, 899, 896, 901, 894, 886, 893, 896, 891, 889, 906, 909, 963, 1026, 1065, 1076, 1105, 1101, 1141, 1148, 1135, 1142, 1256, 1367, 1449, 1580, 1723],
    1004: [1005, 1022, 1038, 1061, 1078, 1080, 1093, 1091, 1110, 1103, 1120, 1150, 1208, 1244, 1336, 1470],
    1024: [1025, 1067, 1103, 1128, 1161, 1165, 1216, 1254, 1301, 1304, 1364, 1381, 1424]
}

rvirs = {
    2: [162.58, 157.76, 153.02, 148.34, 144.52, 143.73, 139.18, 134.69, 129.08, 128.86, 123.1, 113.79, 105.92, 94.98, 89.34, 85.0, 80.5, 76.14, 73.07, 71.68, 67.13, 60.76, 54.29, 50.68, 47.01, 46.61, 42.32, 40.69, 37.81, 36.46, 34.43, 33.58, 29.61, 26.09, 24.73, 18.62, 14.95, 9.85, 8.05, 5.65, 4.66, 3.5, 2.07],
    3: [131.3, 127.41, 123.58, 119.8, 116.71, 116.08, 112.4, 108.77, 105.18, 105.04, 101.63, 98.11, 94.61, 91.15, 87.53, 83.72, 79.11, 75.04, 72.21, 70.82, 66.72, 60.99, 54.84, 51.43, 48.54, 48.2, 44.8, 43.19, 39.27, 34.43, 25.75, 25.2, 24.57, 21.89, 19.1, 14.47, 12.62, 10.43, 8.83, 6.83, 5.47, 4.61, 2.95],
    5: [109.43, 106.19, 102.99, 99.85, 97.27, 96.74, 93.68, 90.65, 87.66, 87.55, 84.7, 81.76, 78.68, 75.56, 72.32, 69.21, 66.11, 62.93, 60.85, 59.88, 56.83, 52.48, 47.57, 44.21, 40.78, 40.51, 37.47, 36.36, 34.36, 33.06, 29.93, 28.71, 24.75, 21.68, 20.2, 16.36, 14.06, 11.73, 10.93, 8.56, 6.25, 4.69, 2.81],
    6: [110.21, 106.95, 103.73, 100.56, 97.97, 97.43, 94.35, 91.3, 88.29, 88.17, 85.31, 82.43, 79.38, 76.28, 73.1, 70.1, 66.97, 63.62, 61.19, 60.12, 56.73, 51.93, 46.0, 42.02, 38.17, 37.9, 35.18, 34.02, 31.68, 30.28, 28.43, 27.98, 25.16, 22.23, 18.38, 14.22, 12.09, 9.81, 8.86, 4.59, 3.6, 2.84, 1.85],
    9: [96.92, 94.04, 91.22, 88.43, 86.15, 85.68, 82.97, 80.29, 77.64, 77.54, 75.02, 72.42, 69.84, 67.28, 64.61, 62.32, 59.78, 56.96, 54.94, 53.97, 50.86, 46.46, 41.81, 38.69, 35.72, 35.46, 32.33, 31.13, 28.95, 27.95, 27.11, 27.08, 25.31, 22.54, 13.97, 13.0, 12.03, 9.9, 8.28, 6.07, 5.03, 3.93, 1.74],
    10: [74.55, 72.34, 70.16, 67.95, 66.12, 65.7, 62.53, 60.22, 58.1, 58.01, 55.9, 53.68, 51.23, 49.1, 46.94, 45.19, 43.3, 41.24, 39.96, 39.41, 37.34, 34.5, 30.57, 28.47, 26.69, 26.56, 24.97, 24.14, 22.21, 20.82, 18.17, 17.27, 15.13, 13.87, 12.44, 9.12, 7.05, 5.23, 4.63, 2.97, 1.57],
    11: [86.53, 83.96, 81.44, 78.95, 76.92, 76.5, 74.08, 71.68, 69.32, 69.22, 66.98, 64.66, 62.35, 60.07, 57.69, 55.53, 53.27, 51.01, 48.64, 47.49, 44.16, 38.95, 34.97, 32.55, 28.63, 28.31, 25.57, 24.81, 24.01, 23.35, 21.09, 20.8, 15.35, 13.9, 12.74, 10.16, 9.15, 7.44, 6.68, 5.73, 4.7, 3.75, 2.33],
    13: [66.71, 64.73, 62.79, 60.87, 59.3, 58.98, 57.11, 55.27, 53.86, 53.8, 61.92, 59.77, 57.64, 55.53, 53.33, 51.33, 49.25, 47.16, 45.47, 44.76, 42.36, 38.69, 34.49, 31.99, 29.66, 29.45, 27.26, 26.38, 24.58, 23.48, 21.58, 21.02, 19.22, 17.96, 17.08, 14.71, 11.05, 9.37, 8.48, 6.68, 5.37, 3.86, 2.58],
    14: [67.92, 65.91, 63.93, 61.97, 60.38, 60.05, 58.28, 56.51, 54.73, 54.67, 53.03, 51.23, 49.46, 47.42, 45.18, 43.15, 40.01, 38.28, 37.13, 36.56, 35.19, 32.58, 28.57, 23.01, 18.13, 17.93, 16.2, 15.6, 14.65, 13.91, 12.79, 12.32, 10.46, 9.66, 9.25, 7.88, 7.07, 5.33, 4.76, 3.56, 2.46, 1.83],
    21: [61.69, 59.87, 58.07, 56.29, 54.84, 54.54, 52.82, 51.18, 49.49, 49.43, 47.75, 46.03, 44.31, 42.52, 40.67, 38.99, 37.24, 35.48, 34.33, 33.88, 32.64, 30.63, 27.13, 23.81, 21.17, 20.85, 18.55, 16.89, 15.14, 12.38, 10.84, 10.55, 9.81, 9.18, 8.55, 5.68, 4.92, 3.78, 3.08, 2.3, 1.75, 1.44, 1.08],
    24: [54.38, 52.79, 51.19, 49.57, 48.35, 48.12, 46.6, 45.01, 43.48, 43.41, 41.83, 40.13, 38.36, 36.72, 35.06, 33.51, 31.85, 30.15, 29.06, 28.57, 26.94, 24.7, 22.0, 20.39, 18.78, 18.65, 17.34, 16.76, 15.88, 15.31, 14.65, 14.37, 12.84, 11.15, 9.04, 6.44, 5.93, 4.91, 3.69, 2.54, 1.66],
    27: [66.71, 64.73, 62.79, 60.87, 59.3, 58.98, 57.11, 55.27, 53.44, 53.37, 51.64, 49.85, 48.07, 46.31, 44.48, 42.81, 41.07, 39.33, 38.13, 37.59, 35.84, 33.36, 30.91, 29.49, 27.13, 26.79, 23.52, 21.87, 20.29, 19.43, 18.11, 17.67, 16.65, 15.47, 13.93, 10.92, 9.53, 7.42, 6.31, 3.51, 2.26, 1.61],
    28: [72.07, 69.93, 67.83, 65.76, 64.06, 63.71, 61.7, 59.7, 57.73, 57.66, 55.78, 53.85, 51.93, 50.03, 48.05, 46.25, 44.37, 42.49, 41.19, 40.6, 38.71, 36.04, 32.42, 30.57, 25.57, 25.15, 25.85, 25.17, 23.76, 22.57, 20.5, 19.55, 17.22, 16.53, 15.49, 9.7, 7.7, 5.07, 4.44, 3.62, 2.84, 2.3, 1.32],
    30: [48.8, 47.12, 45.43, 43.88, 42.67, 42.42, 41.05, 39.74, 38.57, 38.53, 37.82, 37.01, 36.16, 35.02, 33.13, 31.72, 28.89, 24.57, 23.22, 22.75, 21.3, 19.3, 17.39, 16.12, 14.93, 14.84, 13.91, 13.54, 12.79, 12.24, 11.22, 10.86, 9.38, 8.59, 7.88, 6.44, 5.5, 4.57, 4.01, 3.05, 2.59, 2.25, 1.52],
    32: [52.87, 51.3, 49.76, 48.24, 47.0, 46.74, 45.26, 43.8, 42.27, 42.2, 40.68, 39.06, 37.52, 36.14, 34.68, 33.33, 31.83, 30.29, 29.35, 28.86, 27.32, 25.35, 22.9, 19.76, 17.25, 17.11, 15.3, 14.71, 13.65, 13.1, 12.14, 11.79, 10.89, 10.33, 9.73, 7.25, 3.42, 3.92, 3.51, 2.69, 2.08, 1.55, 1.02],
    36: [53.17, 51.59, 50.04, 48.51, 47.26, 47.0, 45.51, 44.04, 42.59, 42.53, 41.15, 39.73, 38.31, 36.95, 35.51, 34.18, 32.87, 31.34, 30.09, 29.54, 27.93, 25.73, 22.92, 21.26, 19.88, 19.76, 18.14, 17.46, 16.17, 15.35, 14.31, 13.87, 12.69, 11.78, 10.69, 9.09, 8.14, 6.72, 5.86, 4.0, 3.04, 2.31, 1.15],
    37: [44.27, 43.06, 41.93, 48.73, 47.47, 47.21, 45.72, 44.24, 42.78, 42.72, 41.33, 39.9, 38.48, 37.07, 35.6, 34.23, 32.77, 31.33, 30.39, 29.98, 28.35, 26.13, 23.34, 21.84, 20.42, 20.29, 18.62, 17.96, 16.56, 15.37, 13.55, 13.2, 12.04, 11.24, 10.15, 6.69, 5.66, 4.17, 3.75, 2.91, 2.17],
    41: [41.52, 40.29, 39.08, 37.89, 36.88, 36.67, 35.48, 34.29, 33.13, 33.1, 32.01, 30.92, 29.86, 28.86, 27.79, 26.77, 25.62, 24.43, 23.53, 23.14, 21.95, 20.12, 18.1, 16.93, 15.99, 15.91, 14.79, 14.27, 13.14, 12.15, 10.29, 9.89, 8.71, 7.98, 7.44, 6.38, 5.82, 4.86, 3.89, 2.59, 2.17, 1.89],
    45: [57.46, 55.76, 54.08, 52.43, 51.08, 50.8, 49.19, 47.6, 46.03, 45.97, 44.48, 42.93, 41.41, 39.89, 38.31, 36.87, 35.37, 33.87, 32.84, 32.36, 30.74, 28.31, 25.49, 23.8, 22.04, 21.87, 20.04, 19.32, 18.17, 17.67, 17.28, 17.09, 15.42, 13.8, 10.65, 8.41, 7.55, 5.8, 4.1, 3.16, 2.12, 1.58, 1.11],
    47: [42.07, 40.82, 39.59, 38.38, 37.4, 37.19, 36.01, 34.85, 33.7, 33.66, 32.56, 31.43, 30.32, 29.13, 27.9, 26.87, 25.76, 24.7, 24.01, 23.69, 22.42, 20.17, 15.92, 14.86, 13.41, 13.26, 11.58, 4.98, 6.18, 6.77, 8.04, 7.9, 7.19, 6.41, 5.94, 4.77, 4.22, 2.99, 2.84, 1.98, 1.57, 1.42],
    48: [48.25, 46.82, 45.41, 44.03, 42.89, 42.66, 41.31, 39.9, 38.29, 38.25, 37.04, 35.86, 34.83, 33.61, 32.02, 30.56, 28.9, 27.51, 26.57, 26.05, 24.46, 22.22, 20.04, 18.6, 16.58, 16.4, 14.45, 13.79, 13.07, 12.68, 11.71, 11.04, 9.54, 8.81, 7.88, 5.35, 3.84, 2.96, 2.62, 1.76],
    58: [36.88, 35.7, 34.55, 33.46, 32.53, 32.34, 31.21, 30.12, 29.09, 29.05, 28.06, 26.99, 25.96, 24.84, 23.81, 22.85, 22.02, 21.03, 20.3, 19.96, 18.79, 16.87, 14.97, 13.84, 12.75, 12.65, 11.63, 11.21, 10.44, 9.95, 9.31, 9.09, 8.32, 7.79, 7.27, 5.9, 5.14, 4.46, 3.96, 2.86, 2.2, 1.9],
    61: [43.75, 42.45, 41.18, 39.92, 38.89, 38.68, 37.45, 36.24, 35.05, 35.0, 33.86, 32.69, 31.53, 30.37, 29.17, 28.08, 26.63, 25.24, 24.29, 23.83, 22.45, 20.55, 18.48, 17.12, 15.73, 15.6, 14.27, 13.79, 12.93, 12.41, 11.61, 11.28, 10.17, 9.49, 8.71, 7.0, 6.32, 5.57, 5.05, 3.93, 3.28, 2.82, 1.84],
    65: [41.52, 40.29, 39.08, 37.89, 36.91, 36.71, 35.55, 34.4, 33.26, 33.22, 32.14, 31.05, 29.88, 28.56, 26.59, 25.18, 23.95, 22.77, 21.94, 21.6, 20.45, 19.07, 17.09, 15.7, 12.27, 11.83, 10.15, 9.34, 8.6, 8.2, 7.67, 7.53, 7.08, 6.9, 6.37, 5.25, 4.73, 4.1, 3.78, 3.06, 2.37, 1.75, 1.12],
    68: [56.06, 54.4, 52.76, 51.15, 49.83, 49.56, 47.99, 46.44, 44.91, 44.85, 43.39, 41.89, 40.4, 38.91, 37.37, 35.97, 34.51, 32.67, 31.66, 31.2, 29.61, 27.14, 24.51, 22.94, 21.87, 21.76, 20.21, 19.31, 17.22, 15.89, 15.62, 13.3, 8.31, 7.53, 6.87, 5.26, 4.59, 3.83, 3.33, 1.91, 1.46],
    80: [23.89, 23.18, 22.48, 21.8, 21.24, 21.12, 20.45, 19.79, 19.14, 19.11, 18.49, 17.85, 17.22, 16.58, 15.93, 15.33, 14.71, 14.08, 13.65, 13.46, 12.83, 11.95, 26.95, 25.33, 23.66, 23.5, 21.85, 21.15, 19.92, 19.2, 18.08, 17.64, 15.93, 14.86, 13.78, 11.71, 11.04, 9.52, 7.66, 5.64, 4.25, 3.12, 2.07],
    81: [35.43, 34.38, 33.35, 32.33, 31.49, 31.32, 30.33, 29.35, 30.02, 30.04, 30.05, 29.24, 27.71, 25.78, 23.77, 22.55, 21.57, 20.64, 20.07, 19.79, 18.71, 16.79, 13.75, 12.52, 11.29, 11.18, 10.13, 9.68, 8.9, 8.47, 7.81, 7.56, 6.88, 6.31, 5.75, 4.56, 4.1, 3.47, 3.26, 2.79, 2.12, 1.72, 1.3],
    96: [29.05, 28.24, 27.47, 26.75, 26.27, 26.18, 33.58, 32.49, 31.42, 31.38, 30.36, 29.31, 28.26, 27.23, 26.15, 25.26, 24.37, 23.48, 22.84, 22.54, 21.49, 19.94, 18.02, 16.78, 15.42, 15.3, 14.05, 13.55, 12.62, 12.12, 11.38, 11.14, 10.24, 9.54, 8.87, 6.86, 6.05, 5.09, 4.67, 3.89, 3.28, 2.82, 2.0],
    105: [28.43, 27.56, 26.71, 25.9, 25.21, 25.06, 24.23, 23.42, 22.62, 22.58, 21.84, 21.09, 20.35, 19.58, 18.8, 18.07, 17.26, 16.49, 15.97, 15.71, 14.93, 13.79, 12.48, 11.66, 10.8, 10.72, 9.93, 9.59, 9.01, 8.65, 8.05, 7.83, 7.12, 6.57, 6.1, 4.91, 4.38, 3.83, 3.44, 2.63, 2.2, 1.91, 1.41],
    119: [30.38, 29.48, 28.6, 27.72, 27.01, 26.86, 26.01, 25.17, 24.34, 24.31, 23.52, 22.7, 21.89, 21.09, 20.26, 19.5, 18.7, 18.22, 17.94, 17.75, 17.26, 16.5, 15.2, 13.83, 12.41, 12.3, 11.17, 10.73, 10.01, 9.6, 8.97, 8.69, 8.05, 7.64, 7.09, 5.37, 4.83, 4.17, 3.81, 2.96, 1.38],
    127: [45.76, 44.4, 43.07, 41.75, 40.67, 40.45, 39.17, 37.91, 36.66, 36.61, 35.42, 34.19, 32.91, 31.72, 30.43, 29.41, 28.22, 26.91, 25.91, 25.46, 23.91, 21.78, 19.21, 17.66, 16.61, 16.51, 15.01, 14.49, 13.49, 12.81, 11.5, 11.07, 9.78, 8.99, 7.41, 5.94, 5.33, 4.23, 3.81, 3.25, 2.65, 1.99, 1.17],
    128: [28.43, 27.59, 26.76, 25.94, 25.28, 25.14, 24.34, 23.56, 22.78, 22.75, 22.01, 21.25, 20.49, 19.74, 18.96, 23.29, 22.42, 21.44, 20.72, 20.38, 19.24, 17.7, 16.15, 15.21, 14.29, 14.22, 13.34, 12.93, 12.22, 11.64, 10.82, 10.51, 9.59, 9.0, 8.55, 7.39, 6.03, 4.06, 3.51, 2.71, 2.27, 1.63, 1.05],
    136: [24.55, 23.83, 23.11, 22.4, 21.83, 21.68, 21.01, 20.29, 19.6, 19.57, 18.94, 18.31, 17.74, 17.35, 17.01, 19.78, 18.98, 18.17, 17.62, 17.37, 16.56, 15.41, 14.72, 9.6, 12.71, 13.92, 12.8, 12.47, 11.9, 11.51, 10.68, 10.36, 8.77, 8.2, 7.62, 5.88, 5.27, 4.53, 4.07, 3.32, 2.81, 2.31, 1.46],
    163: [22.13, 21.39, 20.79, 20.14, 19.66, 19.49, 18.86, 18.21, 17.59, 17.58, 17.03, 16.41, 15.8, 15.2, 14.58, 14.05, 13.44, 12.81, 12.36, 12.14, 11.5, 10.67, 10.55, 14.34, 13.42, 13.34, 12.48, 12.17, 11.64, 11.31, 10.61, 10.32, 9.45, 8.87, 8.28, 6.87, 6.3, 5.56, 5.18, 4.04, 3.24, 1.99, 1.58],
    212: [40.31, 39.12, 37.94, 36.78, 35.83, 35.64, 34.51, 33.4, 32.29, 32.25, 31.2, 30.12, 29.05, 27.99, 26.88, 25.87, 24.82, 23.77, 23.04, 22.71, 21.66, 20.16, 18.44, 17.16, 15.71, 15.6, 13.32, 12.69, 11.49, 10.93, 10.18, 9.92, 9.28, 8.53, 7.64, 5.37, 4.87, 4.23, 3.94, 3.51, 2.62, 1.95, 1.08],
    265: [11.97, 11.61, 11.26, 10.92, 10.64, 10.58, 10.24, 9.91, 9.59, 9.57, 9.26, 8.94, 8.62, 8.31, 7.98, 7.68, 7.37, 7.05, 6.84, 6.74, 6.43, 13.69, 15.13, 14.16, 13.24, 13.16, 12.31, 11.95, 11.26, 10.78, 10.07, 9.79, 8.92, 8.37, 7.78, 6.5, 5.8, 4.88, 4.45, 3.76, 3.14, 2.32, 1.05],
    278: [46.36, 44.99, 43.64, 42.3, 41.21, 40.99, 39.69, 38.41, 37.14, 37.09, 35.89, 34.64, 33.41, 32.18, 30.91, 29.75, 28.54, 27.33, 26.5, 26.12, 24.91, 21.32, 19.02, 17.82, 16.66, 16.55, 15.42, 14.96, 14.11, 13.52, 12.54, 12.19, 10.75, 10.09, 9.45, 7.63, 6.39, 4.72, 4.07, 3.31, 2.4, 1.88, 1.21],
    283: [46.91, 45.52, 44.15, 42.8, 41.7, 41.47, 40.16, 38.86, 37.58, 37.53, 36.31, 35.05, 33.8, 32.56, 31.27, 30.1, 28.88, 27.65, 26.64, 26.15, 24.68, 22.73, 20.65, 19.14, 17.8, 17.7, 16.21, 15.47, 14.25, 13.35, 12.03, 11.66, 10.2, 9.15, 8.4, 6.01, 5.1, 4.34, 4.06, 2.99, 1.99, 1.53],
    329: [43.75, 42.45, 41.18, 39.92, 38.89, 38.68, 37.45, 36.24, 35.05, 35.0, 33.86, 32.69, 31.53, 30.37, 29.17, 28.08, 26.93, 25.79, 25.01, 24.65, 23.5, 21.88, 20.01, 18.82, 17.61, 17.5, 16.37, 15.76, 14.94, 14.35, 11.69, 12.7, 10.82, 9.89, 9.07, 7.6, 6.93, 6.02, 5.7, 4.77, 3.31, 2.34, 1.58],
    372: [21.56, 20.92, 20.29, 19.67, 19.16, 19.06, 18.46, 17.86, 17.27, 17.26, 17.08, 16.73, 16.36, 14.82, 16.7, 16.41, 15.82, 15.27, 14.85, 14.67, 14.05, 13.24, 12.18, 11.43, 10.56, 10.48, 9.56, 9.3, 8.75, 8.36, 7.82, 7.63, 6.84, 6.19, 5.81, 5.11, 4.45, 3.55, 3.18, 2.65, 2.11, 1.66, 1.08],
    377: [21.56, 20.92, 20.29, 19.67, 19.16, 19.06],
    384: [21.87, 21.22, 20.58, 19.95, 19.44, 19.33, 18.72, 18.12, 17.52, 17.5, 16.93, 16.34, 15.76, 15.18, 14.58, 14.03, 13.46, 12.89, 12.52, 12.38, 11.99, 14.25, 13.03, 12.26, 11.47, 11.4, 10.64, 10.33, 9.76, 9.42, 8.88, 8.66, 8.03, 7.57, 7.05, 5.88, 5.35, 4.63, 4.16, 3.21, 2.82, 2.37, 1.19],
    386: [24.63, 23.9, 23.18, 22.47, 21.89, 21.77, 21.08, 20.4, 19.73, 19.7, 19.06, 18.4, 17.75, 17.1, 16.42, 15.8, 15.16, 14.61, 17.19, 16.94, 16.15, 15.04, 13.75, 12.94, 12.08, 12.01, 11.19, 10.88, 10.31, 9.95, 9.4, 9.19, 8.47, 8.0, 7.6, 6.46, 5.81, 4.86, 4.44, 3.59, 2.91, 2.12, 1.59],
    442: [38.85, 37.7, 36.57, 35.45, 34.54, 34.35, 33.26, 32.19, 31.12, 31.08, 30.1, 29.0, 27.89, 26.81, 25.71, 24.65, 23.56, 22.59, 21.84, 21.47, 20.47, 18.85, 16.11, 14.08, 12.83, 12.73, 11.6, 11.15, 10.38, 9.87, 9.1, 8.81, 8.0, 7.57, 7.13, 5.95, 5.39, 4.53, 4.06, 3.19, 2.41, 1.9, 1.36],
    491: [17.16, 16.65, 16.15, 15.66, 15.25, 15.17, 14.69, 14.22, 13.75, 13.73, 13.28, 12.82, 12.37, 11.91, 11.44, 11.01, 10.55, 10.1, 9.8, 9.69, 9.2, 8.74, 10.62, 9.99, 9.35, 9.3, 8.79, 8.62, 8.25, 7.97, 7.47, 7.25, 6.55, 6.08, 5.67, 4.75, 4.38, 4.09, 3.89, 2.96, 2.53, 2.3, 1.81],
    620: [24.18, 23.47, 22.76, 22.07, 21.5, 21.38, 20.7, 20.04, 19.37, 19.35, 18.72, 18.07, 17.43, 16.79, 16.12, 15.52, 14.89, 14.26, 13.81, 13.62, 12.94, 11.96, 10.87, 10.19, 9.45, 9.39, 8.71, 8.41, 7.83, 7.5, 7.0, 6.8, 6.14, 5.65, 5.26, 4.27, 3.84, 3.23, 2.86, 2.29, 2.02, 1.75, 1.38],
    678: [18.73, 18.17, 17.62, 17.09, 16.65, 16.55, 16.03, 15.51, 15.0, 14.98, 14.49, 13.99, 13.49, 13.0, 12.48, 12.02, 11.53, 11.07, 10.86, 10.85, 13.85, 12.89, 11.79, 11.09, 12.96, 12.86, 11.8, 11.36, 10.72, 10.24, 9.52, 9.25, 8.22, 7.54, 7.04, 6.04, 5.09, 3.48, 3.12, 2.44, 2.08, 1.79, 1.29],
    699: [13.31, 12.74, 12.42, 11.96, 11.65, 11.57, 11.22, 10.82, 10.51, 10.51, 10.23, 10.01, 9.92, 11.46, 11.01, 10.6, 10.17, 9.8, 9.57, 9.5, 9.42, 9.47, 10.49, 9.96, 9.44, 9.4, 9.19, 9.17, 8.98, 3.75, 9.14, 8.99, 8.37, 7.86, 7.31, 6.18, 5.66, 4.85, 4.07, 2.77, 1.97, 1.65, 1.21],
    759: [12.99, 12.64, 12.28, 11.9, 11.54, 11.45, 11.18, 10.78, 11.08, 11.06, 10.7, 10.33, 9.96, 9.6, 9.22, 8.87, 8.51, 8.1, 7.86, 7.71, 7.38, 6.88, 6.63, 6.58, 4.63, 5.85, 6.87, 6.67, 6.23, 3.53, 5.71, 5.56, 5.0, 4.67, 4.45, 3.58, 2.84, 2.39, 2.07],
    914: [13.47, 13.07, 12.68, 12.29, 11.98, 11.91, 11.53, 11.16, 10.79, 10.78, 10.43, 10.07, 9.71, 9.35, 8.98, 8.65, 8.29, 7.94, 7.79, 7.67, 7.32, 6.83, 6.15, 5.72, 5.32, 5.28, 4.96, 4.83, 4.54, 4.39, 4.17, 4.07, 3.63, 3.28, 3.0, 2.46, 2.19],
    1004: [74.27, 72.01, 69.88, 67.53, 65.99, 65.77, 63.17, 61.48, 59.54, 59.89, 57.74, 55.66, 53.33, 51.22, 48.3, 45.28],
    1024: [79.64, 76.38, 73.75, 71.37, 69.33, 69.06, 66.12, 63.43, 61.28, 61.1, 58.49, 56.85, 54.02]
}

path= '/home/christenc/Data/Sims/h148.cosmo50PLK.3072g/h148.cosmo50PLK.3072g3HbwK1BH/snapshots_200bkgdens/'

for key in list(haloids.keys()):
    if len(haloids[key]) != len(snapshots):
        for i in range(len(snapshots)-len(haloids[key])):
            haloids[key].append(0)
                        
print(haloids)

savepath = '/home/akinshol/Data/Timescales/DataFiles/h148.data'
if os.path.exists(savepath):
    os.remove(savepath)
    print('Removed previous .data file')
print(f'Saving to {savepath}')

for tstep in range(len(snapshots)):
    bulk_processing(tstep, haloids, rvirs, snapshots, path, savepath)
