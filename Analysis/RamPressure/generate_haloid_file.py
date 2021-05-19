### This file takes the snapshot numbers and main progenitor halo IDs and writes them to a file
### That file can then be read in via the get_stored_filepaths_haloids(sim,z0haloid) function defined in base.py

import pickle

snapnums_h148 = ['004096', '003968', '003840', '003712', '003606', '003584', '003456', '003328', '003200', '003195', 
                 '003072', '002944', '002816', '002688', '002554', '002432', '002304', '002176', '002088', '002048', 
                 '001920', '001740', '001536', '001408', '001280', '001269', '001152', '001106', '001024', '000974', 
                 '000896', '000866', '000768', '000701', '000640', '000512', '000456', '000384', '000347', '000275', 
                 '000225', '000188', '000139']

snapnums_else = ['004096', '004032', '003936', '003840', '003744', '003648', '003606', '003552', '003456', '003360', 
                 '003264', '003195', '003168', '003072', '002976', '002880', '002784', '002688', '002592', '002554', 
                 '002496', '002400', '002304', '002208', '002112', '002088', '002016', '001920', '001824', '001740',
                 '001728', '001632', '001536', '001475', '001440', '001344', '001269', '001248', '001152', '001106', 
                 '001056', '000974', '000960', '000864', '000776', '000768', '000672', '000637', '000576', '000480', 
                 '000456', '000384', '000347', '000288', '000275', '000225', '000192', '000188', '000139', '000107', 
                 '000096', '000071'] 

filepaths_h148 = ['/home/christenc/Data/Sims/h148.cosmo50PLK.3072g/h148.cosmo50PLK.3072g3HbwK1BH/snapshots_200bkgdens/h148.cosmo50PLK.3072g3HbwK1BH.'+s for s in snapnums_h148]
filepaths_h229 = ['/home/christenc/Data/Sims/h229.cosmo50PLK.3072g/h229.cosmo50PLK.3072gst5HbwK1BH/snapshots_200bkgdens/h229.cosmo50PLK.3072gst5HbwK1BH.'+s for s in snapnums_else]
filepaths_h242 = ['/home/christenc/Data/Sims/h242.cosmo50PLK.3072g/h242.cosmo50PLK.3072gst5HbwK1BH/snapshots_200bkgdens/h242.cosmo50PLK.3072gst5HbwK1BH.'+s for s in snapnums_else]
filepaths_h329 = ['/home/christenc/Data/Sims/h329.cosmo50PLK.3072g/h329.cosmo50PLK.3072gst5HbwK1BH/snapshots_200bkgdens/h329.cosmo50PLK.3072gst5HbwK1BH.'+s for s in snapnums_else]

# haloids dictionary defines the major progenitor branch back through all snapshots for each z=0 halo we are 
# interest in ... read this as haloids[1] is the list containing the haloid of the major progenitors of halo 1
# so if we have three snapshots, snapshots = [4096, 2048, 1024] then we would have haloids[1] = [1, 2, 5] 
# --- i.e. in snapshot 2048 halo 1 was actually called halo 2, and in 1024 it was called halo 5

haloids_h148 = {
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

haloids_h229 = {
    1: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4, 13, 16],
    2: [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 5, 5, 5, 5, 4, 4, 4, 4, 3, 4, 4, 5, 5, 7, 15, 20, 23, 24, 23, 24, 48, 87],
    5: [5, 5, 6, 6, 6, 5, 5, 5, 5, 6, 4, 4, 4, 4, 4, 3, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 7, 7, 8, 10, 10, 10, 10, 9, 8, 10, 10, 14, 15, 15, 16, 18, 15, 19, 25, 25, 23, 32, 36, 29, 82],
    7: [7, 7, 7, 7, 8, 7, 6, 6, 7, 7, 7, 6, 6, 6, 7, 6, 7, 8, 8, 8, 7, 6, 6, 8, 8, 8, 8, 8, 7, 8, 8, 9, 9, 10, 11, 11, 11, 12, 12, 12, 12, 11, 12, 12, 21, 20, 24, 24, 28, 35, 38, 53, 59, 77, 87, 217, 234, 231],
    17: [17, 17, 16, 18, 18, 17, 17, 18, 19, 19, 20, 19, 20, 21, 23, 22, 25, 24, 24, 25, 23, 23, 26, 26, 26, 26, 28, 28, 28, 30, 30, 30, 30, 29, 29, 32, 38, 43, 42, 44, 41, 42, 43, 40, 42, 43, 57, 62, 68, 71, 77, 80, 81, 86, 93, 128, 149, 153, 198],
    20: [20, 19, 17, 17, 17, 15, 15, 15, 15, 15, 15, 15, 15, 14, 16, 13, 14, 14, 14, 15, 15, 13, 14, 13, 14, 14, 13, 13, 14, 13, 13, 13, 13, 13, 14, 15, 18, 19, 30, 30, 30, 33, 34, 42, 44, 45, 53, 53, 54, 55, 58, 70, 72, 71, 69, 63, 61, 63, 43, 36, 36, 25],
    22: [22, 22, 22, 22, 22, 21, 20, 21, 21, 21, 21, 20, 19, 18, 20, 18, 17, 16, 16, 14, 10, 10, 10, 11, 10, 10, 10, 10, 9, 9, 9, 8, 7, 7, 8, 9, 9, 10, 9, 9, 8, 8, 8, 9, 9, 8, 9, 9, 10, 9, 10, 9, 6, 17, 20, 20, 21, 22, 20, 40, 40],
    23: [23, 23, 23, 23, 23, 22, 23, 22, 23, 24, 25, 23, 23, 25, 26, 23, 22, 22, 21, 19, 19, 17, 17, 16, 16, 15, 15, 15, 15, 14, 14, 14, 14, 14, 15, 19, 22, 23, 25, 26, 27, 27, 28, 28, 32, 33, 37, 35, 43, 56, 56, 56, 61, 70, 86, 106, 135, 137, 182],
    27: [27, 27, 28, 28, 29, 30, 31, 31, 30, 30, 29, 30, 30, 30, 32, 32, 32, 31, 29, 30, 30, 26, 25, 25, 31, 27, 21, 20, 20, 20, 20, 20, 21, 20, 21, 21, 20, 21, 21, 21, 21, 22, 23, 23, 25, 23, 19, 20, 22, 26, 32, 33, 30, 36, 34, 60, 60, 61, 109],
    29: [29, 29, 30, 30, 32, 32, 33, 33, 32, 33, 33, 34, 34, 36, 38, 38, 39, 37, 37, 37, 37, 38, 38, 37, 36, 36, 38, 37, 37, 41, 41, 36, 34, 36, 36, 35, 35, 38, 37, 36, 36, 38, 40, 37, 39, 40, 47, 46, 50, 52, 55, 60, 64, 61, 62, 57, 49, 48, 40, 26, 20, 12],
    33: [33, 34, 34, 32, 34, 34, 21, 17, 17, 16, 16, 16, 16, 17, 19, 19, 21, 21, 22, 22, 22, 21, 22, 22, 22, 21, 22, 21, 21, 19, 19, 18, 19, 21, 25, 17, 16, 17, 19, 19, 19, 19, 20, 21, 22, 21, 21, 21, 23, 23, 26, 29, 28, 30, 31, 27, 26, 26, 17, 14, 12, 8],
    52: [52, 52, 51, 51, 51, 50, 50, 49, 48, 47, 47, 46, 45, 48, 50, 46, 45, 46, 48, 46, 44, 42, 41, 42, 38, 38, 39, 38, 38, 39, 38, 38, 36, 38, 39, 38, 39, 41, 40, 39, 38, 41, 42, 41, 43, 44, 52, 52, 56, 59, 60, 63, 62, 56, 56, 75, 68, 69, 33, 28, 46],
    53: [53, 54, 54, 69, 53, 52, 51, 50, 49, 48, 48, 47, 46, 47, 48, 47, 46, 47, 46, 44, 42, 40, 39, 41, 40, 39, 36, 31, 27, 28, 28, 26, 25, 26, 27, 27, 27, 28, 28, 28, 24, 24, 25, 69, 56, 56, 62, 63, 64, 82, 88, 95, 106, 279, 291, 277, 253, 245, 190],
    55: [55, 53, 52, 50, 49, 45, 44, 42, 39, 37, 35, 36, 36, 34, 33, 30, 24, 20, 18, 18, 18, 19, 20, 18, 17, 17, 17, 17, 16, 16, 16, 15, 15, 15, 17, 16, 15, 16, 18, 18, 18, 18, 18, 17, 20, 19, 18, 19, 21, 19, 22, 28, 96, 109, 108, 353],
    59: [59, 60, 58, 57, 59, 60, 61, 61, 60, 59, 58, 58, 57, 59, 60, 58, 61, 62, 64, 64, 64, 64, 65, 64, 65, 66, 65, 66, 64, 65, 63, 62, 57, 56, 56, 47, 44, 47, 46, 46, 52, 57, 59, 60, 67, 69, 72, 74, 81, 77, 83, 91, 98, 90, 92, 110, 359, 356],
    61: [61, 61, 62, 63, 64, 64, 64, 65, 65, 64, 66, 67, 65, 66, 67, 65, 67, 66, 67, 65, 67, 67, 66, 65, 66, 67, 70, 69, 71, 71, 71, 73, 76, 76, 76, 77, 81, 81, 81, 78, 80, 81, 84, 92, 100, 100, 109, 110, 111, 106, 120, 138, 170, 178, 181, 174, 145, 144, 118, 85, 70],
    62: [62, 62, 63, 62, 61, 59, 58, 58, 59, 57, 57, 55, 67, 41, 40, 39, 40, 39, 41, 40, 40, 41, 43, 46, 46, 47, 49, 48, 49, 51, 51, 53, 55, 53, 55, 53, 56, 56, 58, 61, 61, 69, 68, 77, 80, 81, 95, 102, 96, 92, 96, 115, 123, 189, 195, 171, 147, 148, 121, 104],
    73: [73, 72, 70, 71, 71, 68, 68, 68, 70, 66, 67, 63, 62, 65, 30, 27, 29, 28, 28, 29, 28, 30, 30, 28, 28, 29, 30, 30, 29, 31, 31, 28, 26, 25, 26, 26, 25, 26, 27, 27, 25, 25, 26, 26, 29, 28, 29, 29, 32, 37, 37, 41, 36, 38, 38, 44, 40, 39, 36, 103],
    104: [104, 106, 105, 107, 105, 105, 107, 109, 106, 102, 105, 105, 105, 106, 102, 98, 94, 87, 88, 86, 84, 84, 83, 81, 82, 81, 83, 82, 82, 82, 82, 80, 81, 80, 79, 74, 73, 73, 64, 66, 63, 58, 55, 49, 54, 54, 68, 70, 74, 72, 79, 90, 99, 110, 113, 105, 126, 128, 185],
    113: [113, 111, 104, 96, 87, 82, 78, 78, 77, 75, 77, 77, 75, 74, 73, 70, 60, 53, 52, 50, 50, 50, 40, 40, 37, 37, 40, 39, 39, 40, 37, 37, 35, 33, 32, 37, 29, 31, 26, 25, 26, 26, 27, 27, 31, 31, 34, 32, 39, 41, 41, 42, 34, 37, 36, 54, 87, 90],
    139: [139, 140, 137, 136, 137, 136, 137, 135, 134, 129, 135, 120, 111, 102, 100, 100, 102, 99, 101, 99, 98, 99, 99, 101, 103, 104, 105, 109, 102, 103, 103, 105, 104, 101, 104, 103, 106, 109, 111, 109, 112, 111, 113, 116, 121, 122, 125, 126, 131, 127, 132, 147, 155, 148, 130, 123, 202, 207, 155],
    212: [212, 212, 208, 200, 186, 178, 175, 171, 159, 85, 79, 80, 79, 81, 80, 77, 77, 73, 69, 66, 66, 65, 48, 35, 34, 34, 35, 35, 33, 35, 35, 32, 32, 31, 35, 29, 31, 34, 35, 35, 35, 37, 38, 34, 36, 36, 45, 41, 49, 49, 50, 49, 45, 58, 61, 89, 76, 76, 56, 53, 55],
    290: [290, 288, 287, 287, 284, 284, 286, 290, 295, 297, 306, 307, 310, 314, 313, 315, 322, 325, 328, 323, 318, 307, 287, 276, 281, 281, 345, 219, 222, 228, 229, 238, 261, 258, 267, 268, 269, 274, 281, 279, 283, 275, 278, 263, 251, 253, 294, 296, 301, 314, 321, 326, 330, 334, 329, 339, 343, 345],
    549: [549, 548, 537, 550, 522, 511, 501]
}

haloids_h242 = {
    1: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 4, 4, 4, 2],
    10: [10, 10, 9, 7, 7, 8, 7, 7, 7, 4, 5, 5, 5, 6, 6, 4, 3, 3, 3, 3, 3, 4, 5, 5, 6, 6, 4, 6, 6, 6, 6, 6, 5, 5, 5, 5, 5, 5, 5, 5, 4, 4, 4, 4, 4, 4, 4, 4, 6, 9, 9, 9, 10, 10, 9, 13, 14, 16, 18, 18, 18, 20],
    12: [12, 12, 12, 9, 10, 11, 10, 10, 10, 9, 10, 6, 6, 5, 4, 3, 4, 4, 4, 4, 4, 5, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 4, 4, 4, 4, 4, 5, 6, 6, 7, 8, 8, 8, 8, 10, 11, 10, 7, 6, 11, 11, 7, 11, 11, 9, 5, 6, 9],
    24: [24, 24, 25, 26, 25, 25, 25, 24, 24, 23, 24, 25, 24, 22, 22, 22, 21, 20, 19, 19, 19, 19, 19, 20, 19, 19, 17, 20, 21, 21, 20, 21, 17, 16, 17, 15, 15, 16, 15, 18, 17, 17, 17, 20, 21, 21, 20, 21, 24, 26, 28, 25, 24, 27, 29, 23, 20, 19, 14, 12, 16, 17],
    30: [30, 29, 29, 31, 29, 28, 29, 29, 30, 32, 30, 29, 28, 26, 25, 24, 22, 22, 20, 20, 20, 21, 21, 22, 21, 21, 18, 19, 18, 17, 16, 17, 15, 13, 14, 12, 13, 14, 13, 13, 13, 14, 14, 13, 15, 15, 14, 15, 16, 16, 17, 18, 16, 14, 12, 8, 9, 9, 8, 6, 5, 4],
    34: [34, 33, 34, 36, 35, 35, 34, 34, 35, 36, 33, 33, 33, 33, 33, 33, 33, 32, 32, 32, 32, 31, 31, 30, 30, 31, 28],
    40: [40, 40, 41, 44, 41, 41, 42, 41, 42, 42, 40, 39, 40, 39, 38, 38, 38, 38, 38, 38, 39, 38, 37, 38, 38, 38, 37, 36, 37, 35, 34, 34, 34, 32, 33, 34, 32, 34, 35, 38, 40, 38, 38, 42, 38, 38, 39, 40, 45, 50, 49, 49, 52, 57, 55, 50, 40, 41, 31, 25, 26],
    41: [41, 41, 42, 43, 44, 29, 26, 23, 20, 19, 19, 19, 19, 18, 18, 18, 19, 18, 18, 18, 18, 20, 20, 21, 22, 22, 21, 23, 28, 28, 28, 27, 29, 27, 28, 29, 27, 29, 32, 34, 35, 29, 29, 28, 26, 26, 34, 43, 121, 121, 115, 101, 103, 87, 82, 77, 122, 156, 122],
    44: [44, 44, 45, 48, 46, 45, 47, 46, 48, 50, 48, 50, 53, 51, 51, 52, 50, 52, 52, 53, 51, 51, 49, 50, 49, 50, 49, 50, 51, 48, 48, 45, 43, 39, 41, 41, 37, 39, 41, 46, 46, 46, 44, 46, 39, 39, 41, 41, 47, 53, 50, 46, 47, 50, 52, 53, 41, 40, 34, 30, 28, 25],
    48: [48, 50, 48, 51, 51, 50, 52, 51, 54, 54, 52, 53, 57, 55, 53, 53, 53, 54, 54, 54, 54, 53, 51, 49, 48, 48, 47, 51, 55, 54, 53, 57, 57, 57, 59, 60, 58, 61, 68, 68, 67, 67, 66, 69, 66, 66, 56, 57, 66, 67, 65, 59, 64, 71, 75, 85, 95, 94, 86, 77],
    49: [49, 49, 47, 50, 50, 49, 50, 50, 53, 55, 53, 55, 69, 56, 55, 55, 56, 55, 55, 57, 56, 58, 54, 57, 57, 57, 57, 60, 64, 65, 65, 69, 72, 73, 75, 79, 78, 83, 86, 86, 85, 81, 81, 83, 79, 79, 85, 94, 97, 100, 100, 103, 95, 92, 87, 91, 83, 84, 150],
    71: [71, 70, 72, 75, 75, 75, 75, 73, 72, 75, 71, 72, 72, 68, 66, 66, 67, 66, 66, 58, 57, 55, 53, 53, 50, 49, 48, 49, 53, 52, 51, 48, 48, 47, 50, 48, 42, 44, 46, 49, 48, 51, 53, 56, 50, 50, 48, 48, 56, 64, 62, 63, 65, 63, 58, 58, 55, 56, 47, 65],
    78: [78, 76, 76, 78, 76, 71, 70, 68, 64, 65, 63, 66, 65, 64, 62, 61, 62, 62, 60, 61, 59, 59, 60, 59, 59, 58, 56, 57, 60, 63, 62, 64, 62, 61, 62, 62, 60, 62, 67, 69, 68, 63, 63, 60, 45, 44, 47, 47, 54, 57, 51, 52, 62, 82, 77],
    80: [80, 268, 64, 39, 33, 31, 31, 30, 29, 29, 27, 26, 25, 25, 24, 19, 15, 14, 14, 14, 14, 14, 14, 14, 14, 14, 12, 14, 15, 14, 14, 13, 12, 12, 11, 11, 12, 13, 12, 14, 15, 15, 15, 15, 17, 17, 16, 17, 18, 21, 22, 29, 55, 53, 50, 47, 48, 48, 100],
    86: [86, 82, 83, 84, 82, 79, 80, 80, 76, 67, 51, 48, 47, 46, 47, 46, 46, 47, 45, 46, 45, 43, 42, 40, 39, 39, 38, 37, 39, 36, 35, 35, 35, 33, 34, 38, 35, 37, 36, 39, 42, 37, 35, 40, 41, 41, 51, 54, 58, 58, 56, 55, 57, 74, 73, 76, 88, 87, 112],
    165: [165, 153, 146, 140, 132, 125, 126, 129, 128, 134, 132, 131, 134, 128, 129, 127, 128, 127, 132, 132, 135, 136, 132, 125, 122, 116, 111, 113, 194, 133, 121, 87, 87, 92, 94, 97, 99, 102, 111, 108, 107, 102, 101, 102, 98, 101, 101, 105, 103, 95, 91, 90, 90, 91, 86, 68, 65, 64, 62],
    223: [223, 214, 228, 69, 52, 48, 48, 47, 49, 52, 50, 51, 54, 52, 52, 51, 51, 53, 53, 52, 50, 50, 48, 48, 51, 51, 50, 52, 54, 53, 52, 49, 50, 49, 52, 52, 47, 50, 49, 50, 50, 49, 49, 51, 44, 45, 49, 50, 55, 56, 54, 95, 134, 140, 135, 134, 157, 184, 146],
    439: [439, 441, 439, 442, 439, 432, 431, 430, 416, 401, 379, 367, 366, 341, 318, 283, 223, 176, 174, 170, 174, 173, 176, 178, 174, 173, 174, 176, 176, 176, 175, 168, 158, 141, 136, 122, 89, 78, 65, 63, 60, 53, 52, 43, 31, 31, 22, 22, 23, 27, 27, 26, 26, 30, 30, 31, 34, 36, 74],
    480: [480, 481, 466, 463, 462, 461, 459, 456, 460, 470, 467, 469, 472, 468, 472, 471, 460, 450, 437, 432, 425, 419, 415, 408, 409, 405, 408, 405, 405, 413, 411, 410, 404, 399, 393, 393, 368, 368, 356, 351, 342, 321, 320, 294, 224, 215, 118, 104, 89, 62, 57, 51, 54, 55, 54, 52, 49, 47, 37, 38, 42]
}

haloids_h329 = {
    1: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 5, 19, 16],
    11: [11, 8, 7, 7, 7, 7, 7, 6, 5, 6, 6, 5, 5, 6, 6, 7, 6, 6, 6, 6, 8, 8, 8, 7, 7, 7, 7, 7, 5, 5, 5, 6, 6, 6, 6, 6, 7, 7, 8, 8, 8, 9, 9, 10, 10, 10, 14, 13, 15, 16, 18, 20, 21, 20, 21, 24, 20, 21, 32],
    31: [31, 31, 31, 32, 32, 29, 28, 19, 15, 14, 15, 14, 14, 15, 14, 14, 14, 14, 14, 14, 13, 13, 13, 12, 11, 11, 11, 11, 11, 11, 11, 10, 10, 9, 10, 10, 19, 23, 24, 26, 27, 38, 39, 38, 41, 41, 48, 43, 46, 46, 49, 61, 62, 70, 73, 92, 84, 81, 106],
    33: [33, 33, 34, 34, 33, 33, 33, 35, 35, 34, 34, 32, 32, 32, 31, 31, 27, 27, 28, 32, 17, 16, 16, 16, 15, 14, 13, 15, 14, 14, 13, 12, 12, 12, 12, 15, 14, 16, 15, 14, 14, 15, 15, 14, 17, 17, 22, 69, 52, 50, 52, 65, 95, 179, 196, 189, 212, 205],
    40: [40, 39, 30, 28, 27, 27, 26, 27, 26, 27, 28, 27, 27, 29, 28, 29, 28, 28, 26, 27, 28, 27, 27, 28, 28, 28, 29, 29, 28, 27, 28, 27, 27, 28, 29, 29, 28, 29, 30, 28, 28, 23, 24, 24, 26, 28, 43, 41, 45, 41, 40, 54, 86, 91, 90, 74, 61, 60, 91],
    64: [64, 63, 63, 62, 61, 65, 64, 64, 60, 58, 59, 56, 57, 56, 52, 50, 47, 44, 45, 44, 43, 44, 41, 41, 40, 40, 40, 38, 39, 41, 41, 43, 43, 43, 42, 39, 44, 46, 45, 43, 44, 41, 41, 40, 53, 52, 54, 59, 70, 151, 151, 157, 159, 170, 176, 143, 99, 95, 77, 60],
    103: [103, 102, 103, 104, 107, 108, 109, 110, 108, 108, 107, 105, 106, 107, 109, 114, 124, 125, 111, 105, 98, 96, 99, 113, 96, 93, 81, 76, 76, 73, 75, 74, 72, 68, 68, 63, 61, 60, 48, 45, 41, 29, 28, 32, 33, 34, 41, 38, 39, 48, 50, 51, 49, 49, 50, 48, 67, 69, 64],
    133: [133, 132, 132, 128, 131, 129, 131, 130, 125, 125, 123, 117, 118, 118, 114, 100, 95, 95, 93, 92, 92, 87, 89, 84, 83, 84, 83, 78, 69, 64, 67, 66, 54, 38, 37, 32, 33, 33, 33, 31, 30, 27, 27, 25, 27, 26, 32, 33, 33, 33, 38, 47, 65, 130, 136, 219],
    137: [137, 138, 136, 137, 140, 136, 137, 138, 139, 139, 139, 137, 140, 143, 140, 133, 122, 122, 117, 113, 66, 55, 56, 53, 54, 54, 52, 52, 52, 46, 45, 44, 37, 31, 24, 14, 13, 15, 14, 13, 13, 16, 16, 15, 16, 16, 20, 21, 23, 23, 20, 23, 23, 22, 19, 19, 15, 15, 11, 20, 23],
    146: [146, 148, 148, 147, 148, 148, 148, 148, 150, 152, 151, 150, 154, 154, 155, 158, 157, 159, 157, 154, 154, 152, 152, 151, 154, 157, 158, 160, 155, 155, 156, 161, 162, 158, 155, 156, 160, 162, 165, 162, 156, 163, 164, 159, 156, 159, 161, 165, 163, 155, 153, 143, 145, 141, 141, 117, 111, 108, 62, 40, 28, 8],
    185: [185, 185, 183, 179, 181, 181, 178, 175, 174, 172, 169, 165, 166, 167, 166, 163, 159, 155, 146, 141, 137, 125, 119, 104, 97, 99, 94, 86, 83, 82, 80, 78, 73, 66, 66, 60, 59, 59, 55, 56, 53, 48, 47, 49, 59, 60, 58, 58, 60, 66, 65, 64, 61, 71, 69, 66, 64, 64, 44, 44, 43],
    447: [447, 438, 429, 409, 381, 359, 549, 434, 230, 191, 213]
}


output = dict(
    filepaths = dict(
        h148 = filepaths_h148,
        h229 = filepaths_h229,
        h242 = filepaths_h242,
        h329 = filepaths_h329
    ),
    haloids = dict(
        h148 = haloids_h148,
        h229 = haloids_h229,
        h242 = haloids_h242,
        h329 = haloids_h329
    )
)

with open('../../Data/filepaths_haloids.pickle','wb') as f:
    pickle.dump(output, f)