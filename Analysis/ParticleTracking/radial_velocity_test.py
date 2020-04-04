import pynbody
import pickle
import numpy as np
import pandas as pd

snapnums_h329 = ['004096', '004032', '003936', '003840', '003744', '003648', '003606', '003552', '003456', '003360', '003264', '003195', '003168', '003072','002976', '002880', '002784', '002688', '002592', '002554', '002496', '002400', '002304','002208', '002112', '002088', '002016', '001920', '001824','001740','001728','001632', '001536', '001475', '001440', '001344', '001269', '001248','001152', '001106', '001056', '000974', '000960','000864', '000776', '000768', '000672', '000637', '000576', '000480', '000456', '000384', '000347', '000288', '000275', '000225', '000192', '000188', '000139', '000107', '000096', '000071']

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

rvirs_h329 = {
    11: [62.0, 61.08, 59.71, 58.36, 57.02, 55.69, 55.21, 54.6, 53.45, 52.19, 51.03, 50.04, 49.57, 48.05, 46.67, 45.2, 43.78, 42.09, 39.97, 38.72, 36.19, 34.38, 33.03, 31.71, 30.55, 30.28, 29.41, 28.28, 27.19, 26.24, 26.09, 24.85, 23.62, 22.76, 22.32, 21.02, 20.06, 19.79, 18.54, 17.98, 17.37, 16.39, 16.19, 14.79, 13.54, 13.44, 12.07, 11.61, 10.88, 9.12, 8.61, 6.85, 5.95, 4.76, 4.42, 3.49, 2.91, 2.83, 1.69],
    31: [49.46, 48.73, 47.63, 46.55, 45.48, 44.43, 43.97, 43.38, 42.34, 41.32, 40.3, 39.57, 39.29, 38.28, 37.22, 36.14, 35.08, 33.98, 32.9, 32.48, 31.83, 30.64, 29.47, 28.46, 27.48, 27.24, 26.44, 25.33, 24.24, 23.31, 23.05, 22.27, 21.41, 20.66, 20.11, 19.08, 16.15, 15.68, 14.13, 13.57, 12.7, 10.25, 10.07, 9.03, 8.2, 8.11, 7.21, 6.89, 6.38, 5.64, 5.37, 4.18, 3.72, 3.03, 2.86, 2.07, 1.85, 1.82, 1.1],
    33: [35.93, 35.37, 34.55, 33.73, 32.95, 32.18, 31.86, 31.46, 30.8, 30.19, 29.66, 29.37, 29.24, 28.91, 34.66, 33.74, 32.83, 31.92, 31.01, 30.65, 30.11, 29.21, 28.31, 27.22, 26.21, 26.0, 25.23, 23.66, 22.75, 21.97, 21.84, 20.88, 19.88, 19.24, 18.83, 17.74, 16.92, 16.67, 15.61, 15.11, 14.58, 13.77, 13.62, 12.51, 11.16, 11.03, 9.91, 3.7, 5.3, 5.31, 5.15, 4.1, 3.14, 2.07, 1.95, 1.58, 1.3, 1.27],
    40: [40.02, 39.42, 38.54, 37.66, 36.8, 35.94, 35.57, 35.1, 34.26, 33.43, 32.56, 31.96, 31.73, 30.81, 29.9, 29.01, 28.1, 27.24, 26.34, 25.96, 25.42, 24.52, 23.56, 22.65, 21.77, 21.56, 20.91, 20.09, 19.35, 18.65, 18.54, 17.71, 16.79, 16.21, 15.88, 14.98, 14.24, 14.03, 13.1, 12.67, 12.26, 11.72, 11.64, 10.7, 9.45, 9.29, 7.35, 6.95, 6.35, 5.79, 5.53, 4.38, 3.28, 2.71, 2.59, 2.23, 2.09, 2.05, 1.15],
    64: [32.14, 31.66, 30.95, 30.25, 29.55, 28.87, 28.57, 28.19, 27.51, 26.85, 26.18, 25.71, 25.53, 24.88, 24.23, 23.59, 22.95, 22.31, 21.68, 21.43, 21.04, 20.4, 19.76, 19.13, 18.5, 18.36, 17.9, 17.28, 16.64, 16.06, 15.98, 15.31, 14.57, 14.11, 13.88, 13.05, 12.27, 12.07, 11.06, 10.73, 10.39, 9.76, 9.68, 8.77, 7.63, 7.58, 6.84, 6.32, 5.51, 3.56, 3.39, 2.82, 2.58, 2.11, 2.01, 1.76, 1.72, 1.69, 1.21, 0.9],
    103: [22.56, 22.3, 21.7, 21.22, 20.76, 20.28, 20.07, 19.81, 19.39, 18.96, 18.57, 18.29, 18.18, 17.81, 15.45, 12.92, 12.57, 12.22, 11.87, 11.74, 11.53, 11.18, 8.38, 4.1, 6.84, 8.19, 12.02, 13.63, 13.15, 12.72, 12.66, 12.13, 11.66, 11.38, 11.25, 10.98, 10.92, 10.93, 10.86, 10.66, 10.83, 11.07, 11.0, 9.9, 8.88, 8.79, 7.55, 7.2, 6.57, 5.53, 5.27, 4.47, 4.14, 3.55, 3.43, 2.8, 2.02, 1.94, 1.29],
    133: [24.29, 23.93, 23.39, 22.86, 22.33, 21.82, 21.59, 21.3, 20.79, 20.29, 19.79, 19.43, 19.29, 18.8, 18.31, 17.82, 17.34, 16.86, 16.38, 16.19, 15.9, 15.47, 15.03, 14.6, 14.18, 14.07, 13.81, 13.53, 17.11, 16.54, 16.45, 15.79, 15.12, 14.7, 14.45, 13.77, 13.35, 13.21, 12.59, 12.32, 11.93, 11.22, 11.1, 10.25, 9.27, 9.25, 8.6, 8.07, 7.33, 6.06, 5.62, 4.61, 3.65, 2.36, 2.2, 1.52],
    137: [19.65, 19.36, 18.93, 18.5, 18.07, 17.65, 17.47, 17.24, 16.83, 16.42, 16.1, 15.77, 15.67, 15.35, 21.84, 21.26, 20.68, 20.11, 19.54, 19.31, 18.97, 18.4, 17.83, 17.27, 16.7, 16.56, 16.15, 15.66, 15.28, 15.15, 21.45, 20.59, 19.72, 19.16, 18.84, 17.95, 17.31, 17.12, 16.04, 15.47, 14.84, 13.75, 13.6, 12.6, 11.65, 11.56, 10.27, 9.85, 9.1, 7.82, 7.43, 6.26, 5.76, 4.78, 4.58, 3.81, 3.37, 3.31, 2.32, 1.28, 1.0],
    146: [19.65, 19.36, 18.93, 18.5, 18.13, 17.75, 17.52, 17.28, 16.83, 16.47, 16.01, 15.8, 15.62, 15.24, 14.88, 14.39, 14.03, 13.58, 13.19, 13.04, 12.8, 12.37, 12.03, 11.58, 11.2, 11.09, 10.81, 10.4, 9.99, 9.65, 9.6, 9.18, 8.75, 8.49, 8.34, 7.9, 7.55, 7.45, 6.99, 6.78, 6.56, 6.17, 6.1, 5.66, 5.22, 5.17, 4.65, 4.44, 4.05, 3.5, 3.37, 2.92, 2.65, 2.27, 2.2, 1.9, 1.65, 1.63, 1.31, 1.03, 0.95],
    185: [25.07, 24.7, 24.14, 23.6, 23.05, 22.52, 22.29, 21.99, 21.46, 20.94, 20.43, 20.06, 19.91, 19.4, 18.9, 18.4, 17.9, 17.4, 16.91, 16.71, 16.42, 15.92, 15.43, 14.94, 14.45, 14.33, 13.96, 13.47, 12.97, 12.54, 12.47, 11.97, 11.66, 12.42, 12.21, 11.64, 11.18, 11.05, 10.46, 10.17, 9.81, 9.29, 9.18, 8.32, 7.35, 7.27, 6.57, 6.32, 5.83, 4.78, 4.54, 4.13, 3.75, 3.02, 2.93, 2.42, 2.05, 2.02, 1.51, 0.99, 0.86],
    447: [25.07, 24.7, 24.14, 23.6, 23.05, 22.52, 22.29, 21.99, 21.46, 20.94, 20.43]
}

def read_timescales():
    '''Function to read in the resulting data file which contains quenching and infall times'''
    data = []
    with open('/home/akinshol/Data/Timescales/QuenchingTimescales_sSFR_F19.data', 'rb') as f:
        while True:
            try:
                data.append(pickle.load(f,encoding='latin1'))
            except EOFError:
                break

    data = pd.DataFrame(data)
    return data

### SELECT WHICH HALOS YOU WANT TO GET DATA FOR
data = read_timescales()
data = data[(data.quenched==True)]
data['timescale'] = data.tinfall - data.tquench
data = data[data.timescale > 0]
data = data[(data.sim=='h329')&(data.haloid==137)]

print(f'Running for {len(data)} halos')
print(data)


age = 13.800797497330507
hubble =  0.6776942783267969


for sim, z0haloid, tinfall, tquench in zip(data.sim, data.haloid, data.tinfall, data.tquench):

    snapnums, haloids, rvirs = snapnums_h329, haloids_h329, rvirs_h329
    f_base = f'/home/christenc/Data/Sims/{sim}.cosmo50PLK.3072g/{sim}.cosmo50PLK.3072gst5HbwK1BH/snapshots_200bkgdens/{sim}.cosmo50PLK.3072gst5HbwK1BH.'

    lbts = np.array([])
    for i, snapnum in enumerate(snapnums):
        f = f_base + snapnums[i]
        s = pynbody.load(f)
        lbt = age - s.properties['time'].in_units('Gyr')
        lbts = np.append(lbts, lbt)
        

    i1 = np.argmin(np.abs(lbts - tinfall))+1
    # iend = np.argmin(lbts) irrelevant, we're only interested in i1 for now


    for i in [i1]: # np.flip(np.arange(iend, i1+1,1))
        f = f_base + snapnums[i]
        s = pynbody.load(f)
        s.physical_units()
        h = s.halos()
        t = s.properties['time'].in_units('Gyr')
        print(f'Snapshot {snapnums[i]}, t = {t:.2f}')

        host = h[haloids[1][i]]
        sat = h[haloids[z0haloid][i]]

        sat_x, sat_y, sat_z = sat.properties['Xc']/hubble, sat.properties['Yc']/hubble, sat.properties['Zc']/hubble
        host_x, host_y, host_z = host.properties['Xc']/hubble, host.properties['Yc']/hubble, host.properties['Zc']/hubble
        r_sat = np.array([sat_x, sat_y, sat_z])
        r_host = np.array([host_x, host_y, host_z])

        v_sat = np.array([sat.properties['VXc'],sat.properties['VYc'],sat.properties['VZc']])
        v_host = np.array(]host.properties['VXc'],host.properties['VYc'],host.properties['VZc']])

        v_rel = v_sat - v_host
        r_rel = r_sat - r_host

        h1dist = np.sqrt(np.dot(r_rel,r_rel))
        v_r = np.dot(v_rel, r_rel)/h1dist # magnitude of radial velocity vector
        # if v_r is negative then the satellite is moving towards halo 1

        print(v_r)
        print(h1dist)



        ### here would be where all the bridge stuff goes
        
        ### satellite is centered to do the stripping calculations in terms of r/rvir








