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
                                          
path= '/home/christenc/Data/Sims/h229.cosmo50PLK.3072g/h229.cosmo50PLK.3072gst5HbwK1BH/snapshots/'
snapshots = ['h229.cosmo50PLK.3072gst5HbwK1BH.004096', 
             'h229.cosmo50PLK.3072gst5HbwK1BH.004032', 
             'h229.cosmo50PLK.3072gst5HbwK1BH.003936', 
             'h229.cosmo50PLK.3072gst5HbwK1BH.003840', 
             'h229.cosmo50PLK.3072gst5HbwK1BH.003744', 
             'h229.cosmo50PLK.3072gst5HbwK1BH.003648', 
             'h229.cosmo50PLK.3072gst5HbwK1BH.003606', 
             'h229.cosmo50PLK.3072gst5HbwK1BH.003552', 
             'h229.cosmo50PLK.3072gst5HbwK1BH.003456', 
             'h229.cosmo50PLK.3072gst5HbwK1BH.003360', 
             'h229.cosmo50PLK.3072gst5HbwK1BH.003264', 
             'h229.cosmo50PLK.3072gst5HbwK1BH.003195', 
             'h229.cosmo50PLK.3072gst5HbwK1BH.003168', 
             'h229.cosmo50PLK.3072gst5HbwK1BH.003072',
             'h229.cosmo50PLK.3072gst5HbwK1BH.002976', 
             'h229.cosmo50PLK.3072gst5HbwK1BH.002880', 
             'h229.cosmo50PLK.3072gst5HbwK1BH.002784', 
             'h229.cosmo50PLK.3072gst5HbwK1BH.002688', 
             'h229.cosmo50PLK.3072gst5HbwK1BH.002592', 
             'h229.cosmo50PLK.3072gst5HbwK1BH.002554', 
             'h229.cosmo50PLK.3072gst5HbwK1BH.002496', 
             'h229.cosmo50PLK.3072gst5HbwK1BH.002400', 
             'h229.cosmo50PLK.3072gst5HbwK1BH.002304',
             'h229.cosmo50PLK.3072gst5HbwK1BH.002208', 
             'h229.cosmo50PLK.3072gst5HbwK1BH.002112', 
             'h229.cosmo50PLK.3072gst5HbwK1BH.002088', 
             'h229.cosmo50PLK.3072gst5HbwK1BH.002016', 
             'h229.cosmo50PLK.3072gst5HbwK1BH.001920', 
             'h229.cosmo50PLK.3072gst5HbwK1BH.001824',
             'h229.cosmo50PLK.3072gst5HbwK1BH.001740',
             'h229.cosmo50PLK.3072gst5HbwK1BH.001728',
             'h229.cosmo50PLK.3072gst5HbwK1BH.001632', 
             'h229.cosmo50PLK.3072gst5HbwK1BH.001536', 
             'h229.cosmo50PLK.3072gst5HbwK1BH.001475', 
             'h229.cosmo50PLK.3072gst5HbwK1BH.001440', 
             'h229.cosmo50PLK.3072gst5HbwK1BH.001344', 
             'h229.cosmo50PLK.3072gst5HbwK1BH.001269', 
             'h229.cosmo50PLK.3072gst5HbwK1BH.001248',
             'h229.cosmo50PLK.3072gst5HbwK1BH.001152', 
             'h229.cosmo50PLK.3072gst5HbwK1BH.001106', 
             'h229.cosmo50PLK.3072gst5HbwK1BH.001056', 
             'h229.cosmo50PLK.3072gst5HbwK1BH.000974', 
             'h229.cosmo50PLK.3072gst5HbwK1BH.000960',
             'h229.cosmo50PLK.3072gst5HbwK1BH.000864', 
             'h229.cosmo50PLK.3072gst5HbwK1BH.000776', 
             'h229.cosmo50PLK.3072gst5HbwK1BH.000768', 
             'h229.cosmo50PLK.3072gst5HbwK1BH.000672', 
             'h229.cosmo50PLK.3072gst5HbwK1BH.000637', 
             'h229.cosmo50PLK.3072gst5HbwK1BH.000576', 
             'h229.cosmo50PLK.3072gst5HbwK1BH.000480', 
             'h229.cosmo50PLK.3072gst5HbwK1BH.000456', 
             'h229.cosmo50PLK.3072gst5HbwK1BH.000384', 
             'h229.cosmo50PLK.3072gst5HbwK1BH.000347', 
             'h229.cosmo50PLK.3072gst5HbwK1BH.000288', 
             'h229.cosmo50PLK.3072gst5HbwK1BH.000275', 
             'h229.cosmo50PLK.3072gst5HbwK1BH.000225', 
             'h229.cosmo50PLK.3072gst5HbwK1BH.000192', 
             'h229.cosmo50PLK.3072gst5HbwK1BH.000188', 
             'h229.cosmo50PLK.3072gst5HbwK1BH.000139', 
             'h229.cosmo50PLK.3072gst5HbwK1BH.000107', 
             'h229.cosmo50PLK.3072gst5HbwK1BH.000096', 
             'h229.cosmo50PLK.3072gst5HbwK1BH.000071']

# haloids dictionary defines the major progenitor branch back through all snapshots for each z=0 halo we are 
# interest in ... read this as haloids[1] is the list containing the haloid of the major progenitors of halo 1
# so if we have three snapshots, snapshots = [4096, 2048, 1024] then we would have haloids[1] = [1, 2, 5] 
# --- i.e. in snapshot 2048 halo 1 was actually called halo 2, and in 1024 it was called halo 5
haloids = {
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



def bulk_processing(tstep):
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
    
    print(f'Gathered {len(current_haloids)} haloids')
    
    h1id = haloids[1][tstep]
    # get h1 properties
    h1d = np.array([hd[h1id].properties['Xc'], hd[h1id].properties['Yc'], hd[h1id].properties['Zc']]) # halo 1 position
    h1r = hd[h1id].properties['Rvir'] # halo 1 virial radius
    h1v = np.array([hd[h1id].properties['VXc'], hd[h1id].properties['VYc'], hd[h1id].properties['VZc']]) # halo 1 velocity
    
    pynbody.analysis.angmom.faceon(h[h1id])
    p = pynbody.analysis.profile.Profile(s.g, min=0.01, max=10*h1r, ndim=3) # make gas density profile
    print('\t Made gas density profile for halo 1 (technically halo %s)' % h1id)
    rbins = p['rbins']
    density = p['density']
    
    for i,z0haloid in zip(current_haloids,z0_haloids):
        print('Major progenitor halod ID:', i)
        halo = h.load_copy(i)        
        properties = hd[i].properties
        rvir = properties['Rvir']
                
        # compute ram pressure on halo from halo 1
        # first compute distance to halo 1

        v_halo = np.array([properties['VXc'],properties['VYc'],properties['VZc']]) # halo i velocity
        v = v_halo - h1v

        d = np.array([properties['Xc'],properties['Yc'],properties['Zc']]) # halo i position
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
                p = pynbody.analysis.profile.Profile(h[i].g,min=.01,max=properties['Rvir'])
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
            dL = .1*properties['Rvir']/properties['h']

            #select the particles in a shell 0.25*Rvir
            inner_sphere2 = pynbody.filt.Sphere(str(.2*properties['Rvir']) + ' kpc', [0,0,0])
            outer_sphere2 = pynbody.filt.Sphere(str(.3*properties['Rvir']) + ' kpc', [0,0,0])
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
            inner_sphereL = pynbody.filt.Sphere(str(.9*properties['Rvir']) + ' kpc', [0,0,0])
            outer_sphereL = pynbody.filt.Sphere(str(properties['Rvir']) + ' kpc', [0,0,0])
            shell_partL = halo[outer_sphereL & ~inner_sphereL].gas

            #Perform calculations
            DD = .1*properties['Rvir']/properties['h']
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
                'haloid': i,
                'z0haloid':z0haloid,
                'mstar': mstar,
                'mgas': mgas,
                'mass':mass,
                'Rvir': rvir,
                'sfr':sfr,
                'Pram': Pram, 
                'Prest': Prest, 
                'v_halo':v_halo,
                'v_halo1':h1v,
                'v_env':env_vel,
                'env_rho':env_rho,
                'ratio': ratio, 
                'h1dist': d/h1r, 
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

    for key in list(haloids.keys()):
        if len(haloids[key]) != len(snapshots):
            for i in range(len(snapshots)-len(haloids[key])):
                haloids[key].append(0)
                            
    print(haloids)

    savepath = f'/home/akinshol/Data/Timescales/DataFiles/h229.data'
    if os.path.exists(savepath):
        os.remove(savepath)
        print('Removed previous .data file')
    print(f'Saving to {savepath}')
	
    for tstep in range(len(snapshots)):
        bulk_processing(tstep)

