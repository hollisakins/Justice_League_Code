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
                                          
path= '/home/christenc/Data/Sims/h242.cosmo50PLK.3072g/h242.cosmo50PLK.3072gst5HbwK1BH/snapshots/'
snapshots = ['h242.cosmo50PLK.3072gst5HbwK1BH.004096', 
             'h242.cosmo50PLK.3072gst5HbwK1BH.004032', 
             'h242.cosmo50PLK.3072gst5HbwK1BH.003936', 
             'h242.cosmo50PLK.3072gst5HbwK1BH.003840', 
             'h242.cosmo50PLK.3072gst5HbwK1BH.003744', 
             'h242.cosmo50PLK.3072gst5HbwK1BH.003648', 
             'h242.cosmo50PLK.3072gst5HbwK1BH.003606', 
             'h242.cosmo50PLK.3072gst5HbwK1BH.003552', 
             'h242.cosmo50PLK.3072gst5HbwK1BH.003456', 
             'h242.cosmo50PLK.3072gst5HbwK1BH.003360', 
             'h242.cosmo50PLK.3072gst5HbwK1BH.003264', 
             'h242.cosmo50PLK.3072gst5HbwK1BH.003195', 
             'h242.cosmo50PLK.3072gst5HbwK1BH.003168', 
             'h242.cosmo50PLK.3072gst5HbwK1BH.003072',
             'h242.cosmo50PLK.3072gst5HbwK1BH.002976', 
             'h242.cosmo50PLK.3072gst5HbwK1BH.002880', 
             'h242.cosmo50PLK.3072gst5HbwK1BH.002784', 
             'h242.cosmo50PLK.3072gst5HbwK1BH.002688', 
             'h242.cosmo50PLK.3072gst5HbwK1BH.002592', 
             'h242.cosmo50PLK.3072gst5HbwK1BH.002554', 
             'h242.cosmo50PLK.3072gst5HbwK1BH.002496', 
             'h242.cosmo50PLK.3072gst5HbwK1BH.002400', 
             'h242.cosmo50PLK.3072gst5HbwK1BH.002304',
             'h242.cosmo50PLK.3072gst5HbwK1BH.002208', 
             'h242.cosmo50PLK.3072gst5HbwK1BH.002112', 
             'h242.cosmo50PLK.3072gst5HbwK1BH.002088', 
             'h242.cosmo50PLK.3072gst5HbwK1BH.002016', 
             'h242.cosmo50PLK.3072gst5HbwK1BH.001920', 
             'h242.cosmo50PLK.3072gst5HbwK1BH.001824',
             'h242.cosmo50PLK.3072gst5HbwK1BH.001740',
             'h242.cosmo50PLK.3072gst5HbwK1BH.001728',
             'h242.cosmo50PLK.3072gst5HbwK1BH.001632', 
             'h242.cosmo50PLK.3072gst5HbwK1BH.001536', 
             'h242.cosmo50PLK.3072gst5HbwK1BH.001475', 
             'h242.cosmo50PLK.3072gst5HbwK1BH.001440', 
             'h242.cosmo50PLK.3072gst5HbwK1BH.001344', 
             'h242.cosmo50PLK.3072gst5HbwK1BH.001269', 
             'h242.cosmo50PLK.3072gst5HbwK1BH.001248',
             'h242.cosmo50PLK.3072gst5HbwK1BH.001152', 
             'h242.cosmo50PLK.3072gst5HbwK1BH.001106', 
             'h242.cosmo50PLK.3072gst5HbwK1BH.001056', 
             'h242.cosmo50PLK.3072gst5HbwK1BH.000974', 
             'h242.cosmo50PLK.3072gst5HbwK1BH.000960',
             'h242.cosmo50PLK.3072gst5HbwK1BH.000864', 
             'h242.cosmo50PLK.3072gst5HbwK1BH.000776', 
             'h242.cosmo50PLK.3072gst5HbwK1BH.000768', 
             'h242.cosmo50PLK.3072gst5HbwK1BH.000672', 
             'h242.cosmo50PLK.3072gst5HbwK1BH.000637', 
             'h242.cosmo50PLK.3072gst5HbwK1BH.000576', 
             'h242.cosmo50PLK.3072gst5HbwK1BH.000480', 
             'h242.cosmo50PLK.3072gst5HbwK1BH.000456', 
             'h242.cosmo50PLK.3072gst5HbwK1BH.000384', 
             'h242.cosmo50PLK.3072gst5HbwK1BH.000347', 
             'h242.cosmo50PLK.3072gst5HbwK1BH.000288', 
             'h242.cosmo50PLK.3072gst5HbwK1BH.000275', 
             'h242.cosmo50PLK.3072gst5HbwK1BH.000225', 
             'h242.cosmo50PLK.3072gst5HbwK1BH.000192', 
             'h242.cosmo50PLK.3072gst5HbwK1BH.000188', 
             'h242.cosmo50PLK.3072gst5HbwK1BH.000139', 
             'h242.cosmo50PLK.3072gst5HbwK1BH.000107', 
             'h242.cosmo50PLK.3072gst5HbwK1BH.000096', 
             'h242.cosmo50PLK.3072gst5HbwK1BH.000071']

# haloids dictionary defines the major progenitor branch back through all snapshots for each z=0 halo we are 
# interest in ... read this as haloids[1] is the list containing the haloid of the major progenitors of halo 1
# so if we have three snapshots, snapshots = [4096, 2048, 1024] then we would have haloids[1] = [1, 2, 5] 
# --- i.e. in snapshot 2048 halo 1 was actually called halo 2, and in 1024 it was called halo 5
haloids = {
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

    savepath = f'/home/akinshol/Data/Timescales/DataFiles/h242.data'
    if os.path.exists(savepath):
        os.remove(savepath)
        print('Removed previous .data file')
    print(f'Saving to {savepath}')
	
    for tstep in range(len(snapshots)):
        bulk_processing(tstep)

