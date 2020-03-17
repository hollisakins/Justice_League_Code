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
                                          
path= '/home/christenc/Data/Sims/h329.cosmo50PLK.3072g/h329.cosmo50PLK.3072gst5HbwK1BH/snapshots/'
snapshots = ['h329.cosmo50PLK.3072gst5HbwK1BH.004096', 
             'h329.cosmo50PLK.3072gst5HbwK1BH.004032', 
             'h329.cosmo50PLK.3072gst5HbwK1BH.003936', 
             'h329.cosmo50PLK.3072gst5HbwK1BH.003840', 
             'h329.cosmo50PLK.3072gst5HbwK1BH.003744', 
             'h329.cosmo50PLK.3072gst5HbwK1BH.003648', 
             'h329.cosmo50PLK.3072gst5HbwK1BH.003606', 
             'h329.cosmo50PLK.3072gst5HbwK1BH.003552', 
             'h329.cosmo50PLK.3072gst5HbwK1BH.003456', 
             'h329.cosmo50PLK.3072gst5HbwK1BH.003360', 
             'h329.cosmo50PLK.3072gst5HbwK1BH.003264', 
             'h329.cosmo50PLK.3072gst5HbwK1BH.003195', 
             'h329.cosmo50PLK.3072gst5HbwK1BH.003168', 
             'h329.cosmo50PLK.3072gst5HbwK1BH.003072',
             'h329.cosmo50PLK.3072gst5HbwK1BH.002976', 
             'h329.cosmo50PLK.3072gst5HbwK1BH.002880', 
             'h329.cosmo50PLK.3072gst5HbwK1BH.002784', 
             'h329.cosmo50PLK.3072gst5HbwK1BH.002688', 
             'h329.cosmo50PLK.3072gst5HbwK1BH.002592', 
             'h329.cosmo50PLK.3072gst5HbwK1BH.002554', 
             'h329.cosmo50PLK.3072gst5HbwK1BH.002496', 
             'h329.cosmo50PLK.3072gst5HbwK1BH.002400', 
             'h329.cosmo50PLK.3072gst5HbwK1BH.002304',
             'h329.cosmo50PLK.3072gst5HbwK1BH.002208', 
             'h329.cosmo50PLK.3072gst5HbwK1BH.002112', 
             'h329.cosmo50PLK.3072gst5HbwK1BH.002088', 
             'h329.cosmo50PLK.3072gst5HbwK1BH.002016', 
             'h329.cosmo50PLK.3072gst5HbwK1BH.001920', 
             'h329.cosmo50PLK.3072gst5HbwK1BH.001824',
             'h329.cosmo50PLK.3072gst5HbwK1BH.001740',
             'h329.cosmo50PLK.3072gst5HbwK1BH.001728',
             'h329.cosmo50PLK.3072gst5HbwK1BH.001632', 
             'h329.cosmo50PLK.3072gst5HbwK1BH.001536', 
             'h329.cosmo50PLK.3072gst5HbwK1BH.001475', 
             'h329.cosmo50PLK.3072gst5HbwK1BH.001440', 
             'h329.cosmo50PLK.3072gst5HbwK1BH.001344', 
             'h329.cosmo50PLK.3072gst5HbwK1BH.001269', 
             'h329.cosmo50PLK.3072gst5HbwK1BH.001248',
             'h329.cosmo50PLK.3072gst5HbwK1BH.001152', 
             'h329.cosmo50PLK.3072gst5HbwK1BH.001106', 
             'h329.cosmo50PLK.3072gst5HbwK1BH.001056', 
             'h329.cosmo50PLK.3072gst5HbwK1BH.000974', 
             'h329.cosmo50PLK.3072gst5HbwK1BH.000960',
             'h329.cosmo50PLK.3072gst5HbwK1BH.000864', 
             'h329.cosmo50PLK.3072gst5HbwK1BH.000776', 
             'h329.cosmo50PLK.3072gst5HbwK1BH.000768', 
             'h329.cosmo50PLK.3072gst5HbwK1BH.000672', 
             'h329.cosmo50PLK.3072gst5HbwK1BH.000637', 
             'h329.cosmo50PLK.3072gst5HbwK1BH.000576', 
             'h329.cosmo50PLK.3072gst5HbwK1BH.000480', 
             'h329.cosmo50PLK.3072gst5HbwK1BH.000456', 
             'h329.cosmo50PLK.3072gst5HbwK1BH.000384', 
             'h329.cosmo50PLK.3072gst5HbwK1BH.000347', 
             'h329.cosmo50PLK.3072gst5HbwK1BH.000288', 
             'h329.cosmo50PLK.3072gst5HbwK1BH.000275', 
             'h329.cosmo50PLK.3072gst5HbwK1BH.000225', 
             'h329.cosmo50PLK.3072gst5HbwK1BH.000192', 
             'h329.cosmo50PLK.3072gst5HbwK1BH.000188', 
             'h329.cosmo50PLK.3072gst5HbwK1BH.000139', 
             'h329.cosmo50PLK.3072gst5HbwK1BH.000107', 
             'h329.cosmo50PLK.3072gst5HbwK1BH.000096', 
             'h329.cosmo50PLK.3072gst5HbwK1BH.000071']

# haloids dictionary defines the major progenitor branch back through all snapshots for each z=0 halo we are 
# interest in ... read this as haloids[1] is the list containing the haloid of the major progenitors of halo 1
# so if we have three snapshots, snapshots = [4096, 2048, 1024] then we would have haloids[1] = [1, 2, 5] 
# --- i.e. in snapshot 2048 halo 1 was actually called halo 2, and in 1024 it was called halo 5
haloids = {
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

    savepath = f'/home/akinshol/Data/Timescales/DataFiles/h329.data'
    if os.path.exists(savepath):
        os.remove(savepath)
        print('Removed previous .data file')
    print(f'Saving to {savepath}')
	
    for tstep in range(len(snapshots)):
        bulk_processing(tstep)

