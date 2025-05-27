c 	=			299792458	                    #speed of light [m/s]
AE84  =         6378137.0                       #earth radius [m]
FLAT84  =       0.00335281066474	            #flattening
FLATINV84  =   	298.257223563
omega      =    7.2921159e-5                    # rad/s
ion_height =    506700

d_f1	      =	 2036.25*1000000.0			    	    # frequency on L1 [hz]
d_f2 	      =	 401.25*1000000.0			            # frequency on L2 [hz]
d_lambda1 	  =	 c/d_f1							        # wavelength of L1 [m] = 0.147m
d_lambda2 	  =	 c/d_f2							        # wavelength on L2 [m] = 0.747m
d_p_multik    =  4/3*(87.0/5/2**26)                      # correction base for time frequency shift
d_geomcorrJa2 =	 0.164						            # geometrical correction Jason-2 [m]
d_geomcorrJa3 =	 0.168						            # geometrical correction Jason-3 [m]
d_geomcorrCr2 =  0.1538							        # geometrical correction Cryosat-2 [m]
d_geomcorrHy2 =  0.162								    # geometrical correction Hy-2a [m]
d_geomcorrSar =  0.158								    # geometrical correction Saral [m]
d_STAREC      =  0.487									# STAREC Antenna
d_ALCATEL     =  0.175									# ALCATEL Antenna
d_k	          =	 (d_f2**2)/((d_f2**2)-(d_f1**2))	    # L4 frequency factor 
d_w 	      =	 (d_f1**2)/(40.3)			            # dion [range] * w	= TEC;

iono_coeff    =  (d_f1 / d_f2) ** 2
jason3_L1_pco =  0.315
jason3_L2_pco =  0.147
starec_L1_pco =  0.487
starec_L2_pco =  0
alcatel_L1_pco =  0.510
alcatel_L2_pco =  0.175
starec_ja3_iono_comb_pco = (iono_coeff * (jason3_L1_pco + starec_L1_pco) - (jason3_L2_pco + starec_L2_pco)) / (iono_coeff - 1)
starec_ja3_iono_comb_pco = (iono_coeff * (jason3_L1_pco + alcatel_L1_pco) - (jason3_L2_pco + alcatel_L2_pco)) / (iono_coeff - 1)
