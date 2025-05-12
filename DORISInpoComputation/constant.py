

c 	       =	299792458	                    #speed of light [m/s]
AE84       =    6378136.3                       #earth radius [m]
FLAT84     =    1.0/298.257	                    #flattening
FLATINV84  =   	298.257
e_omega    =    7.2921159e-5                    # rad/s, Earth’s rotation rate

d_f1	      =	 2036.25*1000000.0			    	    # frequency on L1 [hz]
d_f2 	      =	 401.25*1000000.0			            # frequency on L2 [hz]
d_lambda1 	  =	 c/d_f1							        # wavelength of L1 [m] = 0.147m
d_lambda2 	  =	 c/d_f2							        # wavelength on L2 [m] = 0.747m
d_p_multik    =  4/3*(87.0/5/2**26)                     # correction base for time frequency shift
d_geomcorrJa2 =	 0.164						            # geometrical correction Jason-2 [m]
d_geomcorrJa3 =	 0.168						            # geometrical correction Jason-3 [m]
d_geomcorrCr2 =  0.1538							        # geometrical correction Cryosat-2 [m]
d_geomcorrHy2 =  0.162								    # geometrical correction Hy-2a [m]
d_geomcorrSar =  0.158								    # geometrical correction Saral [m]
d_STAREC      =  0.487									# STAREC Antenna
d_ALCATEL     =  0.175									# ALCATEL Antenna
d_k	          =	 (d_f2**2)/((d_f2**2)-(d_f1**2))	    # L4 frequency factor 
d_w 	      =	 (d_f1**2)/(40.3)			            # dion [range] * w	= TEC;