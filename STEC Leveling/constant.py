

c 	=			299792458	                    #speed of light [m/s]
AE84  =         6378137.0                       #earth radius [m]
FLAT84  =       0.00335281066474	            #flattening
FLATINV84  =   	298.257223563


d_f1	      =	 2036.25*1000000.0			    	    # frequency on L1 [hz]
d_f2 	      =	 401.25*1000000.0			            # frequency on L2 [hz]
d_lambda1 	  =	 c/d_f1							        # wavelength of L1 [m] = 0.147m
d_lambda2 	  =	 c/d_f2							        # wavelength on L2 [m] = 0.747m
d_geomcorrJa2 =	 0.164						            # geometrical correction Jason-2 [m]
d_geomcorrJa3 =	 0.168						            # geometrical correction Jason-3 [m]
d_geomcorrCr2 =  0.1538							        # geometrical correction Cryosat-2 [m]
d_geomcorrHy2 =  0.162								    # geometrical correction Hy-2a [m]
d_geomcorrSar =  0.158								    # geometrical correction Saral [m]
d_STAREC      =  0.487									# STAREC Antenna
d_ALCATEL     =  0.175									# ALCATEL Antenna
d_k	          =	 (d_f2*d_f2)/((d_f2*d_f2)-(d_f1*d_f1))	# L4 frequency factor [hz²] = 1.040
d_w 	      =	 (d_f1*d_f1)/(40.3)			            # dion [range] * w	= TEC;