#!/bin/bash/

python ./calculate_eccentricities.py \
--data-path /project2/kicp/michaelzevin/eccentricity/redshift_evolution/data/clusters_det.h5 \
--output-path './eccentricities_redshifting.hdf5' \
--fLow 10 3 \
--multiproc 8 \
#--calc-with-no-redshifting
