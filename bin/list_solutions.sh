#!/bin/bash

(printf "cs    %s    %s    %s    %s\n" $(cat cs_index*/*.solution);\
printf "pm    %s    %s    %s    %s\n" $(cat pm_index*/*.solution);\
printf "ppmm  %s    %s    %s    %s\n" $(cat ppmm_index*/*.solution);\
printf "pmpm  %s    %s    %s    %s\n" $(cat pmpm_index*/*.solution)
) | sort -k2 -g | cat -n | sort -k3 -gr 
