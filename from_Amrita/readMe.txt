The folder Au_100nm contains x-ray fluorescence (XRF) data from a 100 nm-thick Au film.

Folder name I_#nA_E_#keV corresponds to an ebeam current and ebeam energy. Current is not an input in PENELOPE.
The current actually corresponds to an ebeam probe diameter that can be found in the table beam_diameter_table_4mm.txt.
The columns of that table correspond to 30, 25, 20, 15, 10 and 5 kV acceleration voltages, and the rows correspond
to 0.01, 0.05, 0.1, 0.5, 1, 2, 5, 8, 10, and 100 nA.

Each folder I_#nA_E_#keV contains a file psf-test.dat. Column descriptions:

Col 1: particle type (1 = electron, 2 = photon, 3 = positron)

Col 2: particle energy (eV)

Col 3-5: particle position Col 3 = X, Col 4 = Y, Col 5 = Z (cm)

Col 6-8: unit vector specifying particle direction Col 6 = U, Col 7 = V, Col 8 = W (unitless)

Col 9: Weight (don't worry about this column)

Col 10: ILB(1) --> generation of the particle; 1 for primary particles, 2 for their direct descendants,
	etc. Primary (source) particles are assumed to be labelled
	with ILB(1)=1 in the main program.

Col 11: ILB(2) --> type KPAR of parent particle, only if ILB(1)>1 (secondary particles and
	\scattered" photons).

Col 12: ILB(3) --> interaction mechanism ICOL (see Table 7.5) that originated the particle,
	only when ILB(1)>1.

Col 13: ILB(4) --> a non-zero value identies particles emitted from atomic relaxation events
	and describes the atomic transition where the particle was released. The
	numerical value is = Z  106 + IS1  104 + IS2  100 + IS3, where Z is the
	atomic number of the emitting atom and IS1, IS2 and IS3 are the labels
	of the active atomic electron shells (see Table 7.2).
	For instance, ILB(4) = 29010300 designates a K-L2 x ray from copper
	(Z = 29), and ILB(4) = 29010304 indicates a K-L2-L3 Auger electron
	from the same element.

Col 14: NSHI --> incremental shower number, dened as the dierence between the shower
	numbers of the present particle and the one preceding it in the phase-space le
	(employed instead of the shower number to reduce the le size).

**ILB is explained on page 285 of the PENELOPE manual