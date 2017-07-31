      subroutine example_xraylib_function_calls

!Copyright (c) 2009, 2010, 2011 Tom Schoonjans
!All rights reserved.


!Redistribution and use in source and binary forms, with or without
!modification, are permitted provided that the following conditions are met:
!    * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
!    * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
!    * The names of the contributors may not be used to endorse or promote products derived from this software without specific prior written permission.

!THIS SOFTWARE IS PROVIDED BY Tom Schoonjans ''AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL Tom Schoonjans BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

      use xraylib		! Set by xraylib package (not defined by BAE)
      implicit none 

      TYPE (compoundData_C) :: cd_C
      TYPE (compoundData_F) :: cd_F

      integer*4 i

!C_NULL_CHAR needs to be added since these strings will be passed to C functions
CHARACTER (KIND=C_CHAR,LEN=10) :: compound1 = C_CHAR_'Ca(HCO3)2'// C_NULL_CHAR
CHARACTER (KIND=C_CHAR,LEN=5) :: compound2 = C_CHAR_'SiO2'// C_NULL_CHAR

      print*, ' '
      print*, '---------------------------------------------'
      print*, 'EXAMPLE xraylib FUNCTION CALLS'
      print*, '---------------------------------------------'
      print*, ''

! These line identifiers are clearly interchangeble, and they are integers
  write(*,*)	KA1_LINE,    	KL3_LINE    
  write(*,*)	KA2_LINE,   	KL2_LINE    	
  write(*,*)	KB1_LINE,    	KM3_LINE    	
  write(*,*)	KB2_LINE,    	KN3_LINE    	
  write(*,*)	KB3_LINE,   	KM2_LINE    	
  write(*,*)	KB4_LINE,    	KN5_LINE    	
  write(*,*)	KB5_LINE,   	KM5_LINE    	
  write(*,*)	LA1_LINE, 	L3M5_LINE 	
  write(*,*)	LA2_LINE,	L3M4_LINE 	
  write(*,*)	LB1_LINE, 	L2M4_LINE 	
  write(*,*)	LB2_LINE, 	L3N5_LINE 	
  write(*,*)	LB3_LINE, 	L1M3_LINE 	
  write(*,*)	LB4_LINE,	L1M2_LINE 	
  write(*,*)	LB5_LINE, 	L3O45_LINE 	
  write(*,*)	LB6_LINE, 	L3N1_LINE 	
  write(*,*)	LB7_LINE,	L3O1_LINE 	
  write(*,*)	LB9_LINE, 	L1M5_LINE 	
  write(*,*)	LB10_LINE,	L1M4_LINE 	
  write(*,*)	LB15_LINE, 	L3N4_LINE 	
  write(*,*)	LB17_LINE, 	L2M3_LINE 	
  write(*,*)	LG1_LINE, 	L2N4_LINE 	
  write(*,*)	LG2_LINE,	L1N2_LINE
  write(*,*)	LG3_LINE,	L1N3_LINE 	
  write(*,*)	LG4_LINE, 	L1O3_LINE 	
  write(*,*)	LG5_LINE, 	L2N1_LINE 	
  write(*,*)	LG6_LINE, 	L2O4_LINE 	
  write(*,*)	LG8_LINE, 	L2O1_LINE 	
  write(*,*)	LE_LINE, 	L2M1_LINE 	
  write(*,*)	LL_LINE, 	L3M1_LINE 	
  write(*,*)	LS_LINE, 	L3M3_LINE 	
  write(*,*)	LT_LINE,	L3M2_LINE 	
  write(*,*)	LU_LINE, 	L3N6_LINE 	
  write(*,*)	LV_LINE, 	L2N6_LINE 	
  write(*,*)	MA1_LINE, 	M5N7_LINE 	
  write(*,*)	MA2_LINE, 	M5N6_LINE 	
  write(*,*)	MB_LINE, 	M4N6_LINE 	
  write(*,*)	MG_LINE, 	M3N5_LINE 	

      WRITE (6,'(A,F12.6)') 'Ca K-alpha Fluorescence Line Energy: ', &
            LineEnergy(20,KA_LINE)

      WRITE (6,'(A,F12.6)')                                          &
            'Fe partial photoionization cs of L3 at 6.0 keV: ',      &
            CS_Photo_Partial(26,L3_SHELL,6.0)

      WRITE (6,'(A,F12.6)') 'Zr L1 edge energy: ',                   &
            EdgeEnergy(40,L1_SHELL)

      WRITE (6,'(A,A,F12.6)') 'Pb Lalpha XRF production cs at ',      &
            '20.0 keV (jump approx): ',                              &
            CS_FluorLine(82,LA_LINE,20.0)

      WRITE (6,'(A,F12.6)') 'Pb Lalpha XRF production cs at 20.0 keV (Kissel): ', & 
            CS_FluorLine_Kissel(82,LA_LINE,20.0)

      WRITE (6,'(A,F12.6)') 'Bi M1N2 radiative rate: ', & 
            RadRate(83,M1N2_LINE)

      WRITE (6,'(A,F12.6)') 'U M3O3 Fluorescence Line Energy: ', &
             LineEnergy(92,M3O3_LINE);

!CompoundParser tests
      IF (CompoundParser(compound1,cd_C) == 0) THEN
        CALL EXIT(1)
      ENDIF

      CALL compoundDataAssoc(cd_C,cd_F)

      WRITE (6,'(A,I4,A,I4,A)') 'Ca(HCO3)2 contains ',cd_F%nAtomsAll, & 
             ' atoms and ',cd_F%nElements,' elements'

      DO i=1,cd_F%nElements
               WRITE (6,'(A,I2,A,F12.6,A)') 'Element ', & 
               cd_F%Elements(i), ' : ',                 &
               cd_F%massFractions(i)*100.0_C_DOUBLE,' %'
      ENDDO

!Free the memory allocated for the arrays
      DEALLOCATE(cd_F%Elements,cd_F%massFractions)

      IF (CompoundParser(compound2,cd_C) == 0) THEN
        CALL EXIT(1)
      ENDIF

      CALL compoundDataAssoc(cd_C,cd_F)

      WRITE (6,'(A,I4,A,I4,A)') 'SiO2 contains ',cd_F%nAtomsAll, &
      ' atoms and ',cd_F%nElements,' elements'
      DO i=1,cd_F%nElements
        WRITE (6,'(A,I2,A,F12.6,A)') 'Element ',cd_F%Elements(i),' : ',cd_F%massFractions(i)*100.0_C_DOUBLE,' %'
      ENDDO

      DEALLOCATE(cd_F%Elements,cd_F%massFractions)

      WRITE (6,'(A,F12.6)') 'Ca(HCO3)2 Rayleigh cs at 10.0 keV: ', &
            CS_Rayl_CP('Ca(HCO3)2'//C_NULL_CHAR,10.0)

      WRITE (6,'(A,ES14.6,A,ES14.6,A)') 'CS2 Refractive Index at 10.0 keV : ', &
        Refractive_Index_Re('CS2'//C_NULL_CHAR,10.0,1.261),' - ', &
        Refractive_Index_Im('CS2'//C_NULL_CHAR,10.0,1.261),' i'  

      WRITE (6,'(A,ES14.6,A,ES14.6,A)') 'C16H14O3 Refractive Index at 1 keV : ', &
        Refractive_Index_Re('C16H14O3'//C_NULL_CHAR,1.0,1.2),' - ', &
        Refractive_Index_Im('C16H14O3'//C_NULL_CHAR,1.0,1.2),' i'  

      WRITE (6,'(A,ES14.6,A,ES14.6,A)') 'SiO2 Refractive Index at 5.0 keV : ', &
        Refractive_Index_Re('SiO2'//C_NULL_CHAR,5.0,2.65),' - ', &
        Refractive_Index_Im('SiO2'//C_NULL_CHAR,5.0,2.65),' i'  




      WRITE (6,'(A,F12.6)') 'Compton profile for Fe at pz = 1.1 : ' ,&
        ComptonProfile(26,1.1) 

      WRITE (6,'(A,F12.6)') 'M5 Compton profile for Fe at pz = 1.1 : ' ,&
        ComptonProfile_Partial(26,M5_SHELL,1.1) 

      WRITE (6,'(A,F12.6)') 'K atomic level width for Fe: ',&
        AtomicLevelWidth(26,K_SHELL)

      WRITE (6,'(A,F12.6)') 'Bi L2-M5M5 Auger non-radiative rate: ',&
        AugerRate(86,L2_M5M5_AUGER)

      WRITE (6,'(A,F12.6)') 'M1->M5 Coster-Kronig transition probability for Au : ', &
            CosKronTransProb(79,FM15_TRANS)

      WRITE (6,'(A,F12.6)') 'L1->L3 Coster-Kronig transition probability for Fe : ', &
            CosKronTransProb(26,FL13_TRANS)

      WRITE (6,'(A,F12.6)') 'Au Ma1 XRF production cs at 10.0 keV (Kissel): ', & 
             CS_FluorLine_Kissel(79,MA1_LINE,10.0)

      WRITE (6,'(A,F12.6)') 'Au Mg XRF production cs at 10.0 keV (Kissel): ', &
            CS_FluorLine_Kissel(79,MG_LINE,10.0)

      WRITE (6,'(A,A,F12.6)') 'Pb Malpha XRF production cs at 20.0 keV ',     &
             'with cascade effect: ',                                         &
             CS_FluorLine_Kissel(82,MA1_LINE,20.0)

      WRITE (6,'(A,A,F12.6)') 'Pb Malpha XRF production cs at 20.0 keV ',      &
            ' with radiative cascade effect: ', &
             CS_FluorLine_Kissel_Radiative_Cascade(82,MA1_LINE,20.0)

      WRITE (6,'(A,A,F12.6)') 'Pb Malpha XRF production cs at 20.0 keV ', &
            ' with non-radiative cascade effect: ',                       &
            CS_FluorLine_Kissel_Nonradiative_Cascade(82,MA1_LINE,20.0)

      WRITE (6,'(A,F12.6)') 'Pb Malpha XRF production cs at 20.0 keV without cascade effect: ', &
             CS_FluorLine_Kissel_no_Cascade(82,MA1_LINE,20.0)

      WRITE (6,'(A,A)') 'Symbol of element 31 is: ', &
            AtomicNumberToSymbol(31)

      WRITE (6,'(A,A)') 'Symbol of element 33 is: ', &
            AtomicNumberToSymbol(33)

      WRITE (6,'(A,I3)') 'Number of element Fe is: ', &
            SymbolToAtomicNumber('Fe')

      return
      end subroutine example_xraylib_function_calls
