      subroutine attenuation(atten)

      use, intrinsic :: ISO_C_Binding, only: c_double, c_bool, c_int, c_float
      use module_atten
      use xraylib

      implicit none

      type (type_atten)    :: atten		! type defined in module_atten
      
      !--------------------------------------
      ! Inputs
      !--------------------------------------
      ! (*) atten%mu_case
      ! (*) atten%atomic_number
      ! (*) atten%photon_energy
      !--------------------------------------
      ! Outputs
      !--------------------------------------
      ! (*) atten%mu [for atten%mu_case \in [1,2,3,4,5]
      ! (*) atten%photo, atten%compton, atten%rayleigh, atten%total [for atten%mu_case = 5]


      select case (atten%mu_case)
      case (1)
!     Photoelectric effect using Kissel total photoelectric cross-sections
!     The alternative for the photoelectric contribution is to use CS_Photo
      atten%mu         = CS_Photo_Total(atten%atomic_number, atten%photon_energy)
      case (2)
!     Compton scattering				
      atten%mu         = CS_Compt(atten%atomic_number, atten%photon_energy)
      case (3)
!     Rayleigh scattering
      atten%mu         = CS_Rayl(atten%atomic_number, atten%photon_energy)
      case (4)
!     Sum of Photoelectric effect, Compton scattering and Rayleigh scattering
      atten%mu 	       =   CS_Total_Kissel(atten%atomic_number, atten%photon_energy)
      case (5)
!     All 4 possibilities 1 2 3 4 (for comparison, plotting, and debugging),
!     and, default assigned value of mu is selected as the sum of each possible interaction
      atten%photo      = CS_Photo_Total(  atten%atomic_number, 	atten%photon_energy )		
      atten%compton    = CS_Compt(	  atten%atomic_number, 	atten%photon_energy )
      atten%rayleigh   = CS_Rayl(	  atten%atomic_number, 	atten%photon_energy )
      atten%total      = CS_Total_Kissel( atten%atomic_number, 	atten%photon_energy )
      atten%mu         = atten%total
      end select 

!     write(*,*) 'atten%mu = ', atten%mu

      return
      end subroutine attenuation 


