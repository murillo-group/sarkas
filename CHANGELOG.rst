Sarkas 1.1.0 (2022-06-21)
=========================

Features
--------

- Added 2D capability. Added FMM algorithm. Added Open boundary conditions.
  Added 2D hexagonal and square lattice as initial particles distribution.
  Added gaussian as initial particle distribution.
  Added multithreading for I/O. Added a fdint.py containing all the Fermi-Dirac integrals. (#50)


Improved Documentation
----------------------

- Updated several docstrings.
  Updated the website/documentation with the new modifications and features. (#50)


Deprecations and Removals
-------------------------

- Removed the Thermostat class. It is incorporated into the Integrator class.
  Removed the method calculate_electron_properties from the Potential class and included it into the Parameters class. (#50)


Misc
----

- #50


Sarkas 1.1.0 (2022-06-13)
=========================

No significant changes.
