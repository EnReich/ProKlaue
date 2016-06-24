# ProKlaue
### A Python-Plugin for Maya Autodesk to support biomechanic research

ProKlaue is a biomechanic research project at the Faculty of Veterinary Medicine of the University of Leipzig (Saxony, Germany). The primary goal is to be able to align bone models in a repeatable and deterministic way, as well as to export comparable measures of each bone model over a whole animation for further analysis.

To support the necessary work and to provide some useful routines inside the 3D-modelling and animation program Maya Autodesk the plugin proKlaue was written, which uses the Python-API of Maya and Python-Libraries for numerical computation (numpy, scipy, V-HACD). The Plugin introduces not only different scripts which are registered as commands inside Maya (for easier usage) but also a number of useful functions for general mesh-based tasks. There are a few additional functions like calculation of the convex hull, delaunay triangulation, intersection volume of multiple mesh objects and a cleanup process where one can extract a specific shell of a mesh (to eliminate possible entrapments inside a bone model caused by e.g. air, vessels or imaging errors).

The full Plugin documentation (Sphinx) can be found under doc/_build/html/index.html ([preview](https://htmlpreview.github.io/?https://github.com/EnReich/ProKlaue/blob/master/doc/_build/html/index.html)).
