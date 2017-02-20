# ProKlaue Version 0.3.3
### A Python-Plugin for Maya Autodesk to support biomechanic research

ProKlaue is a biomechanic research project at the Faculty of Veterinary Medicine of the University of Leipzig (Saxony, Germany). The primary goal is to be able to align bone models in a repeatable and deterministic way, as well as to export comparable measures of each bone model over a whole animation for further analysis.

To support the necessary work and to provide some useful routines inside the 3D-modelling and animation program Maya Autodesk the plugin proKlaue was written, which uses the Python-API of Maya and Python-Libraries for numerical computation (numpy, scipy, V-HACD). The Plugin introduces not only different scripts which are registered as commands inside Maya (for easier usage) but also a number of useful functions for general mesh-based tasks. There are a few additional functions like calculation of the convex hull, delaunay triangulation, intersection volume of multiple mesh objects and a cleanup process where one can extract a specific shell of a mesh (to eliminate possible entrapments inside a bone model caused by e.g. air, vessels or imaging errors).

**key features:**
   * Alignment of arbitrary objects according to Eigenvectors of their covariance matrix
   * Export of position and rotation values for further analysis
   * Cleanup-Step to remove entrapments inside model and 'cut' through small tube-like structures (extract only a specific shell)
   * Volume-Calculation of 3D object (~100x faster than standard mel-command *computePolysetVolume*)
   * Approximation of 3D intersection volume using V-HACD convex decomposition (3D Tetrahedra-Tetrahedra collision and intersection methods)
   * Calculation of minimal/maximal axis parallel plane of an object over a whole animation
   * Using axis parallel planes to measure perpendicular distances to object (relief map data)

The full Plugin documentation (Sphinx) can be found under doc/_build/html/index.html.
