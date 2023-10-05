.. Sarkas documentation master file, created by
   sphinx-quickstart on Mon Jun  1 10:34:03 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. meta::
   :format-detection: telephone=no
   :robots: index, follow
   :description: Sarkas: A Fast pure-Python Molecular Dynamics suite for Plasma Physics.
   :keywords: sarkas, plasma physics, plasma, physics, python, md, molecular dynamics
   :author: Stefano Silvestri, Ph.D.
   :designer: Stefano Silvestri, Ph.D.


.. the "raw" directive below is used to hide the title in favor of just the logo being visible

.. raw:: html
   <link rel="canonical" href="https://murillo-group.github.io/sarkas/">

   <link rel="preconnect" href="https://fonts.gstatic.com">
   <link href="https://fonts.googleapis.com/css2?family=RocknRoll+One&display=swap" rel="stylesheet">

   <style media="screen" type="text/css">
     h1 { display:none; }
     h2, h3 { font-family: 'RocknRoll One', sans-serif; }
   </style>


SARKAS: Python MD code for plasma physics
=========================================


.. grid:: 1 1 1 1
    :gutter: 1

    .. grid-item::
        :class: text-center

        .. grid:: 1 2 2 2
            :gutter: 1

            .. grid-item::

                .. raw:: html
                    
                    <h2>SARKAS</h2>
                    <h3>Python MD code for plasma physics</h3>

            .. grid-item::

                .. raw:: html

                    <picture>
                        <source srcset="_static/Sarkas_v1_for_dark_bg.svg" type="image/svg.xml">
                        <img src="_static/Sarkas_v1_for_dark_bg.svg" alt="logo">
                    </picture>

    .. grid-item::

        
        .. grid:: 1 2 2 2
            :gutter: 1

            .. grid-item::
                
                .. image:: _static/BYU.gif

            .. grid-item:: 

                .. image:: _static/codesnippet.png

.. grid:: 1 1 2 2

   .. grid-item-card:: User Friendly
      :class-card: border-0
      :shadow: none
      :class-header: sd-text-center sd-border-0
      :class-title: sd-text-center sd-fs-4

      :fa:`smile;fa-3x mb-3 text-muted`
      ^^^

      Run interactively in Jupyter notebook or via script. Set-up a simulation with only 3 lines of code. Run your simulation with 3 more lines. Calculate physics observables with final 3 lines.

   .. grid-item-card::  Fast Pure Python
      :class-card: border-0
      :shadow: none
      :class-header: sd-text-center sd-border-0
      :class-title: sd-text-center sd-fs-4
      
      :fa:`rocket;fa-3x mb-3 text-muted` 
      ^^^
      
      Sarkas offers the ease of use of Python while being highly performant with execution speeds comparable to that of compiled languages.

   .. grid-item-card::  Plasma Potentials
      :class-card: border-0
      :shadow: none
      :class-header: sd-text-center sd-border-0
      :class-title: sd-text-center sd-fs-4
      
      :fa:`flask;fa-3x mb-3 text-muted` 
      ^^^
      
      Sarkas offers a variety of interaction potentials commonly used in plasma physics. It is the only MD code to support electrons as dynamical particles.

   .. grid-item-card::  Data Science
      :class-card: border-0
      :shadow: none
      :class-header: sd-text-center sd-border-0
      :class-title: sd-text-center sd-fs-4
      
      :fa:`database;fa-3x mb-3 text-muted` 
      ^^^
      
      Sarkas has been developed for data science. You can run multiple simulations and store data with a simple for loop.

   .. grid-item-card::  Publications
      :class-card: border-0
      :shadow: none
      :class-header: sd-text-center sd-border-0
      :class-title: sd-text-center sd-fs-4
      
      :fa:`chart-area;fa-3x mb-3 text-muted` 
      ^^^
      
      Building upon a set of well-tested primitives and on a solid infrastructure, researchers can get publication-grade results in less time.

   .. grid-item-card::  Highly Customizable
      :class-card: border-0
      :shadow: none
      :class-header: sd-text-center sd-border-0
      :class-title: sd-text-center sd-fs-4
      
      :fa:`cogs;fa-3x mb-3 text-muted` 
      ^^^
      
      Sarkas is built in a modular fashion to allow easy implementation of additional features.

   .. grid-item-card::  Collaborative Effort
      :class-card: border-0
      :shadow: none
      :class-header: sd-text-center sd-border-0
      :class-title: sd-text-center sd-fs-4
      
      :fa:`users;fa-3x mb-3 text-muted` 
      ^^^
      
      Sarkas wants to be a common platform for the development of new algorithms to study the most challenging open problems in plasma physics.

   .. grid-item-card::  Open Source
      :class-card: border-0
      :shadow: none
      :class-header: sd-text-center sd-border-0
      :class-title: sd-text-center sd-fs-4
      
      :fa:`github;fa-3x mb-3 text-muted` 
      ^^^
      
      Sarkas is released under the MIT License and maintained by the community on GitHub.


.. grid:: 1 1 2 2
   :class-container: bg-light text-left
   :class-row: bg-light border-0

   .. grid-item-card::

      .. toctree::
         :maxdepth: 1
         :caption: Documentation:

         documentation/why_sarkas
         documentation/get_started
         documentation/features


      .. toctree::
         :maxdepth: 1
         :caption: Theory:

         theory/theory

      .. toctree::
         :maxdepth: 1
         :caption: Contributing:

         contributing/contributing


      .. toctree::
         :maxdepth: 1
         :caption: Simulations:

         examples/examples

   .. grid-item-card::

      .. toctree::
         :maxdepth: 1
         :caption: API:

         api/api

      .. toctree::
         :maxdepth: 1
         :caption: Credits:

         credits/credits

      Indices and tables:

      * :ref:`genindex`
      * :ref:`modindex`
      * :ref:`search`
      