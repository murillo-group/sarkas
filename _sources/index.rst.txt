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


.. raw:: html
  
   <div class="full-width p-0 m-0">
      <div class="jumbotron jumbotron-fluid p-0 bg-white">
         <div class="container-fluid p-0 m-0">
            <div class="row justify-content-center">
               <div class="col-xs-12 col-sm-12 col-lg-8 text-center">
                  <h2 class="display-2 font-weight-bold text-break">SARKAS</h2>
                  <h3 class="lead font-weight-bold">Python MD code for plasma physics</h3>
               </div>
                  <img src="_static/Sarkas_v1_for_light_bg.svg" alt="logo" class="float-lg-right float-xs-none float-sm-none" width="25%">
                  <span class="float-none"></span>
            </div>
         </div>
      </div>

      <div class="row justify-content-center">
         <div id="slideshow" class="carousel slide" data-ride="carousel">
            <ol class="carousel-indicators">
               <li data-target="#slideshow" data-slide-to="0" class="active"></li>
               <li data-target="#slideshow" data-slide-to="1"></li>
            </ol>
            <div class="carousel-inner">
               <div class="carousel-item active mw-50">
                  <img src="_static/codesnippet.png" alt="Code Snippet">
               </div>
               <div class="carousel-item mw-25">
                  <img src="_static/BYU.gif" alt="BYU">
               </div>
            </div>
            <a class="carousel-control-prev" href="#slideshow" role="button" data-slide="prev">
               <span class="carousel-control-prev-icon" aria-hidden="true"></span>
               <span class="sr-only">Previous</span>
            </a>
            <a class="carousel-control-next" href="#slideshow" role="button" data-slide="next">
               <span class="carousel-control-next-icon" aria-hidden="true"></span>
               <span class="sr-only">Next</span>
            </a>
         </div>
      </div>

      <div class="row">
         <div class="text-center col-xs-12 col-sm-12 col-md-6">
            <h4 class="text-center"><i class="far fa-smile fa-3x mb-3 text-muted"></i><br>User Friendly</h4>
            <p>Run interactively in Jupyter notebook or via script. Set-up a simulation with only 3 lines
               of code. Run your simulation with 3 more lines. Calculate physics observables with final 3 lines.</p>
         </div>
         
         <div class="text-center col-xs-12 col-sm-12 col-md-6">
            <h4 class="text-center"><i class="fa fa-rocket fa-3x mb-3 text-muted"></i><br>Fast Pure Python</h4>
            <p>Sarkas offers the ease of use of Python while being highly performant with execution speeds comparable
            to that of compiled languages.</p>
         </div>

         <div class="text-center col-xs-12 col-sm-12 col-md-6">
            <h4 class="text-center"><i class="fa fa-flask fa-3x mb-3 text-muted"></i><br>Plasma Potentials</h4>
            <p>Sarkas offers a variety of interaction potentials commonly used in plasma physics. It is the only
            MD code to support electrons as dynamical particles.</p>
         </div>

         <div class="text-center col-xs-12 col-sm-12 col-md-6">
            <h4 class="text-center"><i class="fa fa-database fa-3x mb-3 text-muted"></i><br>Data Science</h4>
            <p>Sarkas has been developed for data science. You can run multiple simulations and store data with a simple for loop.</p>
         </div>
         
         <div class="text-center col-xs-12 col-sm-12 col-md-6">
            <h4 class="text-center"><i class="fa fa-chart-area fa-3x mb-3 text-muted"></i><br>Publications</h4>
            <p>Building upon a set of well-tested primitives and on a solid infrastructure, researchers can get
            publication-grade results in less time.</p>
         </div>

         <div class="text-center col-xs-12 col-sm-12 col-md-6">
            <h4 class="text-center"><i class="fa fa-cogs fa-3x mb-3 text-muted"></i><br>Highly Customizable</h4>
            <p>Sarkas is built in a modular fashion to allow easy implementation of additional features.</p>
         </div>
         
         <div class="text-center col-xs-12 col-sm-12 col-md-6">
            <h4 class="text-center"><i class="fa fa-users fa-3x mb-3 text-muted"></i><br>Collaborative Effort</h4>
            <p>Sarkas wants to be a common platform for the development of new algorithms to study the most challenging
            open problems in plasma physics.</p>
         </div>
         
         <div class="text-center col-xs-12 col-sm-12 col-md-6">
            <h4 class="text-center"><i class="fa fa-github fa-3x mb-3 text-muted"></i><br>Open Source</h4>
            <p>Sarkas is released under the MIT License and maintained by the community on GitHub.</p>
         </div>

      </div>

.. grid:: 1 1 2 2
   :class-container: bg-light text-left
   :class-row: bg-light border-0

   .. grid-item-card::

      .. toctree::
         :maxdepth: 1
         :caption: Documentation:

         documentation/why_sarkas
         documentation/get_started


      .. toctree::
         :maxdepth: 1
         :caption: Theory:

         theory/theory

      .. toctree::
         :maxdepth: 1
         :caption: Code Dev:

         code_development/code_dev


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

.. raw::html

   </div>