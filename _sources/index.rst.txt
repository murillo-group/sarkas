.. Sarkas documentation master file, created by
   sphinx-quickstart on Mon Jun  1 10:34:03 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


.. the "raw" directive below is used to hide the title in favor of just the logo being visible

.. raw:: html
   <link rel="preconnect" href="https://fonts.gstatic.com">
   <link href="https://fonts.googleapis.com/css2?family=RocknRoll+One&display=swap" rel="stylesheet">  

   <style media="screen" type="text/css">
     h1 { display:none; }
     h2, h3 { font-family: 'RocknRoll One', sans-serif; }
   </style>


SARKAS: Python MD code for plasma physics
=========================================


.. raw:: html

   <div class="jumbotron">
      <h2 style="font-size:100px">SARKAS</h2>
      <h3 style="font-size:50px">Python MD code for plasma physics</h3>
   </div>
   
   
   <div style="position: relative; justify-content: center; align-items: center; display: grid; grid-template-columns:repeat(auto-fit,535px);">
      <img src="_static/logo_green_orange_v5.png" alt="logo" width="35%" style="justify-self:center; align-self:center;">
      <div>
      <!--Carousel Wrapper-->
         <div id="CodeExample" class="carousel slide" data-interval="5000" data-ride="carousel">
          <!--Indicators-->
          <ol class="carousel-indicators">
            <li data-target="#CodeExample" data-slide-to="0" class="active"></li>
            <li data-target="#CodeExample" data-slide-to="1"></li>
          </ol>
          <!--/.Indicators-->
          <!--Slides-->
          <div class="carousel-inner" role="listbox">
            
            <div class="item active">
              <img src="_static/codesnippet.png" width="800px" height="600px" alt="Code Snippet">
            </div>

            <div class="item">
              <img src="_static/BYU.gif" width="100%" alt="BYU">
            </div>
            
          </div>
          
          <!--Controls-->
          <a class="carousel-control left" href="#CodeExample" role="button" data-slide="prev">
          &lsaquo;
          </a>
          <a class="carousel-control right" href="#CodeExample" role="button" data-slide="next">
          &rsaquo;
          </a>
          <!--/.Controls-->
        </div>
      <!--/.Carousel Wrapper-->
      </div>
   </div>

.. raw:: html

   <div class="properties">
      <div class="prprt">
         <h4 class="text-center"><i class="fa fa-smile-o fa-3x mb-3 text-muted"></i><br>User Friendly</h4>
         <p>Run interactively in Jupyter notebook or via script. Set-up a simulation with only 3 lines 
            of code. Run your simulation with 3 more lines. Calculate physics observables with final 3 lines.</p>
      </div>
      <div class="prprt">
         <h4 class="text-center"><i class="fa fa-rocket fa-3x mb-3 text-muted"></i><br>Fast Pure Python</h4>
         <p>Sarkas offers the ease of use of Python while being highly performant with execution speeds comparable 
         to that of compiled languages.</p>
      </div>
      <div class="prprt">
         <h4 class="text-center"><i class="fa fa-flask fa-3x mb-3 text-muted"></i><br>Plasma Potentials</h4>
         <p>Sarkas offers a variety of interaction potentials commonly used in plasma physics. It is the only 
         MD code to support electrons as dynamical particles.</p>
      </div>

      <div class="prprt">
         <h4 class="text-center"><i class="fa fa-database fa-3x mb-3 text-muted"></i><br>Data Science</h4>
         <p>Sarkas has been developed for data science. You can run multiple simulations and store data with a simple for loop.</p>
      </div>
      <div class="prprt">
         <h4 class="text-center"><i class="fa fa-area-chart fa-3x mb-3 text-muted"></i><br>Publications</h4>
         <p>Building upon a set of well-tested primitives and on a solid infrastructure, researchers can get 
         publication-grade results in less time.</p>
      </div>

      <div class="prprt">
         <h4 class="text-center"><i class="fa fa-cogs fa-3x mb-3 text-muted"></i><br>Highly Customizable</h4>
         <p>Sarkas is built in a modular fashion to allow easy implementation of additional features.</p> 
      </div>
      <div class="prprt">
         <h4 class="text-center"><i class="fa fa-users fa-3x mb-3 text-muted"></i><br>Collaborative Effort</h4>
         <p>Sarkas wants to be a common platform for the development of new algorithms to study the most challenging 
         open problems in plasma physics.</p> 
      </div>
      <div class="prprt">
         <h4 class="text-center"><i class="fa fa-github fa-3x mb-3 text-muted"></i><br>Open Source</h4>
         <p>Sarkas is released under the MIT License and maintained by the community on GitHub.</p>
      </div>

   </div>

.. warning:: Sarkas is under development.

.. panels:: 
   :body: bg-light text-left
   :header: bg-light text-center border-0

   :column: col-sm-6
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

      code development/code_dev


   .. toctree::
      :maxdepth: 1
      :caption: Simulations:
      
      examples/examples
   
   ---
   :column: col-sm-6

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
