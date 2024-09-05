<div align="left">
  <h1>Diffusion Model in Riemannian Manifolds</h1>
  <p>
    <a href="#structure">Repository Structure</a>
    ∙
    <a href="#instructions">How to Use</a>
    ∙
    <a href="#documentations">Documentations</a>
  </p>
</div>

<div>
  <h2 id="structure", align="left">Repository Structure</h2>
  <p>
    There are two main directories in this repository: <code>diffusionmodels</code> and <code>projects</code>.
    The former contains the core functionalities to train and evaluate diffusion model in manifolds, including non-Euclidean ones.
    The latter contains worked projects using these functionalities.
  </p>
</div>

<div id="instructions", align="left">
  <h2>How to Use</h2>
  <p>
    To use the project, you need to clone the repository using <a href="https://git-scm.com" target="_blank">Git</a>:
  </p>
  <pre><code class="language-bash"><!--
  -->git clone https://github.com/rezhaTanuharja/manifoldsDiffusion.git<!--
  --></code></pre>
  <p>
    Next, create a symlink inside your system path, pointing to the <code>diffusionmodels</code> directory.
    If you are unsure about your system path, execute the following Python commands:
  </p>
  <pre><code class="language-python"><!--
  -->import sys
  <br><!--
  -->for path in sys.path:
  &nbsp print(path)<!--
  --></code></pre>

  <!-- <p>
    Next, we need to do the following steps (please see detailed instructions):
  </p>

  <ol>
    <li>Download datasets from <a href="https://amass.is.tue.mpg.de" target="_blank">AMASS</a> (<b>A</b>rchive of <b>M</b>otion Capture <b>A</b>s <b>S</b>urface <b>S</b>hapes)</li>
    <li>Preprocess datasets: unpack compressed files and store the data as tensors</li>
    <li>Do some Deep Learnings and save the world!</li>
  </ol>

  <p>More is coming!</p> -->

  <!-- <div>
    <h3>Download datasets from AMASS</h3>
  </div>

  <div>
    <h3>Preprocess Datasets</h3>
  </div>

  <div>
    <h3>Save the World</h3>
  </div> -->

</div>

<div id="documentations", align="left">
  <h2>Documentations</h2>
  <p>Coming soon!</p>
</div>