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
</div>

```bash
git clone https://github.com/rezhaTanuharja/manifoldsDiffusion.git
```

<div>
  <p>
    Next, check your system path, this is the path that Python search for when you import a module.
    Execute the following:
  </p>
</div>

```python
import sys

for path in sys.path:
  print(path)
```

<div>
  <p>
    If the <code>diffusionmodels</code> directory is not located in any of these path, you won't be able to import it as a module.
    Therefore, add a symlink in one of the path, for example the one with <code>.../site-packages/</code>, pointing to the <code>diffusionmodels</code> directory by executing
  </p>
</div>

```bash
ln -s <absolute_path_to_diffusionmodels> .../site-packages/diffusionmodels
```

<div>
  <p>
    Replace <code>&lt;absolute_path_to_diffusionmodels&gt;</code> and complete the <code>...</code> using the actual paths in your machine.
Now, you should be able to import the module, e.g.,
  </p>
</div>

```python
import diffusionmodels as dm


differential_equations = dm.ExplodingVariance()

### the rest of the code
### ...
### ...
```

<div id="documentations", align="left">
  <h2>Documentations</h2>
  <p>Coming soon!</p>
</div>
