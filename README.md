# Surrogate-Assisted Multi-objective Optimization (SAMOO) Tutorials

Contents
-----------------------------------------------
- Part 1: Pareto Optimality

- Part 2: Pareto Optimality Under Uncertainty

- Part 3: Walkthough of SAMOO 

Environment Setup
-----------------------------------------------

1. Clone this repository:
   ```
   git clone https://github.com/rqmacasieb/SAMOO_tutorials.git
   cd SAMOO_tutorials
   ```

2. Create the conda environment:
   ```
   conda env create -f environment.yml
   ```

3. Activate the environment:
   ```
   conda activate samoo_tutorials
   ```

4. Launch Jupyter:
   ```
   jupyter notebook
   ```

Requirements
-----------------------------------------------
All required packages are specified in the `environment.yml` file. 
A pre-compiled version of pestpp-mou that includes SAMOO functionalities is included in base_files.

Related Publications
-----------------------------------------------
Macasieb, R. Q., White, J. T., Pasetto, D., & Siade, A. J. (2025). A probabilistic approach to surrogate‐assisted multi‐objective optimization of complex groundwater problems (in production). Water Resources Research, 61, e2024WR038554. https://doi.org/10.1029/2024WR038554
