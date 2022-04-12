# VR-FW
We develop one distributed stochastic projection-free algorithm for convex and non-convex finite-sum optimization. 
The proposed algorithm is binary_classify_GTSPIDER.py.  
Comparative algorithms include centralized spider projection-free algorithm (cen-spiderfw.py) and a distributed deterministic projection-free algorithm (binary_classify_GTmpi.py).
Two distributed are applied through the MPI communication interface. 
We apply all algorithms to solve a traditional binary classification problem, whose objective function can be convex or non-convex.
min._{x\in \Omega} F(x), F(x)=1/m \sum_{i=1}^m f_i(x)
convex: f_i(x)=1/n_i \sum_{j=1}^{n_i} \ln (1+exp(-l_{i,j}<a_{i,j},x>))
non-convex: f_i(x)=1/n_i\sum_{j=1}^{n_i} 1/(1+exp(l_{i,j}<a_{i,j},x>))
