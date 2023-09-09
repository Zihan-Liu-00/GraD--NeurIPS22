# GraD
Implementation for **NeurIPS 2022** paper 'Towards Reasonable Budget Allocation in Untargeted Graph Structure Attacks via Gradient Debias'

To reproduce the attack performance:
1. Set the attack scenario (dataset, pert. rate) from ```train_GraD.py```
2. Run ```train_GraD.py``` to generate the attacked graph
3. Test the attack performance by running ```test.py```
4. Find the results files in the repository 'results'

**Tips:**  
The value of hyperparameter 'momentum' in <class GraD> in ```attack.py``` is relatively important. The momentum needs to be at an appropriate value that allows the surrogate model to have high accuracy as well as confidence in the prediction. The adjustment of the momtemtum parameter only needs to refer to the performance of the surrogate model during the attack, without using the label of the test nodes.

To cite this paper in latex, please copy and paste:  
to be published...
