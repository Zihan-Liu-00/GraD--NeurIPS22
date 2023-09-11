# GraD
Implementation for **NeurIPS 2022 paper 'Towards Reasonable Budget Allocation in Untargeted Graph Structure Attacks via Gradient Debias'**

In this work, we find a little but critical 'bug' in Metattack (Zugner et al.) and receive a big improvement in attack performance. All you need to think about is that a loss for a node sample in a node-level task doesn't only affect the node itself, but also affect other node because of the edges. A bad prediction will produce a higher gradient by cross-entropy loss from backpropagation, however, is such a big gradient what an attacker really want?

To reproduce the attack performance:
1. Set the attack scenario (dataset, pert. rate) from ```train_GraD.py```
2. Run ```train_GraD.py``` to generate the attacked graph
3. Test the attack performance by running ```test.py```
4. Find the results files in the repository 'results'

**Tips:**  
The value of hyperparameter 'momentum' in <class GraD> in ```attack.py``` is relatively important. The momentum needs to be at an appropriate value that allows the surrogate model to have high accuracy as well as confidence in the prediction. The adjustment of the momtemtum parameter only needs to refer to the performance of the surrogate model during the attack, without using the label of the test nodes.

Please find our paper at:
https://papers.nips.cc/paper_files/paper/2022/hash/b31aec087b4c9be97d7148dfdf6e062d-Abstract-Conference.html

To cite this paper in latex, please use:  
@inproceedings{liu2022towards,
  title={Towards Reasonable Budget Allocation in Untargeted Graph Structure Attacks via Gradient Debias},
  author={Liu, Zihan and Luo, Yun and Wu, Lirong and Liu, Zicheng and Li, Stan Z},
  booktitle={Advances in Neural Information Processing Systems},
  year={2022}
}

For any question, please leave an issue or send email to zihanliu@hotmail.com or liuzihan@westlake.edu.cn
