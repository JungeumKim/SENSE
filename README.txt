This directory provides all the implementations for the results on the CIFAR10 dataset of the proposed SENSE-AT.

--Train
To train sense models: 
Please change the current directory to ./train/subs and then execute train.sh
This will provide both training and the following evaluation on the newly trained models


--Test
To test pretrained models: 
Please change the current directory to ./eval/subs

1) For white-box attacks, please execute the shell file ./eval/subs/sense_model_eval.sh

2) For transfer attacks, 

    i) please download and put the baseline models (WideResNet-34-10) as ./trained_models/BaseIineMethods/XXX/checkpoint.pth, where XXX can be [IAAT, MART, TRADES, MMA12]. One may download the models or codes for training from: 
    IAAT: https://github.com/yogeshbalaji/Instance_Adaptive_Adversarial_Training [1]
    MART: https://github.com/YisenWang/MART [2]
    TRADES: https://github.com/yaodongyu/TRADES [3]
    MMA12: https://github.com/BorealisAI/mma_training [4]
    
    ii) please execute the shell file ./eval/subs/transfer.sh
    
--appendix
For the experiment conducted in appendix,
 1) For CNNs on cifiar, please change the current directory to ./appendix/subs and then excecute train_eval_appendix.sh
    
References: 
[1] Balaji, Yogesh, Tom Goldstein, and Judy Hoffman. "Instance adaptive adversarial training: Improved accuracy tradeoffs in neural nets." arXiv preprint arXiv:1910.08051 (2019).
[2] Wang, Yisen, et al. "Improving adversarial robustness requires revisiting misclassified examples." International Conference on Learning Representations (2020).
[3] Zhang, Hongyang, et al. "Theoretically principled trade-off between robustness and accuracy." Proceedings of the 36th International Conference on Machine Learning, in PMLR 97:7472-7482 (2019)
[4] Ding, Gavin Weiguang, et al. "MMA Training: Direct Input Space Margin Maximization through Adversarial Training." International Conference on Learning Representations (2019).
