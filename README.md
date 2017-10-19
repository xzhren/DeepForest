gcForest v2.0
========
This is improvement version of the official clone for the implementation of gcForest.

Package Official Website: http://lamda.nju.edu.cn/code_gcForest.ashx                      

Reference: [1] Z.-H. Zhou and J. Feng. Deep Forest: Towards an Alternative to Deep Neural Networks.               
            In IJCAI-2017.  (https://arxiv.org/abs/1702.08835v2 )                                                 

Requirements: This package is developed with Python 3.x, please make sure all the dependencies are installed,  which is specified in requirements.txt                                                                            


ToDo List For the Version 2
========
- Train Driver DataSet：
  - python tools/train_cascade.py --model models/driver/gcforest/ca-tree500-n4x2-3folds.json --log_dir logs/gcforest/driver/ca-tree500-n4x2-3folds/
- ​1. Change Python 2.7 to Python 3.x
  - basestring to str
  - / to //
- 2. Add metrics
- 3. Add Best Layer ID Select

Package Overview
========
* lib/gcforest
    - code for the implementations for gcforest
* tools/train_fg.py
    - the demo script used for training Fine grained Layers
* tools/train_cascade.py
    - the demo script used for training Cascade Layers
* models/
    - folder to save models which can be used in tools/train_fg.py and tools/train_cascade.py
    - the gcForest structure is saved in json format
* logs
    - folder logs/gcforest is used to save the logfiles produced by demo scripts


Happy Hacking.