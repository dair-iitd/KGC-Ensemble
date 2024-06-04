# Dynamic Ensembling of Textual and Structure-Based Models for Knowledge Graph Completion

This repository contains the code for the ACL 2024 paper: 
"DynaSemble: Dynamic Ensembling of Textual and Structure-Based Models for Knowledge Graph Completion".

The NBFNet folder contains the code and data for training the Neural Bellman-Ford Network ([Link](https://github.com/DeepGraphLearning/NBFNet)) model. It also contains the code for our ensemble baseline. The SimKGC folder contains code and data for training SimKGC ([Link](https://github.com/intfloat/SimKGC)), and the code for dumping (h,r) and t embeddings from the trained model for use in ensembling. RotatE ([Link](https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding)), ComplEx  ([Link](https://github.com/ttrouill/complex)) and HittER ([Link](https://github.com/microsoft/HittER)) can be trained using their official code releases, and their trained checkpoints can be used to generate ensemble results on these models. Please head inside each folder for instructions on running the code. 

