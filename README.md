# Semantic reasoning of skills in the domain of human resources

This is a repository containing code for the Master Thesis conducted in he University of Koblenz-Landau together with talentsconnect AG.

The research aim is to compare 3 methods of embedding skills into the vector space - distributional (text-based), relational (ontology-based) and hybrid approach using [Attract-Repel model](https://github.com/nmrksic/attract-repel). The paper is available upon request. With this repository you can train the Atrac-Repel and evaluate it on the 2 datasets - intrinsic, that tests the skill embeddings compared to the manual annotaions, and extrinsic ha evaluated the performance of the similar jobs task that takes embeddings as an input. The extrinsic data is provided by talentsconnect AG and can be shared upon request.

## Attract-Repel training

First install the required libraries either with pip:
```
pip install -r requirements.txt
```
or with conda:
```
conda install -r requirements.txt
```

The Attract-Repel is trained on the word2vec vectors and uses linguistic consraints derived from the [ESCO ontology](https://ec.europa.eu/esco/portal), which are contained in the <i>[atttract-repel/word-vectors/init_google_we.txt](atttract-repel/word-vectors/init_google_we.txt)</i> and <i>attract-repel/linguistic_constraints/similar_skills.txt</i> respectively. The file <i>attract-repel/config/experiment_parameters.cfg</i> contains the hyperparameters used in the grid search to find the best combination of attract_margin, batch_size and l2. To only run Attract-Repel on a specified set of hyperparameters, write the same value in the first and second place. 

Run the following command to start the training:
```
python attract-repel/code/attract_repel.py -c config_path -s save_model -e evaluation
```
<b>Arguments</b>:
* config_path : a path to the config file, the default value is <i>attract-repel/config/experiment_parameters.cfg</i>
* save_model : boolean variable, whether to store the model file in the results/grid_search folder, the default value is False
* evaluation : boolean variable, whether to run the evaluation of the models within the training, the default is True. To run the evaluation you need to provide 3 paths in the config file: gold_standard (path to the extrinsic set), companyDataset (path to the list of jobs used in the extrinsic evaluation) and skills_annotated_sample (path to the intrinsic evaluation set).


