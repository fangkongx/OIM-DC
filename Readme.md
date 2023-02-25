

### File Structure

 ``` python
. 
| BanditAlg # OIM Algorithms
| DataProcess
    ├── raw # raw data
    ├── getER_Graph.py # Generate ER Graph
    ├── SampleSubGraph.py # Generate Subgraph of Flickr/NetHEPT 
| datasets 
| Model
    ├── IC.py, DC.py # run on IC/DC 
| Oracle
    ├── CMAB.py, Greedy.py, Greedy_IC.py # 
| SampleFeature # Sample Features of dataset
| Tool
 ```



# Data Generation

##### Synthetic ER Graph Generation(DataProcess/)

```
getER_Graph.py 
```

Parameters

> p: connection probabilities between edges in the ER graph
>
> n: node numbers

##### Real Data Generation(DataProcess/)

```
SampleSubGraph.py   
```

Parameters

> key_node_num：key node numbers
>
> save_dir：

##### Range probability feature generation(SampleFeature/)

```
Probability.py 
```

Parameters

> graph_name_list : graphs needed to be processed
>
> dataset：
>
> prob_list：activation probability



```
NodeFeature.py  # Node Feature Generation
```

> save_dir：
>
> graph_name_list：graphs' names

```
FeatureVector.py  
```

> dataset
>
> graph_name_list

##### homogenerous features generation(SampleFeature/)

```
Probability_fixed.py
```

> graph_name_list：graphs needed to be processed
>
> dataset：
>
> prob_list：activation probability

homogenerous data also needs to use `NodeFeature.py` and `FeatureVector.py` to generate node and edge features.







