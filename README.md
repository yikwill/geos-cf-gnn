<center>
  <img src="images/geos_cf_surface_no2.png">
</center>

# Table of Contents
I. [Introduction](#introduction)

II. [Related Works](#related-works)

III. [Methods](#methods)

IV. [Discussion](#discussion)

V. [Ethical Sweep](#ethical-sweep)

VI. [References](#references)

VII. [Appendix](#appendix)

# Introduction
The chemical composition of the atmosphere has tangible impacts for billions of people around the globe. It is tightly coupled with both surface air pollution levels, which are one of the leading environmental causes of death worldwide, and global climate change via mechanisms such as radiation scattering and aerosol-cloud interactions (GBD 2013 Risk Factors Collaborators et al. 2015; National Academies of Sciences, Engineering, and Medicine et al. 2016). As such, providing high resolution, accurate forecasts of global atmospheric composition is incredibly important for human health, infrastructure, and climate change solutions. The NASA Goddard Earth Observing System (GEOS) composition forecast modeling system, GEOS-CF, is the current state-of-the-art atmospheric composition forecasting system and runs near real-time simulations to provide high quality predictions. GEOS-CF is one of many recently developed Earth system models which predict various geophysical variables (e.g, chemical distributions, humidity, wind speed, etc.) using physical computer models that solve many governing equations on discrete physical grids. While such models have found success, it typically comes at the cost of speed and large computing requirements (Bauer et al. 2015). These tradeoffs have driven recent interest in developing machine learning (ML) models to both improve and speed up Earth system forecasts (Rasp et al. 2020; Watson-Parris et al. 2022). These machine learning models are purely data-driven and seek to emulate the Earth system dynamics without the use of specific governing laws and equations.

In this work, we introduce a graph neural network (GNN) emulator of the NASA GEOS-CF system for forecasting global atmospheric composition. Specifically, our model uses an "encode-process-decode" architecture to transform the original latitude-longitude data to a lower dimensional latent space, perform computations in this space, then re-project back to the original latitude-longitude format for final predictions (Battaglia et al. 2018). The GNN learns on publicly available GEOS-CF assimilated data to emulate its forecasts. Our work is most similar to that of Keisler and Lam et al. in that we use a GNN for making forecasts of Earth system variables. However, our approach differs in two distinct ways. First, we forecast global atmospheric composition of the four chemical speices, ozone (O<sub>3</sub>), nitrogen dioxide (NO<sub>2</sub>), carbon monoxide (CO), sulfur dioxide (SO<sub>2</sub>), as well as particulate matter (PM2.5), though Keisler's work made predictions of weather variables like humidity and temperature. To the authors' best knowledge, this is first work to emulate the NASA GEOS-CF system for making composition forecasts. Second, we make use of graph attention layers in our processor as opposed to the graph convolutional layers of Keisler and Lam et al.

Overall, we find that our GNN model provides similar medium-range (1 to 2 days) forecasts as the NASA GEOS-CF system while being faster and less computationally expensive. In future work, we hope to investigate the quality of forecasts in different parts of the world to identify any potential inequities.

# Related Works
### Atmospheric Composition
The WMO (World Meteorological Organization) tasks itself with assisting in the gathering and sharing of worldwide observational atmospheric composition data, which is produced by many agencies at both national and subnational levels, and in academia and the private sector. Klausen et al. discuss the importance of employing means to gather and share high quality comparable data at a global level, in the context of the WMO’s Unified Data Policy, and addressing issues such as lack of observational infrastructure in some parts of the world, inconsistent data quality, and the lack of sharing of available data with the international community. The GAW has devised guidelines for measuring and quality control when collecting observational data in order to ensure that data from different sources around the world can be comparable, and that meaningful conclusions can be drawn. Using optimal methods for collecting and monitoring this data is imperative for supporting policy for climate crisis solutions, predicting patterns in extreme weather, informing the process of recovery of the ozone layer, and the identification of levels of air pollutants, as well as concentrations of greenhouse gases, since they pose a threat to human health, ecosystems, and agriculture. As many countries have pledged to achieve net-zero greenhouse gas emissions, a failure to adequately manage this atmospheric observational data would preclude the ability to quantify both progress and setbacks toward current and future climate goals.

There have been recent works which make use of atmospheric observational data to improve chemical simulators. Notably, Geiss et al. use GEOS-CF data to train a machine learning model to downscale atmopsheric chemistry simulations with substantially higher accuracy than previous attempts.

### Graph Neural Networks
As explained by Sanchez-Lengeling et al., neural networks have been combined with graphs to leverage spatial properties of graphs. Graph neural networks (GNNs) are composed of vertex, edges, and global nodes where vertexes can be specified as directed or undirected. GNNs have been lervaged to cover a wide variety of projects including semantic language encodings, predicting bonds in molecules, and social networks. GNNs have also been encoded to widely model connections and figure out how information, ideas, or items interact with one another. 

Zhou et al. elaborate and notes that GNNs can capture the dependence and high amount of relational data of graphs via message passing between the nodes of graphs. GNNs and their variants, including propogation modules, sampling modules, and pooling modules, have had ground breaking performance in recent years. Zhou et al. also outline the design process for a GNN to be (1) Find the graph structure, as in what the nodes and edges will be, (2) specify graph type and scale, i.e. whether the graph is homogenous or heterogenous, directed or undirected, simply or dynamic, etc. (3) design the loss function, and finally (4) build the module using computational modules. 

Pfaff et al. build on more basic GNN models to build mesh-based models, which are highly adaptive, efficient architectures for neural networks with the ability to quickly pass information between nodes in a graph neural network. These models can make highly accurate and efficient predictions about the dynamics of physical systems. Pfaff et al. find that mesh graphs are generalizable and can be accurately applied to numerous different systems such as the weather and climate, often with much lower error rates than mesh-free models.

Keisler makes use of a mesh GNN model to learn from multi-resolution weather data make high resolution forecasts of various weather variables such as wind speed, humidity, and temperature. The architecture operates in three discrete steps: an encoder transforms some region of the world that we want to make predictions about into input vectors, a processor analyzes said input vectors, and a decoder maps the resulting output data back onto the physical map. Analysis of the model’s accuracy revealed that it performs either better or at parity with cutting-edge physical (non-ML) weather forecasters, motivating the usage of neural networks in the atmospheric predictions space.

Lam et al. build on the GNN of Keisler to create a highly accurate model for 10-day weather forecasts. Like Keisler's architecture, they use an encode-process-decode structure. The encoder maps from the physical latitute/longitude space to a latent graph. The processor does computation on the latent graph, which has less nodes than the original latitute/longitude. Lastly, the decoder maps from the latent graph back to the latitute/longitude space to create a real forecast. The primary difference between this model and that of Keisler for weather forecasting is that the authors of this work use a multi-scale mesh icosahedron for the latent graph while Keisler uses a static resolution icosahedron. That is, they use multiple icosahedrons of different resolutions to create their latent graph which the processor does computation on. This allows them to effectively capture spatial relations in the data. The low resolution icosahedrons can capture long distance connections and the high resolution icosahedrons can capture local connections.

# Methods

We will train our model on NASA's GEOS Composition Forcasting (GEOS-CF) dataset. This data contains the atmospheric concentration of various compounds across the entire planet on hour intervals from January 2018 through today; one input vector will represent the concentrations for a single hour at a single global coordinate (with additional vectors for that time step representing concentrations spaced 0.25 degrees apart in both the latitude and longitude directions).

Due to the sheer size of the dataset, we plan to initially train on a single compound for a one month period. Depending on time and our preliminary training results, we will expend and train on more time steps.

Before passing our data into our neural network, we will normalize it, converting our data from a lognormal distribution to a normal distribution through the following equation (for arbitrary data point, x): x_norm = 1 / SD * (log(x + epsilon) - mean), where SD is the overall standard deviation of that hour's data, mean is the overall mean of that hour's data, and epsilon is 10^{-32} to prevent us from ever taking log(0).

To load in training data, we will convert the normalized GEOS-CF data into a single csv file (where one column represents one hour for our chosen compound).

This data is then passed into a graph neural network (GNN). The GNN is composed of three discrete layers: an encoder (which transforms the csv file into an icosahedron mesh graph), a processor (which will train the data: broadly, a GAT will be built and used to train on the mesh's edge attributes - which contain data on the connections between different nodes - while a GCN will be built and used to train and update each node's features), and a decoder (which will function as a reverse encoder and transform our mesh graph back into parseable, csv data).

All of the above models are being built using PyTorch, PyTorch Geometric, and H3. Our network is largely modeled off of Ryan Keisler's "Forecasting Global Weather With Graph Neural Networks."

# Discussion
We implemented two types of Graph Neural Networks: Graph Attention Networks and Graph Convolution Networks. GATs train to determine the proper update of weights for each edge to update the data at the center node. GCNs perform the node level updates by doing a convolution sum of all incoming edges with pre-determined weights.

We implemented Keisler's method by metalayering a MLP and GCN together in order to see how the model performs on our data. We plan to implement a similar metalayer with a GAT for the node updates and a GCN for the edge updates, in order to see if the ability to learn weights will increase performance.

We expect to see lower loss and higher accuracy after implementing the GAT-GCN layered model. Additionally, we expect Keisler's method to extend well to our data for hour-long predictions. We also expect weight lernings to be related to geographical constraints and variables such as mountains, cities, and similar factors.

For our initial results, we train a model on a month's worth of data for one chemical, NO<sub>2</sub>, and then predict that chemical's concentration at each latitude-longitude location 12 hours later. Since the data is of reasonably high resolution (721x1440) and the model is somewhat large, training for one epoch on a Google Colab GPU takes around 8-9 minutes. As such, we begin by only training for 5 epochs. In the near future, we will extend the training data to be over several months and/or lessen the lead time for our predictions in order to make more granular forecasts in terms of time.

In our initial training loop, we use the Adam optimizer with a learning rate of 1e-5. For the loss function, we use standard MSE loss but scale both the prediction and truth tensors by the variance of chemical we are predicting as in the original Keisler work. As mentioned previously, we train for 5 epochs and split our data such that the first three weeks of January 2018 are the training set and the last week of that month is the validation set. We haven't yet evaluated the model since we have only trained for 5 epochs as mentioned above, so we don't have test data (yet). Lastly, we use a batch size of just 1 for now, since the training process on Google Colab seems to take up about 10 GB of GPU memory. After 5 epochs, the training loss decreased from 9.9530 to 9.7367 and validation loss decreased from 10.0138 to 9.7756. A screenshot of the training loop reporting is shown below.

<center>
  <img src="images/initial_training_results.png">
</center>

# Ethical Sweep
**General Considerations:** At a high level, this work may help provide accurate forecast models which can help promote global health and awareness for changes in climate. This work can help these causes and has close to no negative use cases. Current approaches use fully-integrated physical chemistry models and simulations in order to forecast composition. Due to the complexity in forecasting, a limited GNN may not provide accurate results for forecasting and may require additional data. Our team consists of a mix of computer science, math, and environmental analysis majors with semi-similar backgrounds, but a few outliers. It is not as diverse as we would hope for in terms of academic background, in part because the topic is not easily approachable. However, it seems we have different experiences and identities in terms of socioeconomic background, ethnicity, and gender. To handle mistakes, we will discuss them during project meetings and go over miscommunications in person for dividing tasks. Additionally, we may check over each other's work to preemptively catch errors.

**Data Curation and Use:** We believe that our data is valid for its intended use. It is simply a collection of global GEOS-CF predictions coupled with real satellite observations. Of any dataset, this one is most well suited for our goal of predicting future atmospheric composition. The most obvious bias that the data could contain is spatial bias. It could be the case that NASA's own predictions or satellite observations are better over certain parts of the globe than others. This may be due to model design choices or sparse observations in specific areas of the globe. We must be aware that this may lead our own forecasting model to make better predictions for certain countries or regions. To combat this, we may look for other data sources to ensure that we have quality data across the entire globe. Furthermore, we may look into ways to enforce spatial fairness constraints into our GNN model. One way we could audit our data and code is to review NASA's own data curation process and make an assessment on its validity and fairness. Additionally, once we have a trained GNN model, we could look at error rates in different regions of the Earth to ensure that our model has similar prediction accuracy across the globe.

**Impact Assessment:** There is a chance we see different error rates for different sub-groups. In this project, sub-groups will be split geographically (e.g. we may have a set of data from western North America, another set from east Asia, etc.); if NASA's GEOS-CF database has not equally sampled from all across the world, then it is possible that some sub-groups with less overall data will have larger error rates due to undertraining on our model. With that said, GEOS-CF holds a fairly mature, developed dataset that has been built by diverse groups of researchers from around the world, so it seems likely that any severe discrepancies in sub-group sampling size have been dealt with by now. One potential path for data misinterpretation could occur when we train our graph neural network. GNN's function by exchanging information with their neighbors; therefore, if individual nodes in the GNN are trained on poor datasets, there is a risk that they decrease the accuracy of neighboring nodes with their own incorrect conclusions and incomplete datasets. A mistake like this could have cascading effects on our entire GNN and severely limit our ability to meaningfully analyze global atmospheric conditons. Preventing this issue goes back to making sure we train our neural network on a good dataset (and, relatedly, taking care to measure the performance of individual nodes and layers, rather than just looking at the GNN as a whole). Fortunately, our GNN deals exclusively with high-level, impersonal atmospheric data, leaving little room for infringing on others' privacy. The only potential privacy risk could come from inadvertently revealing the name of an individual who contributed data to the GEOS-CF dataset (if such data is even recorded by NASA's database).

# References
Battaglia, P. W., and Coauthors, 2018: Relational inductive biases, deep learning, and graph networks. arXiv preprint arXiv:1806.01261.

Bauer, P., A. Thorpe, and G. Brunet, 2015: The quiet revolution of numerical weather prediction. Nature, 525 (7567), 47–55.

GBD 2013 Risk Factors Collaborators, and Coauthors, 2015: Global, regional, and national comparative risk assessment of 79 behavioural, environmental and occupational, and metabolic risks or clusters of risks in 188 countries, 1990–2013: a systematic analysis for the global burden of disease study 2013. Lancet (London, England), 386 (10010), 2287.

Geiss, A., S. J. Silva, and J. C. Hardin, 2022: Downscaling atmospheric chemistry simulations with physically consistent deep learning. Geoscientific Model Development, 15 (17), 6677–6694.

Keisler, R., 2022: Forecasting global weather with graph neural networks. arXiv preprint arXiv:2202.07575.

Klausen, J., C. Volosciuk, O. Tarasova, and S. Netcheva, 2021: Benefits of atmospheric composition monitoring and international data exchange. Bolet ́ın-Organizaci ́on Meteorol ́ogica Mundial, 70 (2), 41–46.

Lam, R., and Coauthors, 2022: Graphcast: Learning skillful medium-range global weather forecasting. arXiv preprint arXiv:2212.12794.

National Academies of Sciences, Engineering, and Medicine, and Coauthors, 2016: The future of atmospheric chemistry research: remembering yesterday, understanding today, anticipating tomorrow. National Academies Press.

Pfaff, T., M. Fortunato, A. Sanchez-Gonzalez, and P. W. Battaglia, 2020: Learning mesh-based simulation with graph networks. arXiv preprint arXiv:2010.03409.

Rasp, S., P. D. Dueben, S. Scher, J. A. Weyn, S. Mouatadid, and N. Thuerey, 2020: Weatherbench: a benchmark data set for data-driven weather forecasting. Journal of Advances in Modeling Earth Systems, 12 (11), e2020MS002 203.

Sanchez-Lengeling, B., E. Reif, A. Pearce, and A. B. Wiltschko, 2021: A gentle introduction to graph neural networks. Distill, 6 (9), e33. Watson-Parris, D., and Coauthors, 2022: Climatebench v1. 0: A benchmark for data-driven climate projections. Journal of Advances in Modeling Earth Systems, 14 (10), e2021MS002 954.

Zhou, J., and Coauthors, 2020: Graph neural networks: A review of methods and applications. AI open, 1, 57–81.

# Appendix
## Old Work
All text in this section is saved previous work. It may be ignored.
### Related Works

**(Keisler, R.) Forecasting Global Weather With Graph Neural Networks**. Graph neural networks are uniquely suited to modelling complex weather systems due to their ability to learn multi-resolution models (that is, output models with different degrees of forecast specificity depending on whether the model is being used to predict weather in a local town or large country) and more accurately modelling shifts in weather over user-defined time steps. The architecture built by the authors of this paper operates in three discrete steps: an encoder transforms some region of the world that we want to make predictions about into input vectors, a processor analyzes said input vectors, and a decoder maps the resulting output data back onto the physical map. Analysis of the model's accuracy revealed that it performs either better or at parity with cutting-edge physical (non-ML) weather forecasters, motivating the usage of neural networks in the atmospheric predictions space. Link: https://arxiv.org/pdf/2202.07575.pdf

**(Klausen et al.) Benefits of Atmospheric Composition Monitoring and International Data Exchange.** The WMO (World Meteorological Organization) tasks itself with assisting in the gathering and sharing of worldwide observational atmospheric composition data, which is produced by many agencies at both national and subnational levels, and in academia and the private sector. This article discusses the importance of employing means to gather and share high quality comparable data at a global level, in the context of the WMO’s Unified Data Policy, and addressing issues such as lack of observational infrastructure in some parts of the world, inconsistent data quality, and the lack of sharing of available data with the international community. The GAW has devised guidelines for measuring and quality control when collecting observational data in order to ensure that data from different sources around the world can be comparable, and that meaningful conclusions can be drawn. Using optimal methods for collecting and monitoring this data is imperative for supporting policy for climate crisis solutions, predicting patterns in extreme weather, informing the process of recovery of the ozone layer, and the identification of levels of air pollutants, as well as concentrations of greenhouse gases, since they pose a threat to human health, ecosystems, and agriculture. As many countries have pledged to achieve net-zero greenhouse gas emissions, a failure to adequately manage this atmospheric observational data would preclude the ability to quantify both progress and setbacks toward current and future climate goals. Link: https://public.wmo.int/en/resources/bulletin/benefits-of-atmospheric-composition-monitoring-and-international-data-exchange

**(Lam et al.) GraphCast: Learning Skillful Medium-Range Global Weather Forecasting**. In this work, the authors bulid on previous graph neural net (GNN) architectures to create a highly accurate model for 10-day weather forecasts. Like previous GNN architectures, they use an encoder-processor-decoder structure. The encoder maps from the physical latitute/longitude space to a latent graph. The processor does computation on the latent graph, which has less nodes than the original latitute/longitude. Lastly, the decoder maps from the latent graph back to the latitute/longitude space to create a real forecast. The primary difference between this model and other GNN architectures for weather forecasting is that the authors of this work use a multi-scale mesh icosahedron for the latent graph. That is, they use multiple icosahedrons of different resolutions to create their latent graph which the processor does computation on. This allows them to effectively capture spatial relations in the data. The low resolution icosahedrons can capture long distance connections and the high resolution icosahedrons can capture local connections. Link: https://arxiv.org/pdf/2212.12794.pdf

**(Pfaff et al.) Learning Mesh-Based Simulation With Graph Networks**. Mesh-based models are highly adaptive, efficient architectures for neural networks with the ability to quickly pass information between nodes in a graph neural network, resulting in a system that can perform highly accurate and efficient predictions about the dynamics of physical systems. This paper's authors found that mesh graphs are generalizable and can be accurately applied to numerous different systems (weather and particle dynamics among them), often with much lower error rates than mesh-free models. Link: https://arxiv.org/pdf/2010.03409v4.pdf

**(Sanchez-Lengeling et al.) A Gentle Introduction to Neural Networks**. Neural networks have been combined with graphs to leverage spatial properties of graphs. Graph Neural networks are composed of vertex, edges, and global nodes where vertexes can be specified as directed or undirected. GNNs have been lervaged to cover a wide variety of projects including semantic language encodings, predicting bonds in molecules, and social networks. GNNs have also been encoded to widely model connections and figure out how information, ideas, or items interact with one another. This resource provides a few interactive exmaples along with more advanced introductions to GANs and Generative Modeling. Link: https://distill.pub/2021/gnn-intro/

**(Zhou et al.) Graph neural networks: A review of methods and applications** Graph neural networks are models that capture the dependence and high amount of relational data of graphs via message passing between the nodes of graphs. GNNs and their variants have had ground breaking performance in recent years. Models for GNNs are built using a variety of modules, including propagation modules, sampling modules, and pooling modules. The paper provides a variety of instantiations of these computational modules, more details for which could be found in the futher papers linked out. The paper outlines the design process for a GNN to be (1) Find the graph structure, as in what the nodes and edges will be, (2) specify graph type and scale, i.e. whether the graph is homogenous or heterogenous, directed or undirected, simply or dynamic, etc. (3) design the loss function, and finally (4) build the module using computational modules. The paper also explores analyses and applications of GNNs. Link: https://doi-org.ccl.idm.oclc.org/10.1016/j.aiopen.2021.01.001

### Introduction Outline

**Team Members:** Alex Fay, Elly Rokeach, Francine Wright, Ryan O'Hara, William Yik

**Problem Introduction:** Highly concentrated air pollutants are widespread across the globe and have been linked to negative health outcomes for a variety of different populations.

**GEOS-CF Overview:** The Goddard Earth Observing System composition forecast (GEOS-CF) is NASA's state-of-the-art modeling system for global atmospheric composition.

**Graph Neural Network Introduction:** Graph neural networks, which utilize the structures and properties of graphs to successfully model complex systems, have been highly successful in emulating global atmospheric and weather conditions.

**Methods Overview:** Broadly, we developed and trained a GNN on the GEOS-CF dataset to predict future distributions of air pollutants such as ozone.

**Details of Data Collection:** The main technical challenge we faced during this project was in developing an accurate, well-trained GNN.

**Conclusions:** Ideally, our model will be able to draw on old atmospheric composition data to make accurate predictions about air quality in various regions around the world.

**Future Directions:** In order to make our model more useful to the atmospheric science community, we hope to expand the number of parameters that our model is trained on.

### Project Description 
The Goddard Earth Observing System composition forecast (GEOS-CF) is NASA's state-of-the-art modeling system for global atmospheric composition [1]. It provides high resolution global forecasts of several chemical species including ozone (O<sub>3</sub>), nitrogen dioxide (NO<sub>2</sub>), and carbon monoxide (CO). Several of these chemicals are air pollutants and/or aersols which have tangible impacts for people around the world. Thus, providing accurate forecasts of their global distribution is key for global health and infrastructure. During the past few years, the development of large global weather, climate, and atmospheric composition models like GEOS-CF which are based on physical laws and differential equations has motivated the creation of machine learning systems which emulate these models [2,3]. The goal of such emulators is to achieve the prediction accuracies of physical models at a fraction of the cost. 

One type of machine learning model that has gained traction in recent years for forecasting tasks is the graph neural network (GNN) [4,5]. These neural networks leverage the structure and properties of graphs and have been successfully applied to emulate global weather features such as temperature, humidity, and wind [6]. This recent success motivates my interest in applying GNNs to emulate the GEOS-CF forecasts. NASA has maintained an assimilated database atmopsheric composition of the Earth since 2018, which is a combination of model predictions and satellite observations. This data could be used to train a GNN, the predictions of which could be compared to the actual GEOS-CF model. If successful, this project could provide a computationally cheaper way to achieve similar results to the GEOS-CF model.

While I think the bulk of the work for this project will be in understanding, designing, and training a reasonable GNN emulator, a potential longer term goal of this project would be to investigate methods for adding physical constraints into the GNN, as neural networks do not necessarily follow laws of physics such as mass conservation out of the box. Ensuring that neural networks are properly physically constrained is a rich research area in data-driven climate science communities [7]. Additionally, should the project succeed, I have a personal goal of writing an academic paper and submitting it for publication to a conference or journal.

[Sam Silva](https://www.samjsilva.com/), Professor of Earth Sciences and Civil and Environmental Engineering at USC, has graciously offerred to provide guidance for this project. He is an expert in computational atmospheric chemistry, and he has ongoing projects applying graph networks to atmopsheric science. His level of involvment will likely depend on how much progress we make on our own and how often we get stuck.

### Project Goals
1. Explore and make use of the NASA's GEOS-CF database of atmospheric composition. Make informed decisions on manipulating the data (e.g., upscaling).
2. Understand and design a reasonable GNN model.
4. Train this model on the GEOS-CF dataset and make any predictions at all (even poor ones).
5. Make informed edits to the model to improve forecasting accuracy.
6. Write an academic paper documenting our model/results and submit it for publication.

### Works Cited
[1] Keller, C. A., Knowland, K. E., Duncan, B. N., Liu, J., Anderson, D. C., Das, S., ... & Pawson, S. (2021). Description of the NASA GEOS composition forecast modeling system GEOS‐CF v1. 0. Journal of Advances in Modeling Earth Systems, 13(4), e2020MS002413.

[2] Rasp, S., Dueben, P. D., Scher, S., Weyn, J. A., Mouatadid, S., & Thuerey, N. (2020). WeatherBench: a benchmark data set for data‐driven weather forecasting. Journal of Advances in Modeling Earth Systems, 12(11), e2020MS002203.

[3] Watson‐Parris, D., Rao, Y., Olivié, D., Seland, Ø., Nowack, P., Camps‐Valls, G., ... & Roesch, C. (2022). ClimateBench v1. 0: A Benchmark for Data‐Driven Climate Projections. Journal of Advances in Modeling Earth Systems, 14(10), e2021MS002954.

[4] Zhou, J., Cui, G., Hu, S., Zhang, Z., Yang, C., Liu, Z., ... & Sun, M. (2020). Graph neural networks: A review of methods and applications. AI Open, 1, 57-81.

[5] Cao, D., Wang, Y., Duan, J., Zhang, C., Zhu, X., Huang, C., ... & Zhang, Q. (2020). Spectral temporal graph neural network for multivariate time-series forecasting. Advances in neural information processing systems, 33, 17766-17778.

[6] Keisler, R. (2022). Forecasting Global Weather with Graph Neural Networks. arXiv preprint arXiv:2202.07575.

[7] Beucler, T., Pritchard, M., Rasp, S., Ott, J., Baldi, P., & Gentine, P. (2021). Enforcing analytic constraints in neural networks emulating physical systems. Physical Review Letters, 126(9), 098302.
