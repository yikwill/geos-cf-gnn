# Forecasting Global Atmospheric Composition with Graph Neural Networks

<center>
  <img src="images/geos_cf_surface_no2.png">
</center>

## Related Works

**Forecasting Global Weather With Graph Neural Networks**. Graph neural networks are uniquely suited to modelling complex weather systems due to their ability to learn multi-resolution models (that is, output models with different degrees of forecast specificity depending on whether the model is being used to predict weather in a local town or large country) and more accurately modelling shifts in weather over user-defined time steps. The architecture built by the authors of this paper operates in three discrete steps: an encoder transforms some region of the world that we want to make predictions about into input vectors, a processor analyzes said input vectors, and a decoder maps the resulting output data back onto the physical map. Analysis of the model's accuracy revealed that it performs either better or at parity with cutting-edge physical (non-ML) weather forecasters, motivating the usage of neural networks in the atmospheric predictions space. Link: https://arxiv.org/pdf/2202.07575.pdf

## Introduction Outline

**Team Members:** Alex Fay, Elly Rokeach, Francine Wright, Ryan O'Hara, William Yik

**Problem Introduction:** Highly concentrated air pollutants are widespread across the globe and have been linked to negative health outcomes for a variety of different populations.

**GEOS-CF Overview:** The Goddard Earth Observing System composition forecast (GEOS-CF) is NASA's state-of-the-art modeling system for global atmospheric composition.

**Graph Neural Network Introduction:** Graph neural networks, which utilize the structures and properties of graphs to successfully model complex systems, have been highly successful in emulating global atmospheric and weather conditions.

**Methods Overview:** Broadly, we developed and trained a GNN on the GEOS-CF dataset to predict future distributions of air pollutants such as ozone.

**Details of Data Collection:** The main technical challenge we faced during this project was in developing an accurate, well-trained GNN.

**Conclusions:** Ideally, our model will be able to draw on old atmospheric composition data to make accurate predictions about air quality in various regions around the world.

**Future Directions:** In order to make our model more useful to the atmospheric science community, we hope to expand the number of parameters that our model is trained on.

## Ethical Sweep
**General Considerations:** At a high level, this work may help provide accurate forecast models which can help promote global health and awareness for changes in climate. This work can help these causes and has close to no negative use cases. Current approaches use fully-integrated physical chemistry models and simulations in order to forecast composition. Due to the complexity in forecasting, a limited GNN may not provide accurate results for forecasting and may require additional data. Our team consists of a mix of computer science, math, and environmental analysis majors with semi-similar backgrounds, but a few outliers. It is not as diverse as we would hope for in terms of academic background, in part because the topic is not easily approachable. However, it seems we have different experiences and identities in terms of socioeconomic background, ethnicity, and gender. To handle mistakes, we will discuss them during project meetings and go over miscommunications in person for dividing tasks. Additionally, we may check over each other's work to preemptively catch errors.

**Data Curation and Use:** We believe that our data is valid for its intended use. It is simply a collection of global GEOS-CF predictions coupled with real satellite observations. Of any dataset, this one is most well suited for our goal of predicting future atmospheric composition. The most obvious bias that the data could contain is spatial bias. It could be the case that NASA's own predictions or satellite observations are better over certain parts of the globe than others. This may be due to model design choices or sparse observations in specific areas of the globe. We must be aware that this may lead our own forecasting model to make better predictions for certain countries or regions. To combat this, we may look for other data sources to ensure that we have quality data across the entire globe. Furthermore, we may look into ways to enforce spatial fairness constraints into our GNN model. One way we could audit our data and code is to review NASA's own data curation process and make an assessment on its validity and fairness. Additionally, once we have a trained GNN model, we could look at error rates in different regions of the Earth to ensure that our model has similar prediction accuracy across the globe.

**Impact Assessment:** There is a chance we see different error rates for different sub-groups. In this project, sub-groups will be split geographically (e.g. we may have a set of data from western North America, another set from east Asia, etc.); if NASA's GEOS-CF database has not equally sampled from all across the world, then it is possible that some sub-groups with less overall data will have larger error rates due to undertraining on our model. With that said, GEOS-CF holds a fairly mature, developed dataset that has been built by diverse groups of researchers from around the world, so it seems likely that any severe discrepancies in sub-group sampling size have been dealt with by now. One potential path for data misinterpretation could occur when we train our graph neural network. GNN's function by exchanging information with their neighbors; therefore, if individual nodes in the GNN are trained on poor datasets, there is a risk that they decrease the accuracy of neighboring nodes with their own incorrect conclusions and incomplete datasets. A mistake like this could have cascading effects on our entire GNN and severely limit our ability to meaningfully analyze global atmospheric conditons. Preventing this issue goes back to making sure we train our neural network on a good dataset (and, relatedly, taking care to measure the performance of individual nodes and layers, rather than just looking at the GNN as a whole). Fortunately, our GNN deals exclusively with high-level, impersonal atmospheric data, leaving little room for infringing on others' privacy. The only potential privacy risk could come from inadvertently revealing the name of an individual who contributed data to the GEOS-CF dataset (if such data is even recorded by NASA's database).

## Project Description 
The Goddard Earth Observing System composition forecast (GEOS-CF) is NASA's state-of-the-art modeling system for global atmospheric composition [1]. It provides high resolution global forecasts of several chemical species including ozone (O<sub>3</sub>), nitrogen dioxide (NO<sub>2</sub>), and carbon monoxide (CO). Several of these chemicals are air pollutants and/or aersols which have tangible impacts for people around the world. Thus, providing accurate forecasts of their global distribution is key for global health and infrastructure. During the past few years, the development of large global weather, climate, and atmospheric composition models like GEOS-CF which are based on physical laws and differential equations has motivated the creation of machine learning systems which emulate these models [2,3]. The goal of such emulators is to achieve the prediction accuracies of physical models at a fraction of the cost. 

One type of machine learning model that has gained traction in recent years for forecasting tasks is the graph neural network (GNN) [4,5]. These neural networks leverage the structure and properties of graphs and have been successfully applied to emulate global weather features such as temperature, humidity, and wind [6]. This recent success motivates my interest in applying GNNs to emulate the GEOS-CF forecasts. NASA has maintained an assimilated database atmopsheric composition of the Earth since 2018, which is a combination of model predictions and satellite observations. This data could be used to train a GNN, the predictions of which could be compared to the actual GEOS-CF model. If successful, this project could provide a computationally cheaper way to achieve similar results to the GEOS-CF model.

While I think the bulk of the work for this project will be in understanding, designing, and training a reasonable GNN emulator, a potential longer term goal of this project would be to investigate methods for adding physical constraints into the GNN, as neural networks do not necessarily follow laws of physics such as mass conservation out of the box. Ensuring that neural networks are properly physically constrained is a rich research area in data-driven climate science communities [7]. Additionally, should the project succeed, I have a personal goal of writing an academic paper and submitting it for publication to a conference or journal.

[Sam Silva](https://www.samjsilva.com/), Professor of Earth Sciences and Civil and Environmental Engineering at USC, has graciously offerred to provide guidance for this project. He is an expert in computational atmospheric chemistry, and he has ongoing projects applying graph networks to atmopsheric science. His level of involvment will likely depend on how much progress we make on our own and how often we get stuck.

## Project Goals
1. Explore and make use of the NASA's GEOS-CF database of atmospheric composition. Make informed decisions on manipulating the data (e.g., upscaling).
2. Understand and design a reasonable GNN model.
4. Train this model on the GEOS-CF dataset and make any predictions at all (even poor ones).
5. Make informed edits to the model to improve forecasting accuracy.
6. Write an academic paper documenting our model/results and submit it for publication.

## Works Cited
[1] Keller, C. A., Knowland, K. E., Duncan, B. N., Liu, J., Anderson, D. C., Das, S., ... & Pawson, S. (2021). Description of the NASA GEOS composition forecast modeling system GEOS‐CF v1. 0. Journal of Advances in Modeling Earth Systems, 13(4), e2020MS002413.

[2] Rasp, S., Dueben, P. D., Scher, S., Weyn, J. A., Mouatadid, S., & Thuerey, N. (2020). WeatherBench: a benchmark data set for data‐driven weather forecasting. Journal of Advances in Modeling Earth Systems, 12(11), e2020MS002203.

[3] Watson‐Parris, D., Rao, Y., Olivié, D., Seland, Ø., Nowack, P., Camps‐Valls, G., ... & Roesch, C. (2022). ClimateBench v1. 0: A Benchmark for Data‐Driven Climate Projections. Journal of Advances in Modeling Earth Systems, 14(10), e2021MS002954.

[4] Zhou, J., Cui, G., Hu, S., Zhang, Z., Yang, C., Liu, Z., ... & Sun, M. (2020). Graph neural networks: A review of methods and applications. AI Open, 1, 57-81.

[5] Cao, D., Wang, Y., Duan, J., Zhang, C., Zhu, X., Huang, C., ... & Zhang, Q. (2020). Spectral temporal graph neural network for multivariate time-series forecasting. Advances in neural information processing systems, 33, 17766-17778.

[6] Keisler, R. (2022). Forecasting Global Weather with Graph Neural Networks. arXiv preprint arXiv:2202.07575.

[7] Beucler, T., Pritchard, M., Rasp, S., Ott, J., Baldi, P., & Gentine, P. (2021). Enforcing analytic constraints in neural networks emulating physical systems. Physical Review Letters, 126(9), 098302.
