# Analysis of Iowa's Recidivism Data 

## Motivation and Overview

Recidivism, or the re-offense of a previously convicted individual, is a critical topic in criminal justice. For this analysis, I will be focusing on recidivism from an economics standpoint, still recognizing the importance of understanding the human side of this issue. As for some of the economic consequences of recidivism, an individual who spends more time incarcerated is unable to work or otherwise contribute to the economy through consumption. Additionally, the difficulty of securing employment after the end of a sentence--many companies do not hire individuals with prior felony convictions--compounds the stunting of the economic contribution of this individual. The impact goes beyond the incarcerated individual. Higher levels of incarceration due to higher recidivism means more public funds being utilized to maintain the supervision facilities.

So, it is in the public interest to reduce the recidivism rate. While I will not cover potential solutions, leaving that to the public policy realm, we will be striving to gain a better understanding of the driving factors behind recidivism. he analysis is split into three different topics of recidivism to shine some light on the underlying causes:

1) Risk Ranking Determinant Analysis: In this section, we want to gain a better understanding of the determinants of the Risk Ranking assigned to individuals. The rankings have three possible levels: Low, Medium, and High. These correspond to the perceived risk of reoffense prior to an individual's release. We want to gain a better understanding of what factors determine an individual's Ranking, and whether or not there are concerning differences in Risk Ranking based on, for example, immutable characteristics such as race.

2) Risk Ranking Recidivism Prediction: In this section, we want to evaluate the efficacy of these Risk Rankings: do they actually predict recidivism well? We would expect to see a direct positive relationship with level of Risk Ranking and recidivism rate, but if we do not, then the Rankings may not be capturing the true risk of reoffense, but other factors such as systemic bias.

3) Recidvism Differentials: In this section, we want to understand if conditioning on some of the different determinants of Risk Rankings reveals any differentials in the rate of recidivism. That is, are there differences in the risk of reoffense based on race, sex, etc.? Any possible differences can open the door for potential further research into why these differences exist, and consequent interventions directed at closing these gaps.  

## Data 

I will be using recividism data from the State of Iowa's current prison cohort linked [here](https://data.iowa.gov/Correctional-System/Iowa-Prison-Recidivism-Status-Current-Cohort/msmx-x2q6). Note this data has all PII (Personally Identifiable Information) removed.

## Analysis

`main.ipynb` : main narrative notebook that summarizes all the results. This includes all 3 topics covered in Motivation and Overview, as well as concluding remarks.

## License 

This project is released under the terms of the MIT License.
