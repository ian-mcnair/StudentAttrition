# StudentAttrition
### Goal: To predict if a student would leave the institution based on non-private, general data.

## General Process
1. Data was given to me by the DBA. Features included general student descriptors: Ethnicity, Gender, Age, Discipline, Zip Code, GPA, etc. It did not include financial, scholarship, or other private information.
2. Data was cleaned. Basic statistical charts and visualizations were made in order to explore data. Heatmaps were also created to give an overview of where students came from and locations that have higher turnover rates.
3. Three ML models were explored, Artificial Neural Networks, Naive Bayes, and K-Nearest Neighbors. The results of these three were combined using a majority voting method, giving overall better results than any one model alone.
4. Findings were coompiled along the way into three different reports, Data Collection, Data Explorations, and Final Forcasting Report. The purpose of these was to inform administration using language and visuals they could easily grasp. 
5. A final research report was compiled to show the scope of what was done in this project.

## Overall Results and Evaluation
- 94% accuracy achieved for forecasting student turnover. Accuracy may be greater however, because students do not all leave at the same time. A follow up study cannot be done at this time.
- 91% accuracy would be achieved for a model guessing all students stay.
- Of the students that left, 30/69 were predicted correctly at the time of the study.
- Recommendations were given to the administration. Students were identified by ID as high risk of leaving. Actual plan to address these students was left up to the discretion of the administration.
More detailed results can be found in the report itself, which is included in this repository

## Reflection of Study
- Class Imbalance: There was overwhelming more students who stayed vs who left. This could have been better addressed with outlier analysis, however at the time of study, this method was not personally known to me. I tried to address this with over and undersampling, but it did not seem to change outcomes much. The biggest impactor was implementing majority voting to the final predictions.
- Evaluation Metric: ROC Curve was probably not the best metric to use because of the class imbalance. Some form of Precision Recall and F1 Score would have probably been better.
- Limited & Inconsistent Data: The data itself was completely human entered. There were mistakes for the zip codes, genders, levels of education. It was difficult to catch all of these, which could have impacted the results. The data itself did not contain some other important pieces of data, specifically related to finances. Knowing which students had scholarships and which lost those as well as parental socioeconomic level could have potentially added to the accuracy of predictions.
- Timeframe of Study: The school actually went through a huge transition which also impacts how well the historal data actually reflects the current data. 
- Although the school was interested in why students left, the scope of this study could only answer which students might leave. The why is needs to be inferred based on reports, potentially surveys of students, and other methods. 

### ROC Curves of 3 Models Used
<img src="https://github.com/ian-mcnair/StudentAttrition/blob/master/Visualizations/ANN%20ROC.png" height="210"> <img src="https://github.com/ian-mcnair/StudentAttrition/blob/master/Visualizations/NB%20ROC.png" height="210"> <img src="https://github.com/ian-mcnair/StudentAttrition/blob/master/Visualizations/KNN%20ROC.png" height="210">
