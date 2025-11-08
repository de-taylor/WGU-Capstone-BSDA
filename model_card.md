# Model Card

## Model Details

This is a Logistic Regression model created with scikit-learn v1.5.1. After
hyperparameter tuning, this model is using the following hyperparameters:

- `C`: 1.27
- `max_iter`: 500
- `solver`: 'lbfgs'
- `penalty`: 'L2'
- `tol`: 0.0001
- `random_state`: 72925 (for reproducing these results)

## Intended Use

The intended use of this model is to predict the salary classification of an 
individual based on census data, whether that is over $50,000/year, or under/equal
to $50,000/year.

Due to the limitations of this Logistic Regression model, this model should
NOT be used to make decisions at this time. It is not a strong enough recall
model to catch all true positives, though it does have relatively good precision.

## Training Data

Both data sets were derived from the [UCI Census Income dataset](https://archive.ics.uci.edu/dataset/20/census+income), which was
donated on April 30th, 1996 and accessed on November 3rd, 2025.

This is a multivariate dataset with 1 label and 14 features, with a mix of both
categorical and numeric features. Its primary purpose is to demonstrate
classification models.

Per the UCI dataset page, this data was extracted from the 1994 Census database.

### APA Citation

Kohavi, R. (1996). Census Income [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5GP7S.

## Evaluation Data

The original dataset has 32,561 rows and was split into a training-testing split
of 70/30. No stratification was done at this time.

The reason why stratification was omitted was because I was interested in seeing
how the various slices of the model would perform given a random split.

### Preprocessing

For preprocessing on both of the testing and training sets, a OneHotEncoder was
used to encode categorical features, and a StandardScaler was used to scale the
numeric features. This was done to prepare the dataset for L2 regularization for
Logistic Regression. A label binarizer was used on the labels as well.

## Metrics

This model was evaluated with precision, recall, and F1 scores. The overall
metrics are below:

- **Precision**: 0.7371
- **Recall**: 0.6046
- **F1**: 0.6643

To demonstrate this model's effectiveness against random chance, it also had
its accuracy compared to the baseline accuracy. The baseline accuracy was 
calculated as if the majority class label, '<=50K>', was the only label chosen.

- **Baseline Accuracy**: 0.7631
- **Model Accuracy**: 0.8553

This accuracy comparison shows that this model performs above the baseline, though
it is still a conservative model that misses a substantial number of true positives.

## Ethical Considerations

I was unable to run a bias package for this project because the one I was recommended
does not work with the version of Python I used to develop all other parts of this
project. This ethical analysis is performed manually based on the slice data
that can be found under [data/slice_metrics.csv](data/slice_metrics.csv) in
this project.

When evaluating all of the slices for biases and fairness, it is important to keep
the overall precision, recall, and F1 scores in mind. When looking at each slice,
it is a good idea maintain the expectation that we should see slices performing
around the same values, and not significantly higher or lower.

Finally, this data is from the 1994 Census, and is not likely to represent income
in the United States in 2025. Anything learned from this data solely represents
the economic reality in 1994.

All else being held equal, this model performs relatively well in the 'workclass',
'marital status', 'occupation', 'race', and 'gender' features. There are, of course,
notable outliers where performance reaches unexpected highs or lows. This is
primarily due to low representation of those particular categorical slices, as
the outlier rows tend to have very few samples.

With that being said, this model does not have exceptionally high precision,
recall, or F-beta scores. It is a conservative model that, due to its low recall,
tends to miss true positive values. If this model were to be used for decision-
making of any kind, it would be recommended to use a different type of machine
learning algorithm. See the Caveats and Recommendations section below.

## Caveats and Recommendations

Based on the metrics used to evaluate this model, it is not likely to be the
most accurate or useful model to solve this classification problem.

The most likely reason is that Logistic Regression assumes that the underlying
data has a linear relationship. If this is not the case, then performance
may suffer, and the model will be a relatively weaker classifier for this
problem.

During hyperparameter tuning, I ran into a performance plateau, and was
unable to produce significantly different results. This was either due to the
invalid assumption that the data has a linear relationship, there is some
collinearity between features that is not being considered, or I would need
to introduce stratification to the sampling process in case entire categories
were assigned to one side of the split and not the other.

My recommendation for next steps is actually to move away from Logistic Regression,
and instead try to use Support Vector Machines or a Random Forest classifier
instead. These models do not rely on linear relationships (if a non-linear
kernel is chosen in SVM's case), and would be less susceptible to the effects
of collinearity between features.

Even using another algorithm, the next experiment should take advantage of
stratified sampling methods, and potentially KFold cross-validation instead
of the 70-30 train-test split method used in this case.

## Versioning and Maintenance

This model is currently at version 1.0, and was trained on 11/3/2025.

There are no current plans to continue to update or retrain this model given that it
was trained on data that is more than 30 years old and that this project will
not be ongoing.

This model will not be put into full production.