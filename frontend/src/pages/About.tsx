import { Card, PageHeader } from '../components/ui'

const METRICS = [
  ['Accuracy', 'Share of test samples that were predicted correctly.'],
  ['Precision', 'Of the samples predicted malignant, how many really were malignant.'],
  ['Recall (sensitivity)', 'Of the real malignant samples, how many the model found. Missing a malignant sample is the most serious mistake, so this matters a lot.'],
  ['Specificity', 'Of the real benign samples, how many were correctly called benign.'],
  ['F1 score', 'A single number that balances precision and recall.'],
  ['ROC-AUC', 'How well the model ranks malignant samples above benign ones over all thresholds (1.0 = perfect, 0.5 = random).'],
  ['CV accuracy', 'Average accuracy over 5-fold cross-validation on the training data. Used to choose the best model.'],
]

const MODELS = [
  ['Logistic Regression', 'Draws a straight boundary between the classes. Simple and easy to interpret.'],
  ['Decision Tree', 'Asks a series of yes/no questions about the measurements.'],
  ['Random Forest', 'Combines 200 decision trees and takes a vote.'],
  ['K-Nearest Neighbors', 'Looks at the most similar samples in the training data.'],
  ['Support Vector Machine', 'Finds the widest possible margin between the classes, with a curved (RBF) boundary.'],
  ['Naive Bayes', 'Uses probabilities, assuming the measurements are independent.'],
]

export default function About() {
  return (
    <div className="space-y-6">
      <PageHeader title="About & Methods" subtitle="How the project works, what the numbers mean, and what its limits are." />

      <div className="grid gap-6 lg:grid-cols-2">
        <Card title="Dataset">
          <p className="text-sm text-slate-700">
            The <b>Breast Cancer Wisconsin (Diagnostic)</b> dataset has 569 samples (357 benign, 212 malignant). Each sample
            describes the cell nuclei in a digitised image of a fine needle aspirate (FNA) of a breast mass. This project uses
            the 10 “mean” measurements and gives them simple names. There are no missing values.
          </p>
        </Card>
        <Card title="Method">
          <ol className="list-decimal space-y-1.5 pl-5 text-sm text-slate-700">
            <li>Stratified 80/20 split: 455 training and 114 test samples (random seed 42).</li>
            <li>Each model is a pipeline: <code className="text-xs">StandardScaler → model</code>, so scaling is learned from training data only.</li>
            <li>Grid search with 5-fold cross-validation tunes each model's settings.</li>
            <li>The model with the highest CV accuracy is chosen (the smaller CV spread breaks ties).</li>
            <li>Every model is evaluated once on the test data.</li>
            <li>Prediction level: chance of malignant below 30% = Low, 30–70% = Medium, 70% or more = High.</li>
          </ol>
        </Card>
      </div>

      <div className="grid gap-6 lg:grid-cols-2">
        <Card title="The 6 models">
          <dl className="space-y-2 text-sm">
            {MODELS.map(([m, d]) => (
              <div key={m}><dt className="font-medium text-navy-900">{m}</dt><dd className="text-slate-600">{d}</dd></div>
            ))}
          </dl>
        </Card>
        <Card title="What the metrics mean">
          <dl className="space-y-2 text-sm">
            {METRICS.map(([m, d]) => (
              <div key={m}><dt className="font-medium text-navy-900">{m}</dt><dd className="text-slate-600">{d}</dd></div>
            ))}
          </dl>
        </Card>
      </div>

      <div className="grid gap-6 lg:grid-cols-2">
        <Card title="Explainability">
          <ul className="list-disc space-y-1.5 pl-5 text-sm text-slate-700">
            <li><b>Permutation importance</b>: shuffle one measurement and see how much the test accuracy drops.</li>
            <li><b>Random Forest importance</b> and <b>Logistic Regression coefficients</b>: built into those models.</li>
            <li><b>Per-prediction explanation (Shapley values)</b>: every combination of the 10 measurements is switched between the sample's value and the average training value. Each measurement gets its fair share of the change in the chance of malignant, and the shares add up exactly.</li>
          </ul>
        </Card>
        <Card title="Limitations">
          <ul className="list-disc space-y-1.5 pl-5 text-sm text-slate-700">
            <li>Small dataset (569 samples) from one hospital in the early 1990s. No testing on other hospitals.</li>
            <li>Only 114 test samples, so one mistake changes accuracy by almost 1 percentage point.</li>
            <li>The measurements need a special image analysis process and are not on normal medical reports.</li>
            <li>High accuracy on this dataset does not mean the model would work on real patients.</li>
            <li>This is an educational project, not a medical device.</li>
          </ul>
        </Card>
      </div>

      <Card title="References">
        <ul className="list-disc space-y-1.5 pl-5 text-sm text-slate-700">
          <li>
            Wolberg, W., Mangasarian, O., Street, N., & Street, W. (1993). Breast Cancer Wisconsin (Diagnostic) [Dataset].
            UCI Machine Learning Repository.{' '}
            <a className="text-teal-700 underline" href="https://doi.org/10.24432/C5DW2B" target="_blank" rel="noreferrer">doi:10.24432/C5DW2B</a>
          </li>
          <li>
            Street, W. N., Wolberg, W. H., & Mangasarian, O. L. (1993). Nuclear feature extraction for breast tumor diagnosis.
            SPIE Biomedical Image Processing and Biomedical Visualization, 1905.
          </li>
          <li>
            Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python. Journal of Machine Learning Research, 12.{' '}
            <a className="text-teal-700 underline" href="https://scikit-learn.org" target="_blank" rel="noreferrer">scikit-learn.org</a>
          </li>
        </ul>
      </Card>
    </div>
  )
}
