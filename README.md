# 🏋️ GymSense ML — Workout Recommendation Engine
A lightweight scikit-learn pipeline that classifies the optimal workout type from member biometrics, then exports the model as a plain JSON file that runs in Flutter with zero ML dependencies.

Pythonscikit-learnFlutter

📌 What this repo does
Step	Description
EDA	Visualise class balance, gender split, BMI distribution, and BMI-by-workout-type
Preprocessing	Label-encode categoricals, StandardScaler normalisation, stratified 80/20 split
Training	Benchmark Random Forest · Gradient Boosting · Decision Tree; auto-select best
Evaluation	Classification report, confusion matrix heatmap, feature importance chart
Export	Serialize the winning model as a self-contained JSON file for mobile inference
Flutter	Ready-to-paste Dart class — no TFLite, no ML library required
🗂️ Repository structure
gymsense-ml/
├── gym_model_sklearn_EN.ipynb ← Main notebook (this file)
├── assets/
│ └── gym_model.json ← Model output — copy to Flutter assets/
├── outputs/
│ ├── gym_model.pkl ← Pickled model (Python only)
│ ├── scaler.pkl ← Pickled scaler (Python only)
│ ├── target_encoder.pkl
│ ├── label_encoders.pkl
│ ├── eda_charts.png
│ ├── confusion_matrix.png
│ └── feature_importance.png
├── requirements.txt
└── README.md

text


---

## ⚡ Quick start

### 1 — Clone & install

```bash
git clone https://github.com/Rozera-xalil/gymsense-ml.git
cd gymsense-ml
2 — dataset
text

data\gym_members_exercise_tracking.csv
3 — Run the notebook
bash

jupyter notebook gym_model_sklearn_EN.ipynb
Execute all cells top-to-bottom. The notebook will:

Auto-create the output directory
Train three classifiers and pick the best
Save all artefacts (.pkl, .json, .png)
📊 Dataset
Source: gym_members_exercise_tracking.csv

Column
Type
Description
Age	int	Member age in years
Gender	str	Male / Female
Weight (kg)	float	Body weight
Height (m)	float	Body height
BMI	float	Body Mass Index
Experience_Level	int	1 = Beginner · 2 = Intermediate · 3 = Advanced
Workout_Type	str	Target — Cardio · HIIT · Strength · Yoga

973 rows · 15 columns · no missing values.

🌲 Model
Three tree-based classifiers are trained and evaluated. The one with the highest
test-set accuracy is automatically saved as the production model.

Model
Rationale
Random Forest	Robust ensemble baseline; low variance
Gradient Boosting	Often achieves the highest accuracy via sequential boosting
Decision Tree	Fast, interpretable single-tree baseline

All models are evaluated with:

Accuracy (train & test)
Per-class precision / recall / F1
Confusion matrix heatmap
Feature importance ranking
📱 Flutter integration
The exported gym_model.json can be loaded and executed in Dart with a single
class — no TFLite, no ML package, no platform-specific setup needed.

1 — Add the asset
Copy assets/gym_model.json into your Flutter project's assets/ folder, then
register it in pubspec.yaml:

yaml

flutter:
  assets:
    - assets/gym_model.json
2 — Add the predictor class
Create lib/workout_predictor.dart:

dart

import 'dart:convert';
import 'package:flutter/services.dart';

class WorkoutPredictor {
  Map<String, dynamic>? _model;

  Future<void> loadModel() async {
    final raw = await rootBundle.loadString('assets/gym_model.json');
    _model = json.decode(raw);
  }

  List<double> _normalize(List<double> input) {
    final mean  = List<double>.from(_model!['scaler']['mean']);
    final scale = List<double>.from(_model!['scaler']['scale']);
    return List.generate(input.length, (i) => (input[i] - mean[i]) / scale[i]);
  }

  int _traverse(Map<String, dynamic> node, List<double> x) {
    if (node['leaf'] == true) return node['value'] as int;
    final fi  = node['feature']    as int;
    final thr = (node['threshold'] as num).toDouble();
    return _traverse(x[fi] <= thr ? node['left'] : node['right'], x);
  }

  Map<String, dynamic> predict({
    required double age,
    required int    gender,           // 0 = Female, 1 = Male
    required double weightKg,
    required double heightM,
    required double bmi,
    required int    experienceLevel,  // 1 · 2 · 3
  }) {
    final raw        = [age, gender.toDouble(), weightKg, heightM, bmi, experienceLevel.toDouble()];
    final normalized = _normalize(raw);
    final classNames = List<String>.from(_model!['class_names']);
    final trees      = _model!['trees'] as List;
    final votes      = List<int>.filled(classNames.length, 0);

    for (final tree in trees) {
      votes[_traverse(tree, normalized)]++;
    }

    final maxVotes   = votes.reduce((a, b) => a > b ? a : b);
    final bestIdx    = votes.indexOf(maxVotes);
    final confidence = maxVotes / trees.length * 100;

    return {
      'recommendation': classNames[bestIdx],
      'confidence':     confidence.toStringAsFixed(1),
      'all_scores': Map.fromIterables(
        classNames,
        votes.map((v) => (v / trees.length * 100).toStringAsFixed(1)),
      ),
    };
  }
}
3 — Call it in your widget
dart

final predictor = WorkoutPredictor();
await predictor.loadModel();

final result = predictor.predict(
  age:             25,
  gender:          1,    // Male
  weightKg:        80.0,
  heightM:         1.80,
  bmi:             24.7,
  experienceLevel: 1,    // Beginner
);

print(result['recommendation']); // e.g. "Cardio"
print(result['confidence']);     // e.g. "68.5"
📦 Requirements
text

pandas>=1.5
scikit-learn>=1.3
matplotlib>=3.7
seaborn>=0.12
joblib>=1.3
numpy>=1.24
jupyter
Install everything with:

bash

pip install -r requirements.txt
No TensorFlow · No Keras · No NumPy version conflicts

👨‍💻 About the Author
ROZÊRA-XELÎL - Full Stack Developer & AI Engineering Student

2+2=1
```
