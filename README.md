🏋️ GymSense ML — Workout Recommendation EngineGymSense ML is a lightweight machine learning pipeline that classifies the optimal workout type based on member biometrics. It trains a model in Python using scikit-learn and exports it as a zero-dependency JSON file, allowing you to run inference in Flutter/Dart without TFLite, ONNX, or any heavy ML libraries.📌 Project OverviewThis repository bridges the gap between Python data science and mobile development by converting complex tree ensembles into a simple nested JSON structure that Dart can parse natively.StepDescriptionEDAVisualizes class balance, gender distribution, and BMI-by-workout-type.PreprocessingHandles Label Encoding, StandardScaler normalization, and stratified 80/20 splitting.TrainingBenchmarks Random Forest, Gradient Boosting, and Decision Trees to select the winner.EvaluationGenerates classification reports, confusion matrices, and feature importance charts.ExportSerializes the winning model into a self-contained gym_model.json.FlutterProvides a ready-to-use Dart class for instant mobile integration.🗂️ Repository StructureBashgymsense-ml/
├── gym_model_sklearn_EN.ipynb   # Main Jupyter Notebook
├── assets/
│   └── gym_model.json           # Final Model (Copy this to Flutter)
├── outputs/                     # Visualizations & Python Pickles
│   ├── gym_model.pkl            # Pickle format (Python use)
│   ├── scaler.pkl               # Scaler parameters
│   ├── confusion_matrix.png     # Model performance chart
│   └── feature_importance.png   # Key biometric drivers
├── requirements.txt             # Python dependencies
└── README.md                    # Project documentation
📊 Dataset SpecificationsThe engine is trained on gym_members_exercise_tracking.csv (approx. 973 rows).Features: Age, Gender, Weight (kg), Height (m), BMI, Experience Level (1-3).Target Classes: Cardio, HIIT, Strength, Yoga.⚡ Quick Start1. Clone & InstallBashgit clone https://github.com/Rozera-xalil/gymsense-ml.git
cd gymsense-ml
pip install -r requirements.txt
2. Train the ModelRun the Jupyter notebook gym_model_sklearn_EN.ipynb. The script will automatically:Process the data.Select the best performing model (highest accuracy).Generate the gym_model.json in the /assets folder.📱 Flutter IntegrationThe exported JSON model can be executed in Dart with a single helper class.1. Add the AssetCopy assets/gym_model.json to your Flutter project and register it in pubspec.yaml:YAMLflutter:
  assets:
    - assets/gym_model.json
2. The Predictor ClassCreate lib/workout_predictor.dart and paste the following logic:Dartimport 'dart:convert';
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
    final fi  = node['feature'] as int;
    final thr = (node['threshold'] as num).toDouble();
    return _traverse(x[fi] <= thr ? node['left'] : node['right'], x);
  }

  Map<String, dynamic> predict({
    required double age,
    required int gender,         // 0 = Female, 1 = Male
    required double weightKg,
    required double heightM,
    required double bmi,
    required int experienceLevel, 
  }) {
    final raw = [age, gender.toDouble(), weightKg, heightM, bmi, experienceLevel.toDouble()];
    final normalized = _normalize(raw);
    final classNames = List<String>.from(_model!['class_names']);
    final trees = _model!['trees'] as List;
    final votes = List<int>.filled(classNames.length, 0);

    for (final tree in trees) {
      votes[_traverse(tree, normalized)]++;
    }

    final maxVotes = votes.reduce((a, b) => a > b ? a : b);
    final bestIdx = votes.indexOf(maxVotes);

    return {
      'recommendation': classNames[bestIdx],
      'confidence': (maxVotes / trees.length * 100).toStringAsFixed(1),
    };
  }
}
📦 RequirementsPython 3.8+scikit-learn >= 1.3pandas, numpy, matplotlib, seaborn👨‍💻 About the AuthorROZÊRA-XELÎL Full Stack Developer & AI Engineering Student“Building intelligent solutions with minimal footprint.”
