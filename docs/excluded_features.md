       Summary Report: Feature Exclusions in Early Warning Model Training

       Based on my thorough search of the codebase, I've found exactly where features are excluded in the early warning model training. Here's the
       complete analysis:

       1. EXCLUSION PATTERNS (train_optimized_early_warning.py)

       Location: /home/paul/projects/uautonoma/scripts/train_optimized_early_warning.py (lines 78-85)

       Pattern-based exclusion:
       EXCLUDE_PATTERNS = [
           'quiz', 'quizzes',
           'assi', 'assignment',
           'grade', 'grad',
           'score',
           'submission',
       ]

       Function implementing the logic:
       def is_valid_feature(col):
           """Check if a feature should be included (not assessment/grade related)."""
           col_lower = col.lower()

           # Exclude IDs and target
           if col in ['user_id', 'course_id', 'final_score', 'current_score', 'failed', 'enrollment_state']:
               return False

           # Exclude assessment patterns
           for pattern in EXCLUDE_PATTERNS:
               if pattern in col_lower:
                   return False

           return True

       How it works: Any feature containing these substrings (case-insensitive) is excluded, plus explicit ID/target columns.

       ---
       2. EXPLICIT FEATURE EXCLUSION LIST (train_early_warning_model.py)

       Location: /home/paul/projects/uautonoma/scripts/train_early_warning_model.py (lines 44-64)

       Explicitly excluded features:
       EXCLUDED_FEATURES = [
           # Grade-related
           'grades_views', 'grades_views_pct', 'grades_unique_resources', 'grades_time_min',
           'grades_check_per_week', 'grades_first_access_week',
           'grad_mean_pct', 'grad_median_pct', 'grad_std_pct', 'grad_access_rate',
           'grad_top25_rate', 'grad_top50_rate', 'grad_n_resources',
           'grad_hist_b1', 'grad_hist_b2', 'grad_hist_b3', 'grad_hist_b4', 'grad_hist_b5',

           # Score-related (these are actual grades, not activity)
           'avg_score', 'min_score', 'max_score', 'score_std', 'first_score',
           'num_graded', 'num_scores', 'submission_rate',

           # Quiz scores (quizzes are fine as activity, but scores are grades)
           'quiz_score_avg', 'quiz_score_min', 'quiz_score_max',

           # Target variables (obviously exclude)
           'final_score', 'current_score', 'failed', 'enrollment_state',

           # IDs
           'user_id', 'course_id'
       ]

       ---
       3. ACTUAL EXCLUDED FEATURES IN RESULTS

       Location: /home/paul/projects/uautonoma/data/report/early_warning_model_metrics.json

       The optimized model actually excluded 54 feature out of 94 total (57% exclusion rate):

       Top excluded features by category:

       Quiz-related (prefix "quiz_"):
       - quiz_hist_b1, quiz_hist_b2, quiz_hist_b3, quiz_hist_b4, quiz_hist_b5 (histogram bins)
       - quiz_mean_pct, quiz_median_pct, quiz_std_pct (statistics)
       - quiz_access_rate, quiz_top25_rate, quiz_top50_rate (engagement rates)
       - quiz_n_resources, quiz_score_avg/min/max (score features)

       Quiz/assessment-related (prefix "quizzes_"):
       - quizzes_views, quizzes_views_pct, quizzes_time_min, quizzes_unique_resources

       Assignment-related (prefix "assi_"):
       - assi_hist_b1 through assi_hist_b5 (histogram bins)
       - assi_mean_pct, assi_median_pct, assi_std_pct (statistics)
       - assi_access_rate, assi_top25_rate, assi_top50_rate (engagement rates)
       - assi_n_resources, assi_unique_resources

       Assignment-related (prefix "assignments_"):
       - assignments_views, assignments_views_pct, assignments_time_min, assignments_unique_resources

       Grades-related (prefix "grades_"):
       - grades_views, grades_views_pct, grades_unique_resources, grades_time_min
       - grades_check_per_week, grades_first_access_week

       Transition features (bigrams - excluded because they reference assessments):
       - bigram_home_to_assignments, bigram_assignments_to_assignments, bigram_assignments_to_files
       - bigram_assignments_to_home, bigram_assignments_to_quizzes, bigram_quizzes_to_quizzes
       - bigram_other_to_assignments

       Other:
       - jaccard_to_passing (assessment-focused similarity metric)
       - Normalized versions with _znorm suffix for excluded features

       ---
       4. EXCLUSION LOGIC FLOWCHART

       Feature presented to is_valid_feature()
           ↓
       1. Check if in explicit exclude list → EXCLUDE
       2. Check if col_lower contains 'quiz' → EXCLUDE
       3. Check if col_lower contains 'quizzes' → EXCLUDE
       4. Check if col_lower contains 'assi' → EXCLUDE
       5. Check if col_lower contains 'assignment' → EXCLUDE
       6. Check if col_lower contains 'grade' → EXCLUDE
       7. Check if col_lower contains 'grad' → EXCLUDE
       8. Check if col_lower contains 'score' → EXCLUDE
       9. Check if col_lower contains 'submission' → EXCLUDE
       10. If none match → INCLUDE

       ---
       5. WHY FEATURES WERE EXCLUDED

       From documentation in train_optimized_early_warning.py (lines 3-23):

       This script trains the BEST possible model for predicting student failure using ONLY engagement with learning materials (Files, Discussions,
       Pages, Modules, Home, Announcements) and session patterns.

       COMPLETELY EXCLUDES (enables prediction BEFORE first exam):
       - ALL quiz features (quiz_*, quizzes_*)
       - ALL assignment features (assi_*, assignments_*)
       - ALL grades/score features (grades_*, grad_*, score*)

       Key Principle: Enable early warning prediction BEFORE any grades exist - these exclusions ensure the model captures pure engagement behavior,
       not assessment outcomes.

       ---
       6. STATISTICS

       - Total features before exclusion: 94
       - Features included: 40 (43%)
       - Features excluded: 54 (57%)
       - Exclusion categories:
         - Quiz-related: ~15 features
         - Assignment-related: ~18 features
         - Grades-related: ~10 features
         - Bigrams (assessment-related): ~8 features
         - Other (scores, ratios): ~3 features

       ---
       7. FILES IMPLEMENTING THE EXCLUSIONS

       1. /home/paul/projects/uautonoma/scripts/train_optimized_early_warning.py - Primary implementation with pattern matching
       2. /home/paul/projects/uautonoma/scripts/train_early_warning_model.py - Baseline with explicit list
       3. /home/paul/projects/uautonoma/scripts/optimize_threshold_f2.py - Uses same patterns
       4. /home/paul/projects/uautonoma/scripts/generate_shap_explanations.py - Uses same patterns
       5. /home/paul/projects/uautonoma/scripts/pooled_binary_classifier.py - Discusses leakage prevention (lines 58-70)

       This design ensures that the early warning model predicts student failure based purely on engagement patterns (page views, discussions, files,
       modules) rather than assessment performance, enabling interventions before students take their first exams.