#!/usr/bin/env python3
"""
Deep Activity Analysis Script
Analyzes the activity_analysis CSV to find correlations, patterns, and insights.

Usage:
    python3 scripts/discovery/deep_activity_analysis.py --input data/discovery/activity_analysis_*.csv
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
from scipy.cluster.hierarchy import linkage, fcluster
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Try to import visualization libraries
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False
    print("Warning: matplotlib/seaborn not available. Skipping visualizations.")


class DeepActivityAnalyzer:
    """Comprehensive statistical analysis of course activity data."""

    def __init__(self, csv_path: str, output_dir: str = None):
        self.csv_path = csv_path
        self.output_dir = Path(output_dir) if output_dir else Path(csv_path).parent / 'deep_analysis'
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load data
        self.df = pd.read_csv(csv_path)
        self.insights = []
        self.stats_report = []

        print(f"Loaded {len(self.df)} courses from {csv_path}")

    def run_full_analysis(self):
        """Run all analysis modules."""
        print("\n" + "="*80)
        print("DEEP ACTIVITY ANALYSIS")
        print("="*80)

        # 1. Data Overview
        self._data_overview()

        # 2. Descriptive Statistics
        self._descriptive_statistics()

        # 3. Correlation Analysis
        self._correlation_analysis()

        # 4. Campus Comparison
        self._campus_analysis()

        # 5. Engagement Patterns
        self._engagement_patterns()

        # 6. Tardiness Analysis
        self._tardiness_analysis()

        # 7. Activity vs Grades
        self._activity_grade_relationship()

        # 8. Clustering Analysis
        self._clustering_analysis()

        # 9. Top Courses Deep Dive
        self._top_courses_analysis()

        # 10. Risk Identification
        self._risk_identification()

        # 11. Temporal Patterns
        self._temporal_patterns()

        # 12. Generate Visualizations
        if HAS_PLOTTING:
            self._generate_visualizations()

        # 13. Save Reports
        self._save_reports()

        # 14. Print Summary
        self._print_summary()

    def _data_overview(self):
        """Basic data overview and quality check."""
        self._section_header("1. DATA OVERVIEW")

        total = len(self.df)
        with_activity = len(self.df[self.df['students_with_activity'] >= 15])
        with_grades = len(self.df[self.df['students_with_grades'] >= 15])

        print(f"  Total Courses: {total}")
        print(f"  With Activity Data (≥15 students): {with_activity} ({with_activity/total*100:.1f}%)")
        print(f"  With Grade Data (≥15 students): {with_grades} ({with_grades/total*100:.1f}%)")
        print(f"  With Both: {len(self.df[(self.df['students_with_activity'] >= 15) & (self.df['students_with_grades'] >= 15)])}")

        # Data quality
        null_counts = self.df.isnull().sum()
        high_null_cols = null_counts[null_counts > total * 0.5]
        if len(high_null_cols) > 0:
            print(f"\n  Columns with >50% null values: {list(high_null_cols.index)}")

        self.insights.append(f"De {total} cursos, solo {with_grades} ({with_grades/total*100:.1f}%) tienen datos de notas suficientes.")

    def _descriptive_statistics(self):
        """Comprehensive descriptive statistics."""
        self._section_header("2. DESCRIPTIVE STATISTICS")

        # Filter to courses with meaningful data
        df_active = self.df[self.df['students_with_activity'] >= 15].copy()

        # Key metrics
        metrics = {
            'total_students': 'Estudiantes por Curso',
            'avg_page_views': 'Page Views Promedio',
            'avg_participations': 'Participaciones Promedio',
            'avg_missing_rate': 'Tasa de Missing (%)',
            'failure_rate': 'Tasa de Reprobación (%)',
            'activity_prediction_score': 'Score de Predicción'
        }

        print("\n  Estadísticas de Métricas Clave (cursos con ≥15 estudiantes activos):")
        print("  " + "-"*76)
        print(f"  {'Métrica':<30} {'Media':>10} {'Mediana':>10} {'Std':>10} {'Min':>8} {'Max':>8}")
        print("  " + "-"*76)

        for col, name in metrics.items():
            if col in df_active.columns:
                data = df_active[col].dropna()
                if len(data) > 0:
                    # Convert rates to percentages for display
                    multiplier = 100 if 'rate' in col and data.max() <= 1 else 1
                    print(f"  {name:<30} {data.mean()*multiplier:>10.1f} {data.median()*multiplier:>10.1f} "
                          f"{data.std()*multiplier:>10.1f} {data.min()*multiplier:>8.1f} {data.max()*multiplier:>8.1f}")

        # Distribution analysis
        print("\n  Distribución de Page Views por Estudiante:")
        pv_quartiles = df_active['avg_page_views'].quantile([0.25, 0.5, 0.75, 0.9, 0.95])
        print(f"    Q1 (25%): {pv_quartiles[0.25]:.0f}")
        print(f"    Q2 (50%): {pv_quartiles[0.5]:.0f}")
        print(f"    Q3 (75%): {pv_quartiles[0.75]:.0f}")
        print(f"    P90: {pv_quartiles[0.9]:.0f}")
        print(f"    P95: {pv_quartiles[0.95]:.0f}")

        self.stats_report.append(('descriptive', df_active.describe()))

    def _correlation_analysis(self):
        """Deep correlation analysis between variables."""
        self._section_header("3. CORRELATION ANALYSIS")

        df_active = self.df[self.df['students_with_activity'] >= 15].copy()

        # Numeric columns for correlation
        numeric_cols = [
            'avg_page_views', 'avg_participations', 'avg_page_views_level', 'avg_participations_level',
            'avg_on_time_rate', 'avg_late_rate', 'avg_missing_rate',
            'grade_mean', 'grade_std', 'failure_rate',
            'assignment_count', 'graded_assignment_count',
            'students_active_last_7_days', 'avg_days_since_last_login',
            'activity_engagement_score', 'tardiness_score', 'activity_prediction_score'
        ]

        available_cols = [c for c in numeric_cols if c in df_active.columns]
        corr_matrix = df_active[available_cols].corr()

        # Find strongest correlations
        correlations = []
        for i, col1 in enumerate(available_cols):
            for j, col2 in enumerate(available_cols):
                if i < j:  # Upper triangle only
                    corr = corr_matrix.loc[col1, col2]
                    if not np.isnan(corr):
                        correlations.append({
                            'var1': col1,
                            'var2': col2,
                            'correlation': corr,
                            'abs_corr': abs(corr)
                        })

        corr_df = pd.DataFrame(correlations).sort_values('abs_corr', ascending=False)

        print("\n  Top 15 Correlaciones Más Fuertes:")
        print("  " + "-"*70)
        for _, row in corr_df.head(15).iterrows():
            direction = "↑↑" if row['correlation'] > 0 else "↑↓"
            strength = "FUERTE" if abs(row['correlation']) > 0.7 else "MODERADA" if abs(row['correlation']) > 0.4 else "DÉBIL"
            print(f"  {direction} {row['var1']:<30} vs {row['var2']:<25} r={row['correlation']:+.3f} ({strength})")

        # Key insights from correlations
        print("\n  Hallazgos de Correlación:")

        # Activity vs Grades
        if 'grade_mean' in df_active.columns:
            df_with_grades = df_active[df_active['students_with_grades'] >= 15]
            if len(df_with_grades) > 10:
                r_pv_grade = df_with_grades[['avg_page_views', 'grade_mean']].corr().iloc[0,1]
                r_part_grade = df_with_grades[['avg_participations', 'grade_mean']].corr().iloc[0,1]
                print(f"    • Page Views vs Nota Promedio: r={r_pv_grade:.3f}")
                print(f"    • Participaciones vs Nota Promedio: r={r_part_grade:.3f}")

                if r_pv_grade > 0.3:
                    self.insights.append(f"Correlación positiva entre page views y notas (r={r_pv_grade:.2f}): más actividad → mejores notas")

        # Missing rate vs failure
        if 'failure_rate' in df_active.columns:
            df_with_fail = df_active[df_active['failure_rate'].notna() & (df_active['students_with_grades'] >= 15)]
            if len(df_with_fail) > 10:
                r_missing_fail = df_with_fail[['avg_missing_rate', 'failure_rate']].corr().iloc[0,1]
                print(f"    • Tasa Missing vs Tasa Reprobación: r={r_missing_fail:.3f}")
                if r_missing_fail > 0.4:
                    self.insights.append(f"Alta correlación entre tareas faltantes y reprobación (r={r_missing_fail:.2f})")

        # Save correlation matrix
        corr_matrix.to_csv(self.output_dir / 'correlation_matrix.csv')
        corr_df.to_csv(self.output_dir / 'top_correlations.csv', index=False)

    def _campus_analysis(self):
        """Compare metrics across campuses."""
        self._section_header("4. CAMPUS COMPARISON")

        # Map account IDs to campus names
        campus_mapping = {
            177: 'Temuco', 178: 'Temuco', 179: 'Temuco', 180: 'Temuco', 181: 'Temuco',
            228: 'San Miguel', 229: 'San Miguel', 230: 'San Miguel', 231: 'San Miguel',
            244: 'Providencia', 245: 'Providencia', 246: 'Providencia', 247: 'Providencia', 248: 'Providencia',
            249: 'Providencia', 250: 'Providencia', 251: 'Providencia'
        }

        # Add parent campus column
        self.df['campus'] = self.df['account_id'].map(lambda x: campus_mapping.get(x, 'Otro'))
        df_active = self.df[self.df['students_with_activity'] >= 15].copy()

        # Group by campus
        campus_stats = df_active.groupby('campus').agg({
            'course_id': 'count',
            'total_students': 'mean',
            'avg_page_views': 'mean',
            'avg_participations': 'mean',
            'avg_missing_rate': 'mean',
            'activity_prediction_score': 'mean',
            'failure_rate': lambda x: x.dropna().mean() if len(x.dropna()) > 0 else np.nan
        }).round(2)

        campus_stats.columns = ['Cursos', 'Est.Prom', 'PageViews', 'Partic.', 'Missing%', 'Score', 'Reprob%']
        campus_stats = campus_stats.sort_values('Cursos', ascending=False)

        print("\n  Comparación por Campus:")
        print("  " + "-"*85)
        print(f"  {'Campus':<15} {'Cursos':>8} {'Est.Prom':>10} {'PageViews':>12} {'Partic.':>10} {'Missing%':>10} {'Score':>8}")
        print("  " + "-"*85)

        for campus, row in campus_stats.iterrows():
            missing_pct = row['Missing%'] * 100 if row['Missing%'] <= 1 else row['Missing%']
            print(f"  {campus:<15} {int(row['Cursos']):>8} {row['Est.Prom']:>10.1f} {row['PageViews']:>12.1f} "
                  f"{row['Partic.']:>10.2f} {missing_pct:>10.1f} {row['Score']:>8.1f}")

        # Statistical test between campuses
        campuses = df_active['campus'].unique()
        if len(campuses) >= 2:
            campus_groups = [df_active[df_active['campus'] == c]['avg_page_views'].dropna() for c in campuses if c != 'Otro']
            if all(len(g) > 5 for g in campus_groups):
                f_stat, p_val = stats.f_oneway(*campus_groups)
                print(f"\n  ANOVA Page Views entre Campus: F={f_stat:.2f}, p={p_val:.4f}")
                if p_val < 0.05:
                    self.insights.append(f"Diferencia significativa en engagement entre campus (p={p_val:.4f})")

        campus_stats.to_csv(self.output_dir / 'campus_comparison.csv')

    def _engagement_patterns(self):
        """Analyze engagement patterns."""
        self._section_header("5. ENGAGEMENT PATTERNS")

        df_active = self.df[self.df['students_with_activity'] >= 15].copy()

        # Create engagement segments
        df_active['engagement_level'] = pd.cut(
            df_active['avg_page_views'],
            bins=[0, 100, 300, 600, 1000, float('inf')],
            labels=['Muy Bajo', 'Bajo', 'Medio', 'Alto', 'Muy Alto']
        )

        engagement_dist = df_active['engagement_level'].value_counts().sort_index()

        print("\n  Distribución de Niveles de Engagement (por Page Views promedio):")
        print("  " + "-"*50)
        for level, count in engagement_dist.items():
            pct = count / len(df_active) * 100
            bar = "█" * int(pct / 2)
            print(f"  {level:<10} {count:>4} cursos ({pct:>5.1f}%) {bar}")

        # Engagement vs outcomes
        print("\n  Engagement vs Resultados (cursos con notas):")
        df_with_grades = df_active[df_active['students_with_grades'] >= 15].copy()
        if len(df_with_grades) > 20:
            df_with_grades['engagement_level'] = pd.cut(
                df_with_grades['avg_page_views'],
                bins=[0, 200, 500, float('inf')],
                labels=['Bajo', 'Medio', 'Alto']
            )

            engagement_outcomes = df_with_grades.groupby('engagement_level').agg({
                'grade_mean': 'mean',
                'failure_rate': 'mean',
                'course_id': 'count'
            }).round(3)

            print("  " + "-"*60)
            print(f"  {'Engagement':<12} {'N':>6} {'Nota Prom':>12} {'Tasa Reprob':>15}")
            print("  " + "-"*60)
            for level, row in engagement_outcomes.iterrows():
                fail_pct = row['failure_rate'] * 100 if row['failure_rate'] <= 1 else row['failure_rate']
                print(f"  {level:<12} {int(row['course_id']):>6} {row['grade_mean']:>12.1f}% {fail_pct:>14.1f}%")

        # Participation patterns
        print("\n  Patrones de Participación:")
        part_stats = df_active['avg_participations'].describe()
        print(f"    Media: {part_stats['mean']:.2f}")
        print(f"    Mediana: {part_stats['50%']:.2f}")
        print(f"    Máximo: {part_stats['max']:.2f}")

        low_part = len(df_active[df_active['avg_participations'] < 1])
        print(f"    Cursos con <1 participación promedio: {low_part} ({low_part/len(df_active)*100:.1f}%)")

        self.insights.append(f"El {low_part/len(df_active)*100:.1f}% de cursos tiene muy baja participación (<1 promedio)")

    def _tardiness_analysis(self):
        """Deep dive into tardiness patterns."""
        self._section_header("6. TARDINESS ANALYSIS")

        df_active = self.df[self.df['students_with_activity'] >= 15].copy()

        # Distribution of tardiness
        print("\n  Distribución de Entrega de Tareas:")
        print("  " + "-"*50)

        on_time_mean = df_active['avg_on_time_rate'].mean() * 100
        late_mean = df_active['avg_late_rate'].mean() * 100
        missing_mean = df_active['avg_missing_rate'].mean() * 100

        print(f"    A Tiempo (promedio): {on_time_mean:.1f}%")
        print(f"    Tarde (promedio): {late_mean:.1f}%")
        print(f"    Missing (promedio): {missing_mean:.1f}%")

        # Create tardiness segments
        df_active['tardiness_level'] = pd.cut(
            df_active['avg_missing_rate'],
            bins=[0, 0.3, 0.5, 0.7, 1.0],
            labels=['Bueno (<30%)', 'Moderado (30-50%)', 'Alto (50-70%)', 'Crítico (>70%)']
        )

        tardiness_dist = df_active['tardiness_level'].value_counts().sort_index()

        print("\n  Segmentación por Tasa de Missing:")
        print("  " + "-"*50)
        for level, count in tardiness_dist.items():
            pct = count / len(df_active) * 100
            bar = "█" * int(pct / 2)
            print(f"  {level:<20} {count:>4} ({pct:>5.1f}%) {bar}")

        # Correlation: tardiness vs grades
        df_with_grades = df_active[df_active['students_with_grades'] >= 15]
        if len(df_with_grades) > 10:
            r, p = stats.pearsonr(
                df_with_grades['avg_missing_rate'].dropna(),
                df_with_grades['failure_rate'].dropna()
            )
            print(f"\n  Correlación Missing Rate vs Failure Rate: r={r:.3f} (p={p:.4f})")

            if r > 0.4 and p < 0.05:
                self.insights.append(f"Tareas faltantes predicen fuertemente la reprobación (r={r:.2f})")

        # High-missing assignments analysis
        if 'assignments_with_high_missing' in df_active.columns:
            high_missing_mean = df_active['assignments_with_high_missing'].mean()
            print(f"\n  Tareas con alto missing (>50%): {high_missing_mean:.1f} por curso en promedio")

    def _activity_grade_relationship(self):
        """Analyze relationship between activity and grades."""
        self._section_header("7. ACTIVITY VS GRADES RELATIONSHIP")

        df_both = self.df[
            (self.df['students_with_activity'] >= 15) &
            (self.df['students_with_grades'] >= 15)
        ].copy()

        if len(df_both) < 20:
            print(f"\n  Insuficientes cursos con ambos datos ({len(df_both)}). Saltando análisis.")
            return

        print(f"\n  Cursos con datos de actividad Y notas: {len(df_both)}")

        # Multiple regression-style analysis
        activity_vars = ['avg_page_views', 'avg_participations', 'avg_missing_rate', 'avg_on_time_rate']

        print("\n  Correlaciones con Nota Promedio:")
        print("  " + "-"*55)

        for var in activity_vars:
            if var in df_both.columns:
                r, p = stats.pearsonr(df_both[var].dropna(), df_both['grade_mean'].dropna())
                sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
                print(f"    {var:<25} r={r:+.3f} (p={p:.4f}) {sig}")

        print("\n  Correlaciones con Tasa de Reprobación:")
        print("  " + "-"*55)

        for var in activity_vars:
            if var in df_both.columns:
                valid = df_both[[var, 'failure_rate']].dropna()
                if len(valid) > 10:
                    r, p = stats.pearsonr(valid[var], valid['failure_rate'])
                    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
                    print(f"    {var:<25} r={r:+.3f} (p={p:.4f}) {sig}")

        # Segmented analysis
        print("\n  Análisis por Cuartiles de Page Views:")
        df_both['pv_quartile'] = pd.qcut(df_both['avg_page_views'], 4, labels=['Q1 (Bajo)', 'Q2', 'Q3', 'Q4 (Alto)'])

        quartile_analysis = df_both.groupby('pv_quartile').agg({
            'grade_mean': ['mean', 'std'],
            'failure_rate': 'mean',
            'course_id': 'count'
        }).round(2)

        print("  " + "-"*65)
        print(f"  {'Cuartil':<15} {'N':>6} {'Nota Prom':>12} {'Nota Std':>10} {'Reprob%':>12}")
        print("  " + "-"*65)

        for idx in quartile_analysis.index:
            row = quartile_analysis.loc[idx]
            fail_pct = row[('failure_rate', 'mean')] * 100 if row[('failure_rate', 'mean')] <= 1 else row[('failure_rate', 'mean')]
            print(f"  {idx:<15} {int(row[('course_id', 'count')]):>6} {row[('grade_mean', 'mean')]:>12.1f}% "
                  f"{row[('grade_mean', 'std')]:>10.1f} {fail_pct:>12.1f}%")

    def _clustering_analysis(self):
        """Cluster courses by behavior patterns."""
        self._section_header("8. CLUSTERING ANALYSIS")

        df_active = self.df[self.df['students_with_activity'] >= 15].copy()

        # Features for clustering
        cluster_features = ['avg_page_views', 'avg_participations', 'avg_missing_rate', 'activity_engagement_score']
        df_cluster = df_active[cluster_features].dropna()

        if len(df_cluster) < 30:
            print("  Insuficientes datos para clustering.")
            return

        # Normalize
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df_cluster)

        # Hierarchical clustering
        linkage_matrix = linkage(X_scaled, method='ward')
        clusters = fcluster(linkage_matrix, t=4, criterion='maxclust')

        df_active.loc[df_cluster.index, 'cluster'] = clusters

        # Analyze clusters
        print("\n  Perfiles de Clusters:")
        print("  " + "-"*75)

        cluster_profiles = df_active.groupby('cluster').agg({
            'course_id': 'count',
            'avg_page_views': 'mean',
            'avg_participations': 'mean',
            'avg_missing_rate': 'mean',
            'activity_prediction_score': 'mean'
        }).round(2)

        # Name clusters based on characteristics
        cluster_names = {}
        for c in cluster_profiles.index:
            profile = cluster_profiles.loc[c]
            if profile['avg_page_views'] > 500 and profile['avg_missing_rate'] < 0.5:
                cluster_names[c] = "Alto Engagement"
            elif profile['avg_page_views'] < 200:
                cluster_names[c] = "Bajo Engagement"
            elif profile['avg_missing_rate'] > 0.7:
                cluster_names[c] = "Alta Deserción"
            else:
                cluster_names[c] = "Engagement Medio"

        print(f"  {'Cluster':<20} {'N':>6} {'PageViews':>12} {'Partic':>10} {'Missing%':>10} {'Score':>8}")
        print("  " + "-"*75)

        for c, row in cluster_profiles.iterrows():
            name = cluster_names.get(c, f"Cluster {c}")
            missing_pct = row['avg_missing_rate'] * 100 if row['avg_missing_rate'] <= 1 else row['avg_missing_rate']
            print(f"  {name:<20} {int(row['course_id']):>6} {row['avg_page_views']:>12.1f} "
                  f"{row['avg_participations']:>10.2f} {missing_pct:>10.1f} {row['activity_prediction_score']:>8.1f}")

        # Save cluster assignments
        df_active[['course_id', 'course_name', 'cluster']].dropna().to_csv(
            self.output_dir / 'course_clusters.csv', index=False
        )

    def _top_courses_analysis(self):
        """Deep dive into top performing courses."""
        self._section_header("9. TOP COURSES ANALYSIS")

        df_active = self.df[self.df['students_with_activity'] >= 15].copy()

        # Top by prediction score
        top_25 = df_active.nlargest(25, 'activity_prediction_score')

        print("\n  Top 25 Cursos por Score de Predicción:")
        print("  " + "-"*95)
        print(f"  {'#':<3} {'Curso':<45} {'Score':>8} {'PageViews':>10} {'Missing%':>10} {'Est':>6}")
        print("  " + "-"*95)

        for i, (_, row) in enumerate(top_25.iterrows(), 1):
            name = row['course_name'][:43] if len(row['course_name']) > 43 else row['course_name']
            missing_pct = row['avg_missing_rate'] * 100 if row['avg_missing_rate'] <= 1 else row['avg_missing_rate']
            print(f"  {i:<3} {name:<45} {row['activity_prediction_score']:>8.1f} "
                  f"{row['avg_page_views']:>10.1f} {missing_pct:>10.1f} {int(row['total_students']):>6}")

        # Characteristics of top courses
        print("\n  Características Comunes de Top 25:")
        print("  " + "-"*50)
        print(f"    Page Views Promedio: {top_25['avg_page_views'].mean():.1f}")
        print(f"    Missing Rate Promedio: {top_25['avg_missing_rate'].mean()*100:.1f}%")
        print(f"    Estudiantes Promedio: {top_25['total_students'].mean():.1f}")

        # Course type patterns
        if 'course_name' in top_25.columns:
            # Extract patterns from names
            patterns = {
                'MATEMÁTICAS/ÁLGEBRA': top_25['course_name'].str.contains('MATEM|ÁLGEBRA|CÁLCULO', case=False).sum(),
                'COMPETENCIAS DIGITALES': top_25['course_name'].str.contains('COMPETENCIAS DIGITALES', case=False).sum(),
                'PSICOLOGÍA': top_25['course_name'].str.contains('PSICOL', case=False).sum(),
                'TALLER': top_25['course_name'].str.contains('TALLER|TALL', case=False).sum()
            }

            print("\n  Tipos de Curso en Top 25:")
            for pattern, count in sorted(patterns.items(), key=lambda x: x[1], reverse=True):
                if count > 0:
                    print(f"    {pattern}: {count} cursos")

        # Save top courses
        top_25.to_csv(self.output_dir / 'top_25_courses.csv', index=False)

    def _risk_identification(self):
        """Identify courses at risk."""
        self._section_header("10. RISK IDENTIFICATION")

        df_active = self.df[self.df['students_with_activity'] >= 15].copy()

        # Risk criteria
        high_missing = df_active['avg_missing_rate'] > 0.7
        low_engagement = df_active['avg_page_views'] < 100
        low_participation = df_active['avg_participations'] < 0.5

        # Combine risks
        risk_count = high_missing.astype(int) + low_engagement.astype(int) + low_participation.astype(int)
        df_active['risk_score'] = risk_count

        print("\n  Distribución de Riesgo:")
        risk_dist = df_active['risk_score'].value_counts().sort_index()

        risk_labels = {0: 'Sin Riesgo', 1: 'Riesgo Bajo', 2: 'Riesgo Medio', 3: 'Riesgo Alto'}

        print("  " + "-"*50)
        for score, count in risk_dist.items():
            pct = count / len(df_active) * 100
            bar = "█" * int(pct / 2)
            label = risk_labels.get(score, f"Score {score}")
            print(f"  {label:<15} {count:>4} cursos ({pct:>5.1f}%) {bar}")

        # High risk courses
        high_risk = df_active[df_active['risk_score'] >= 2].sort_values('risk_score', ascending=False)

        if len(high_risk) > 0:
            print(f"\n  Cursos de Alto Riesgo (score ≥2): {len(high_risk)}")
            print("  " + "-"*80)

            for _, row in high_risk.head(10).iterrows():
                name = row['course_name'][:40] if len(row['course_name']) > 40 else row['course_name']
                missing_pct = row['avg_missing_rate'] * 100 if row['avg_missing_rate'] <= 1 else row['avg_missing_rate']
                print(f"    {name:<42} Missing: {missing_pct:.0f}%, PageViews: {row['avg_page_views']:.0f}")

            self.insights.append(f"{len(high_risk)} cursos identificados como alto riesgo por múltiples indicadores")

        # Save risk analysis
        df_active[['course_id', 'course_name', 'risk_score', 'avg_missing_rate', 'avg_page_views']].to_csv(
            self.output_dir / 'risk_analysis.csv', index=False
        )

    def _temporal_patterns(self):
        """Analyze temporal activity patterns."""
        self._section_header("11. TEMPORAL PATTERNS")

        df_active = self.df[self.df['students_with_activity'] >= 15].copy()

        # Recent activity analysis
        if 'students_active_last_7_days' in df_active.columns:
            recent_7 = df_active['students_active_last_7_days'] / df_active['total_students']
            recent_30 = df_active['students_active_last_30_days'] / df_active['total_students']

            print("\n  Actividad Reciente:")
            print(f"    Estudiantes activos últimos 7 días: {recent_7.mean()*100:.1f}% (promedio)")
            print(f"    Estudiantes activos últimos 30 días: {recent_30.mean()*100:.1f}% (promedio)")

            # Courses with dropping activity
            activity_drop = recent_7.mean() / recent_30.mean() if recent_30.mean() > 0 else 0
            print(f"    Ratio actividad 7d/30d: {activity_drop:.2f}")

        # Days since login
        if 'avg_days_since_last_login' in df_active.columns:
            days_stats = df_active['avg_days_since_last_login'].describe()
            print(f"\n  Días Desde Último Login (promedio por curso):")
            print(f"    Media: {days_stats['mean']:.1f} días")
            print(f"    Mediana: {days_stats['50%']:.1f} días")
            print(f"    Máximo: {days_stats['max']:.1f} días")

            stale_courses = len(df_active[df_active['avg_days_since_last_login'] > 30])
            print(f"    Cursos con >30 días sin login promedio: {stale_courses}")

        # Peak activity day
        if 'peak_activity_day' in df_active.columns:
            peak_dist = df_active['peak_activity_day'].value_counts().head(10)
            print("\n  Días con Mayor Actividad (top 10):")
            for day, count in peak_dist.items():
                print(f"    {day}: {count} cursos")

    def _generate_visualizations(self):
        """Generate all visualizations."""
        self._section_header("12. GENERATING VISUALIZATIONS")

        df_active = self.df[self.df['students_with_activity'] >= 15].copy()

        # Set style
        plt.style.use('seaborn-v0_8-whitegrid')

        # 1. Correlation Heatmap
        fig, ax = plt.subplots(figsize=(14, 12))
        numeric_cols = ['avg_page_views', 'avg_participations', 'avg_on_time_rate', 'avg_missing_rate',
                       'grade_mean', 'failure_rate', 'activity_engagement_score', 'tardiness_score',
                       'activity_prediction_score']
        available = [c for c in numeric_cols if c in df_active.columns]
        corr = df_active[available].corr()
        sns.heatmap(corr, annot=True, cmap='RdYlBu_r', center=0, fmt='.2f', ax=ax)
        ax.set_title('Matriz de Correlación - Métricas de Actividad', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'viz_01_correlation_heatmap.png', dpi=150)
        plt.close()
        print("  Saved: viz_01_correlation_heatmap.png")

        # 2. Engagement Distribution
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Page views distribution
        ax = axes[0, 0]
        df_active['avg_page_views'].hist(bins=50, ax=ax, color='steelblue', edgecolor='white')
        ax.set_xlabel('Page Views Promedio')
        ax.set_ylabel('Número de Cursos')
        ax.set_title('Distribución de Page Views')
        ax.axvline(df_active['avg_page_views'].median(), color='red', linestyle='--', label=f'Mediana: {df_active["avg_page_views"].median():.0f}')
        ax.legend()

        # Missing rate distribution
        ax = axes[0, 1]
        (df_active['avg_missing_rate'] * 100).hist(bins=30, ax=ax, color='coral', edgecolor='white')
        ax.set_xlabel('Tasa de Missing (%)')
        ax.set_ylabel('Número de Cursos')
        ax.set_title('Distribución de Tasa de Missing')

        # Score distribution
        ax = axes[1, 0]
        df_active['activity_prediction_score'].hist(bins=30, ax=ax, color='seagreen', edgecolor='white')
        ax.set_xlabel('Score de Predicción')
        ax.set_ylabel('Número de Cursos')
        ax.set_title('Distribución de Score de Predicción')

        # Participation levels
        ax = axes[1, 1]
        df_active['avg_participations_level'].value_counts().sort_index().plot(kind='bar', ax=ax, color='purple')
        ax.set_xlabel('Nivel de Participación (Canvas)')
        ax.set_ylabel('Número de Cursos')
        ax.set_title('Distribución de Niveles de Participación')

        plt.tight_layout()
        plt.savefig(self.output_dir / 'viz_02_engagement_distributions.png', dpi=150)
        plt.close()
        print("  Saved: viz_02_engagement_distributions.png")

        # 3. Activity vs Missing scatter
        fig, ax = plt.subplots(figsize=(12, 8))
        scatter = ax.scatter(
            df_active['avg_page_views'],
            df_active['avg_missing_rate'] * 100,
            c=df_active['activity_prediction_score'],
            cmap='RdYlGn',
            s=df_active['total_students'] * 2,
            alpha=0.6
        )
        ax.set_xlabel('Page Views Promedio', fontsize=12)
        ax.set_ylabel('Tasa de Missing (%)', fontsize=12)
        ax.set_title('Page Views vs Missing Rate\n(color=Score, tamaño=estudiantes)', fontsize=14, fontweight='bold')
        plt.colorbar(scatter, label='Prediction Score')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'viz_03_activity_vs_missing.png', dpi=150)
        plt.close()
        print("  Saved: viz_03_activity_vs_missing.png")

        # 4. Campus comparison
        if 'campus' in self.df.columns:
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            campus_data = df_active[df_active['campus'] != 'Otro']

            # Page views by campus
            campus_data.boxplot(column='avg_page_views', by='campus', ax=axes[0])
            axes[0].set_title('Page Views por Campus')
            axes[0].set_xlabel('')

            # Missing rate by campus
            campus_data['missing_pct'] = campus_data['avg_missing_rate'] * 100
            campus_data.boxplot(column='missing_pct', by='campus', ax=axes[1])
            axes[1].set_title('Tasa Missing (%) por Campus')
            axes[1].set_xlabel('')

            # Score by campus
            campus_data.boxplot(column='activity_prediction_score', by='campus', ax=axes[2])
            axes[2].set_title('Score de Predicción por Campus')
            axes[2].set_xlabel('')

            plt.suptitle('')
            plt.tight_layout()
            plt.savefig(self.output_dir / 'viz_04_campus_comparison.png', dpi=150)
            plt.close()
            print("  Saved: viz_04_campus_comparison.png")

        # 5. Top 25 courses bar chart
        top_25 = df_active.nlargest(25, 'activity_prediction_score')
        fig, ax = plt.subplots(figsize=(14, 10))

        y_pos = range(len(top_25))
        colors = plt.cm.RdYlGn(top_25['activity_prediction_score'] / 100)

        bars = ax.barh(y_pos, top_25['activity_prediction_score'], color=colors)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([n[:45] for n in top_25['course_name']], fontsize=9)
        ax.set_xlabel('Score de Predicción', fontsize=12)
        ax.set_title('Top 25 Cursos por Score de Predicción de Actividad', fontsize=14, fontweight='bold')
        ax.invert_yaxis()

        # Add score labels
        for i, (score, pv) in enumerate(zip(top_25['activity_prediction_score'], top_25['avg_page_views'])):
            ax.text(score + 1, i, f'{score:.1f} (PV:{pv:.0f})', va='center', fontsize=8)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'viz_05_top_courses.png', dpi=150)
        plt.close()
        print("  Saved: viz_05_top_courses.png")

        # 6. Activity vs Grades (if available)
        df_both = df_active[df_active['students_with_grades'] >= 15]
        if len(df_both) > 20:
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))

            # Page views vs grade
            ax = axes[0]
            ax.scatter(df_both['avg_page_views'], df_both['grade_mean'],
                      c=df_both['failure_rate'], cmap='RdYlGn_r', s=60, alpha=0.7)
            ax.set_xlabel('Page Views Promedio')
            ax.set_ylabel('Nota Promedio (%)')
            ax.set_title('Page Views vs Nota Promedio\n(color=Tasa Reprobación)')

            # Missing rate vs failure rate
            ax = axes[1]
            ax.scatter(df_both['avg_missing_rate'] * 100, df_both['failure_rate'] * 100,
                      c='coral', s=60, alpha=0.7)
            ax.set_xlabel('Tasa de Missing (%)')
            ax.set_ylabel('Tasa de Reprobación (%)')
            ax.set_title('Missing Rate vs Failure Rate')

            # Add trendline
            z = np.polyfit(df_both['avg_missing_rate'].dropna() * 100,
                          df_both['failure_rate'].dropna() * 100, 1)
            p = np.poly1d(z)
            x_line = np.linspace(0, 100, 100)
            ax.plot(x_line, p(x_line), "r--", alpha=0.8, label=f'Tendencia')
            ax.legend()

            plt.tight_layout()
            plt.savefig(self.output_dir / 'viz_06_activity_vs_grades.png', dpi=150)
            plt.close()
            print("  Saved: viz_06_activity_vs_grades.png")

    def _save_reports(self):
        """Save all analysis reports."""
        self._section_header("13. SAVING REPORTS")

        # Save insights
        with open(self.output_dir / 'insights.txt', 'w') as f:
            f.write("INSIGHTS DEL ANÁLISIS DE ACTIVIDAD\n")
            f.write("="*50 + "\n\n")
            for i, insight in enumerate(self.insights, 1):
                f.write(f"{i}. {insight}\n\n")

        print(f"  Saved: insights.txt ({len(self.insights)} insights)")

        # Save full dataframe with added columns
        self.df.to_csv(self.output_dir / 'full_analysis.csv', index=False)
        print(f"  Saved: full_analysis.csv")

    def _print_summary(self):
        """Print final summary."""
        print("\n" + "="*80)
        print("RESUMEN DE INSIGHTS")
        print("="*80)

        for i, insight in enumerate(self.insights, 1):
            print(f"\n  {i}. {insight}")

        print("\n" + "="*80)
        print(f"Análisis completado. Resultados guardados en: {self.output_dir}")
        print("="*80)

    def _section_header(self, title: str):
        """Print section header."""
        print(f"\n{'─'*80}")
        print(f"  {title}")
        print(f"{'─'*80}")


def main():
    parser = argparse.ArgumentParser(description='Deep Activity Analysis')
    parser.add_argument('--input', required=True, help='Input CSV file from activity_analysis.py')
    parser.add_argument('--output-dir', help='Output directory for results')

    args = parser.parse_args()

    # Find the latest file if glob pattern
    input_path = args.input
    if '*' in input_path:
        from glob import glob
        files = sorted(glob(input_path))
        if files:
            input_path = files[-1]
            print(f"Using latest file: {input_path}")
        else:
            print(f"No files found matching: {args.input}")
            return

    analyzer = DeepActivityAnalyzer(input_path, args.output_dir)
    analyzer.run_full_analysis()


if __name__ == '__main__':
    main()
