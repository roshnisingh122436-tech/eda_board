#!/usr/bin/env python3
"""
Single-File Interactive Exploratory Data Analysis (EDA) Dashboard
Built for educational and classroom use with Flask, Pandas, NumPy, Matplotlib, and Seaborn.

Dataset: Students Performance in Exams (Kaggle)
File: eda_dashboard.py
"""

import os
import sys
import io
import base64
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from flask import Flask, request, jsonify, render_template_string

# Configurable CSV path variable
CSV_PATH = os.environ.get("CSV_PATH", "StudentsPerformance.csv")

# Global DataFrame holder
DF = None
LOAD_ERROR = None


# ---------------------------------------------------------------------------
# Helper function: Convert Matplotlib Figure to Base64 PNG string
# ---------------------------------------------------------------------------
def fig_to_base64(fig):
    """Encodes a Matplotlib figure to a base64 PNG data URL and closes the figure."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=120, bbox_inches='tight')
    buf.seek(0)
    encoded = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close(fig)
    return f"data:image/png;base64,{encoded}"


# ---------------------------------------------------------------------------
# 1. load_dataset()
# ---------------------------------------------------------------------------
def load_dataset(csv_path=CSV_PATH):
    """
    Loads the Students Performance dataset from CSV.
    Validates file existence and returns (df, None) or (None, error_message).
    """
    resolved_path = csv_path
    if not os.path.exists(resolved_path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        alt_path = os.path.join(script_dir, csv_path)
        if os.path.exists(alt_path):
            resolved_path = alt_path
        else:
            return None, f"Dataset not found at '{csv_path}' or '{alt_path}'. Please ensure 'StudentsPerformance.csv' is in the application folder."

    try:
        df = pd.read_csv(resolved_path)
        cleaned_df = clean_dataset(df)
        return cleaned_df, None
    except Exception as e:
        return None, f"Error loading CSV file: {str(e)}"


# ---------------------------------------------------------------------------
# 2. clean_dataset()
# ---------------------------------------------------------------------------
def clean_dataset(df):
    """
    Cleans the raw dataset:
    - Strips whitespace from column names and string cells
    - Ensures score columns are numeric integers/floats
    """
    cleaned = df.copy()
    cleaned.columns = [col.strip() for col in cleaned.columns]

    for col in cleaned.columns:
        if cleaned[col].dtype == 'object':
            cleaned[col] = cleaned[col].astype(str).str.strip()

    score_cols = ['math score', 'reading score', 'writing score']
    for col in score_cols:
        if col in cleaned.columns:
            cleaned[col] = pd.to_numeric(cleaned[col], errors='coerce')

    return cleaned


# ---------------------------------------------------------------------------
# 3. get_dataset_summary()
# ---------------------------------------------------------------------------
def get_dataset_summary(df):
    """
    Computes summary metrics, missing value counts, duplicates,
    column categorization, and descriptive statistics.
    """
    if df is None:
        return {"error": "Dataset not loaded."}

    num_cols = get_numeric_columns(df)
    cat_cols = get_categorical_columns(df)

    avg_math = round(float(df['math score'].mean()), 2) if 'math score' in df else 0.0
    avg_reading = round(float(df['reading score'].mean()), 2) if 'reading score' in df else 0.0
    avg_writing = round(float(df['writing score'].mean()), 2) if 'writing score' in df else 0.0
    overall_avg = round((avg_math + avg_reading + avg_writing) / 3.0, 2)

    missing_dict = {col: int(df[col].isnull().sum()) for col in df.columns}
    total_missing = sum(missing_dict.values())
    total_duplicates = int(df.duplicated().sum())

    stats_dict = {}
    for col in num_cols:
        stats_dict[col] = {
            "count": int(df[col].count()),
            "mean": round(float(df[col].mean()), 2),
            "std": round(float(df[col].std()), 2),
            "min": round(float(df[col].min()), 2),
            "q25": round(float(df[col].quantile(0.25)), 2),
            "median": round(float(df[col].median()), 2),
            "q75": round(float(df[col].quantile(0.75)), 2),
            "max": round(float(df[col].max()), 2),
        }

    preview_records = df.head(10).to_dict(orient='records')

    return {
        "total_students": len(df),
        "total_columns": len(df.columns),
        "columns": list(df.columns),
        "numeric_columns": num_cols,
        "categorical_columns": cat_cols,
        "avg_math_score": avg_math,
        "avg_reading_score": avg_reading,
        "avg_writing_score": avg_writing,
        "overall_avg_score": overall_avg,
        "missing_values": missing_dict,
        "total_missing": total_missing,
        "duplicate_count": total_duplicates,
        "descriptive_stats": stats_dict,
        "preview": preview_records
    }


# ---------------------------------------------------------------------------
# 4. get_numeric_columns()
# ---------------------------------------------------------------------------
def get_numeric_columns(df):
    """Returns a list of all numeric column names in the DataFrame."""
    if df is None:
        return []
    return list(df.select_dtypes(include=[np.number]).columns)


# ---------------------------------------------------------------------------
# 5. get_categorical_columns()
# ---------------------------------------------------------------------------
def get_categorical_columns(df):
    """Returns a list of all categorical/object column names in the DataFrame."""
    if df is None:
        return []
    return list(df.select_dtypes(include=['object', 'category']).columns)


# ---------------------------------------------------------------------------
# 6. create_histogram()
# ---------------------------------------------------------------------------
def create_histogram(df, column):
    """
    Generates a univariate histogram with KDE curve and mean/median reference lines.
    Returns base64 PNG data URI.
    """
    if df is None or column not in df.columns:
        raise ValueError(f"Column '{column}' not found in dataset.")
    if column not in get_numeric_columns(df):
        raise ValueError(f"Histogram requires a numerical column, but '{column}' is categorical.")

    fig, ax = plt.subplots(figsize=(8, 4.8))
    data = df[column].dropna()
    mean_val = float(data.mean())
    median_val = float(data.median())

    sns.histplot(data=df, x=column, bins=15, kde=True, color="#2563eb", edgecolor="white", alpha=0.65, ax=ax)
    ax.axvline(mean_val, color="#dc2626", linestyle="--", linewidth=2, label=f"Mean: {mean_val:.1f}")
    ax.axvline(median_val, color="#16a34a", linestyle=":", linewidth=2, label=f"Median: {median_val:.1f}")

    ax.set_title(f"Histogram: Distribution of {column.title()}", fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel(f"{column.title()} (Score out of 100)", fontsize=11)
    ax.set_ylabel("Student Frequency (Count)", fontsize=11)
    ax.set_xlim(0, 105)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    ax.legend(frameon=True, facecolor='white', loc='upper left')

    fig.tight_layout()
    return fig_to_base64(fig)


# ---------------------------------------------------------------------------
# 7. create_bar_chart()
# ---------------------------------------------------------------------------
def create_bar_chart(df, x_column, y_column):
    """
    Generates a bivariate bar chart showing average Y score across categories of X.
    Returns base64 PNG data URI.
    """
    if df is None or x_column not in df.columns or y_column not in df.columns:
        raise ValueError("Invalid column selection for bar chart.")

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    grouped = df.groupby(x_column)[y_column].mean().reset_index()
    grouped = grouped.sort_values(by=y_column, ascending=False)

    palette = sns.color_palette("Blues_r", len(grouped))
    bars = ax.bar(grouped[x_column], grouped[y_column], color=palette, edgecolor="#1e3a8a", width=0.55)

    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, h + 1.2, f"{h:.1f}", ha='center', va='bottom', fontsize=10, fontweight='bold', color='#1e293b')

    ax.set_title(f"Bar Chart: Average {y_column.title()} by {x_column.title()}", fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel(x_column.title(), fontsize=11)
    ax.set_ylabel(f"Average {y_column.title()}", fontsize=11)
    ax.set_ylim(0, 105)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    if len(str(grouped[x_column].iloc[0])) > 8:
        plt.xticks(rotation=20, ha='right')

    fig.tight_layout()
    return fig_to_base64(fig)


# ---------------------------------------------------------------------------
# 8. create_line_chart()
# ---------------------------------------------------------------------------
def create_line_chart(df, x_column, y_column):
    """
    Generates an ordered progression line chart across ranked categories or percentiles.
    Returns base64 PNG data URI.
    """
    if df is None or x_column not in df.columns or y_column not in df.columns:
        raise ValueError("Invalid column selection for line chart.")

    fig, ax = plt.subplots(figsize=(9, 4.8))

    education_hierarchy = [
        "some high school", "high school", "some college",
        "associate's degree", "bachelor's degree", "master's degree"
    ]

    if x_column == "parental level of education":
        grouped = df.groupby(x_column)[y_column].mean().reindex(education_hierarchy).dropna().reset_index()
    elif x_column in get_categorical_columns(df):
        grouped = df.groupby(x_column)[y_column].mean().sort_values().reset_index()
    else:
        df_sorted = df.sort_values(by=x_column).copy()
        df_sorted['bin'] = pd.qcut(df_sorted[x_column], q=10, duplicates='drop')
        grouped = df_sorted.groupby('bin', observed=False)[y_column].mean().reset_index()
        grouped[x_column] = [f"Decile {i+1}" for i in range(len(grouped))]

    x_vals = grouped[x_column].astype(str)
    y_vals = grouped[y_column]

    ax.plot(x_vals, y_vals, marker='o', markersize=8, color="#0284c7", linewidth=2.5, label=f"Mean {y_column.title()}")
    ax.fill_between(range(len(x_vals)), y_vals, alpha=0.15, color="#0284c7")

    for i, (xv, yv) in enumerate(zip(x_vals, y_vals)):
        ax.text(i, yv + 0.8, f"{yv:.1f}", ha='center', va='bottom', fontsize=9.5, fontweight='bold', color="#0369a1")

    ax.set_title(f"Line Chart: Score Progression in {y_column.title()} by {x_column.title()}", fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel(f"Ordered {x_column.title()}", fontsize=11)
    ax.set_ylabel(f"Mean {y_column.title()}", fontsize=11)
    ax.set_ylim(min(y_vals) - 5, max(y_vals) + 6)
    ax.grid(True, linestyle='--', alpha=0.4)
    plt.xticks(rotation=20, ha='right')

    fig.tight_layout()
    return fig_to_base64(fig)


# ---------------------------------------------------------------------------
# 9. create_box_plot()
# ---------------------------------------------------------------------------
def create_box_plot(df, column, category_column=None):
    """
    Generates a box plot (univariate or bivariate grouped by category).
    Shows median, IQR, whiskers, and fliers/outliers.
    Returns base64 PNG data URI.
    """
    if df is None or column not in df.columns:
        raise ValueError(f"Column '{column}' not found.")
    if column not in get_numeric_columns(df):
        raise ValueError(f"Box plot requires a numerical target column, got '{column}'.")

    fig, ax = plt.subplots(figsize=(8.5, 4.8))

    if category_column and category_column in df.columns:
        sns.boxplot(data=df, x=category_column, y=column, hue=category_column, legend=False, palette="Set2", ax=ax,
                    flierprops={'marker': 'o', 'markersize': 5, 'markerfacecolor': '#dc2626', 'alpha': 0.6})
        ax.set_title(f"Box Plot: {column.title()} Distribution by {category_column.title()}", fontsize=13, fontweight='bold', pad=12)
        ax.set_xlabel(category_column.title(), fontsize=11)
        if len(str(df[category_column].iloc[0])) > 8 or df[category_column].nunique() > 3:
            plt.xticks(rotation=20, ha='right')
    else:
        sns.boxplot(y=df[column], color="#93c5fd", width=0.35, ax=ax,
                    flierprops={'marker': 'o', 'markersize': 6, 'markerfacecolor': '#dc2626', 'alpha': 0.7})
        median_val = float(df[column].median())
        q1_val = float(df[column].quantile(0.25))
        q3_val = float(df[column].quantile(0.75))
        ax.text(0.22, median_val, f"Median: {median_val:.1f}", va='center', fontsize=10, fontweight='bold', color='#16a34a')
        ax.text(0.22, q1_val, f"Q1: {q1_val:.1f}", va='center', fontsize=9.5, color='#475569')
        ax.text(0.22, q3_val, f"Q3: {q3_val:.1f}", va='center', fontsize=9.5, color='#475569')
        ax.set_title(f"Box Plot: {column.title()} Summary (Median, Quartiles & Outliers)", fontsize=13, fontweight='bold', pad=12)
        ax.set_xlabel("Overall Cohort", fontsize=11)

    ax.set_ylabel(f"{column.title()} (0 - 100)", fontsize=11)
    ax.set_ylim(-2, 105)
    ax.grid(axis='y', linestyle='--', alpha=0.4)

    fig.tight_layout()
    return fig_to_base64(fig)


# ---------------------------------------------------------------------------
# 10. create_scatter_plot()
# ---------------------------------------------------------------------------
def create_scatter_plot(df, x_column, y_column, hue=None):
    """
    Generates a scatter plot comparing two numerical variables with an optional
    categorical hue and a linear regression trendline with Pearson r.
    Returns base64 PNG data URI.
    """
    if df is None or x_column not in df.columns or y_column not in df.columns:
        raise ValueError("Invalid columns selected for scatter plot.")

    fig, ax = plt.subplots(figsize=(8.5, 5.0))

    if hue and hue in df.columns:
        sns.scatterplot(data=df, x=x_column, y=y_column, hue=hue, alpha=0.65, s=40, palette="tab10", ax=ax)
        ax.legend(title=hue.title(), frameon=True, facecolor='white', loc='upper left')
    else:
        ax.scatter(df[x_column], df[y_column], alpha=0.55, color="#2563eb", s=35, edgecolors='none')

    # Fit linear regression trend line
    x_clean = df[x_column].dropna()
    y_clean = df[y_column].dropna()
    m, b = np.polyfit(x_clean, y_clean, 1)
    x_trend = np.linspace(x_clean.min(), x_clean.max(), 100)
    y_trend = m * x_trend + b
    corr_coef = float(df[x_column].corr(df[y_column]))

    ax.plot(x_trend, y_trend, color="#dc2626", linewidth=2.2, linestyle="--", label=f"Trendline (r = {corr_coef:.2f})")
    if not (hue and hue in df.columns):
        ax.legend(loc='upper left', frameon=True, facecolor='white')

    ax.set_title(f"Scatter Plot: {y_column.title()} vs {x_column.title()} (Pearson r = {corr_coef:.2f})", fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel(f"{x_column.title()} (0 - 100)", fontsize=11)
    ax.set_ylabel(f"{y_column.title()} (0 - 100)", fontsize=11)
    ax.set_xlim(0, 105)
    ax.set_ylim(0, 105)
    ax.grid(True, linestyle='--', alpha=0.4)

    fig.tight_layout()
    return fig_to_base64(fig)


# ---------------------------------------------------------------------------
# 11. create_heatmap()
# ---------------------------------------------------------------------------
def create_heatmap(df):
    """
    Generates a correlation heatmap across all numerical score columns.
    Returns base64 PNG data URI.
    """
    if df is None:
        raise ValueError("Dataset is not loaded.")

    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    num_df = df[get_numeric_columns(df)]
    corr = num_df.corr()

    labels = [c.replace(' score', '').title() + ' Score' for c in corr.columns]
    sns.heatmap(
        corr,
        annot=True,
        cmap="coolwarm",
        fmt=".3f",
        vmin=0.7,
        vmax=1.0,
        linewidths=2.0,
        linecolor='white',
        cbar_kws={'label': 'Pearson Correlation Coefficient (r)'},
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
        annot_kws={'size': 12, 'weight': 'bold'}
    )

    ax.set_title("Correlation Heatmap: Student Exam Scores", fontsize=13, fontweight='bold', pad=14)
    plt.xticks(rotation=0, fontsize=10.5)
    plt.yticks(rotation=0, fontsize=10.5)

    fig.tight_layout()
    return fig_to_base64(fig)


# ---------------------------------------------------------------------------
# 12. create_count_plot()
# ---------------------------------------------------------------------------
def create_count_plot(df, column):
    """
    Generates a categorical count plot displaying frequency and percentage.
    Returns base64 PNG data URI.
    """
    if df is None or column not in df.columns:
        raise ValueError(f"Column '{column}' not found.")
    if column not in get_categorical_columns(df):
        raise ValueError(f"Count plot requires a categorical column, got '{column}'.")

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    counts = df[column].value_counts()
    total = len(df)

    palette = sns.color_palette("mako", len(counts))
    bars = ax.bar(counts.index, counts.values, color=palette, edgecolor="#1e293b", width=0.55)

    for bar in bars:
        h = bar.get_height()
        pct = (h / total) * 100.0
        ax.text(bar.get_x() + bar.get_width() / 2.0, h + (total * 0.015),
                f"{h}\n({pct:.1f}%)", ha='center', va='bottom', fontsize=9.5, fontweight='bold', color='#0f172a')

    ax.set_title(f"Count Plot: Frequency of Students by {column.title()}", fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel(column.title(), fontsize=11)
    ax.set_ylabel("Student Count", fontsize=11)
    ax.set_ylim(0, max(counts.values) * 1.22)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    if len(str(counts.index[0])) > 8 or len(counts) > 3:
        plt.xticks(rotation=20, ha='right')

    fig.tight_layout()
    return fig_to_base64(fig)


# ---------------------------------------------------------------------------
# 13. create_pie_chart()
# ---------------------------------------------------------------------------
def create_pie_chart(df, column):
    """
    Generates a proportional pie chart for categorical variables.
    Returns base64 PNG data URI.
    """
    if df is None or column not in df.columns:
        raise ValueError(f"Column '{column}' not found.")
    if column not in get_categorical_columns(df):
        raise ValueError(f"Pie chart requires a categorical column, got '{column}'.")

    fig, ax = plt.subplots(figsize=(6.5, 5.0))
    counts = df[column].value_counts()

    # Subtle explode for the largest category
    explode = [0.06 if i == 0 else 0 for i in range(len(counts))]
    colors = sns.color_palette("pastel", len(counts))

    wedges, texts, autotexts = ax.pie(
        counts.values,
        labels=counts.index,
        autopct='%1.1f%%',
        startangle=140,
        explode=explode,
        colors=colors,
        shadow=True,
        wedgeprops={'edgecolor': '#475569', 'linewidth': 0.8}
    )

    for at in autotexts:
        at.set_color('#0f172a')
        at.set_fontsize(10)
        at.set_weight('bold')

    ax.set_title(f"Pie Chart: Proportions of {column.title()}", fontsize=13, fontweight='bold', pad=14)

    fig.tight_layout()
    return fig_to_base64(fig)


# ---------------------------------------------------------------------------
# 14. create_pairplot()
# ---------------------------------------------------------------------------
def create_pairplot(df, hue=None):
    """
    Generates a Seaborn Pair Plot grid of all exam score distributions
    and pairwise scatter relationships, optionally colored by hue.
    Returns base64 PNG data URI.
    """
    if df is None:
        raise ValueError("Dataset not loaded.")

    cols = ['math score', 'reading score', 'writing score']
    use_cols = cols.copy()

    if hue and hue in df.columns and hue in get_categorical_columns(df):
        use_cols.append(hue)
        g = sns.pairplot(
            df[use_cols],
            hue=hue,
            palette="Set2",
            corner=False,
            height=2.3,
            aspect=1.15,
            plot_kws={'alpha': 0.6, 's': 22}
        )
        hue_title = f" (Colored by {hue.title()})"
    else:
        g = sns.pairplot(
            df[use_cols],
            corner=False,
            height=2.3,
            aspect=1.15,
            plot_kws={'alpha': 0.6, 's': 22, 'color': '#2563eb'},
            diag_kws={'color': '#2563eb'}
        )
        hue_title = ""

    g.fig.subplots_adjust(top=0.92)
    g.fig.suptitle(f"Pair Plot: Multivariate Score Interactions{hue_title}", fontsize=13, fontweight='bold')

    buf = io.BytesIO()
    g.savefig(buf, format='png', dpi=110, bbox_inches='tight')
    plt.close(g.fig)
    encoded = base64.b64encode(buf.getvalue()).decode('utf-8')
    return f"data:image/png;base64,{encoded}"


# ---------------------------------------------------------------------------
# Helper: create_side_by_side_comparison()
# ---------------------------------------------------------------------------
def create_side_by_side_comparison(df, chart_type):
    """
    Generates a side-by-side visual comparison figure:
    Left: Matplotlib implementation | Right: Seaborn implementation.
    Used in Section 5 (Matplotlib vs Seaborn).
    """
    if df is None:
        raise ValueError("Dataset not loaded.")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.6))

    if chart_type == "histogram":
        # Left: Matplotlib
        ax1.hist(df['math score'], bins=15, color='#3b82f6', edgecolor='white', alpha=0.7)
        ax1.set_title("Matplotlib: plt.hist(bins=15)", fontsize=11, fontweight='bold', color='#1e3a8a')
        ax1.set_xlabel("Math Score")
        ax1.set_ylabel("Student Count")
        ax1.grid(axis='y', linestyle='--', alpha=0.4)

        # Right: Seaborn
        sns.histplot(data=df, x="math score", bins=15, kde=True, color="#059669", ax=ax2)
        ax2.set_title("Seaborn: sns.histplot(kde=True)", fontsize=11, fontweight='bold', color='#065f46')
        ax2.set_xlabel("Math Score")
        ax2.set_ylabel("Count")

    elif chart_type == "bar_chart":
        # Left: Matplotlib
        means = df.groupby('lunch')['math score'].mean()
        ax1.bar(means.index, means.values, color=['#f59e0b', '#3b82f6'], edgecolor='#1e293b', width=0.5)
        for i, v in enumerate(means.values):
            ax1.text(i, v + 1.5, f"{v:.1f}", ha='center', fontweight='bold')
        ax1.set_title("Matplotlib: plt.bar(groups, means)", fontsize=11, fontweight='bold', color='#1e3a8a')
        ax1.set_xlabel("Lunch")
        ax1.set_ylabel("Average Math Score")
        ax1.set_ylim(0, 100)
        ax1.grid(axis='y', linestyle='--', alpha=0.4)

        # Right: Seaborn
        sns.barplot(data=df, x='lunch', y='math score', hue='lunch', legend=False, ax=ax2, palette=['#f59e0b', '#3b82f6'], errorbar=None)
        for p in ax2.patches:
            ax2.annotate(f"{p.get_height():.1f}", (p.get_x() + p.get_width() / 2., p.get_height() + 1.5),
                         ha='center', va='bottom', fontweight='bold')
        ax2.set_title("Seaborn: sns.barplot(data=df, ...)", fontsize=11, fontweight='bold', color='#065f46')
        ax2.set_xlabel("Lunch")
        ax2.set_ylabel("Average Math Score")
        ax2.set_ylim(0, 100)

    elif chart_type == "box_plot":
        # Left: Matplotlib
        groups = [df[df['gender'] == 'female']['math score'], df[df['gender'] == 'male']['math score']]
        ax1.boxplot(groups, tick_labels=['Female', 'Male'], patch_artist=True,
                    boxprops=dict(facecolor='#93c5fd', color='#1e40af'))
        ax1.set_title("Matplotlib: plt.boxplot(data_groups)", fontsize=11, fontweight='bold', color='#1e3a8a')
        ax1.set_xlabel("Gender")
        ax1.set_ylabel("Math Score")
        ax1.grid(axis='y', linestyle='--', alpha=0.4)

        # Right: Seaborn
        sns.boxplot(data=df, x='gender', y='math score', hue='gender', legend=False, ax=ax2, palette='Set2')
        ax2.set_title("Seaborn: sns.boxplot(data=df, ...)", fontsize=11, fontweight='bold', color='#065f46')
        ax2.set_xlabel("Gender")
        ax2.set_ylabel("Math Score")

    elif chart_type == "scatter_plot":
        # Left: Matplotlib
        ax1.scatter(df['reading score'], df['writing score'], alpha=0.5, color='#0284c7', s=30)
        ax1.set_title("Matplotlib: plt.scatter(x, y)", fontsize=11, fontweight='bold', color='#1e3a8a')
        ax1.set_xlabel("Reading Score")
        ax1.set_ylabel("Writing Score")
        ax1.grid(True, linestyle='--', alpha=0.4)

        # Right: Seaborn
        sns.scatterplot(data=df, x='reading score', y='writing score', hue='gender', alpha=0.6, s=30, ax=ax2)
        ax2.set_title("Seaborn: sns.scatterplot(hue='gender')", fontsize=11, fontweight='bold', color='#065f46')
        ax2.set_xlabel("Reading Score")
        ax2.set_ylabel("Writing Score")

    elif chart_type == "heatmap":
        # Left: Matplotlib
        corr = df[['math score', 'reading score', 'writing score']].corr()
        im = ax1.imshow(corr, cmap='coolwarm', vmin=0.7, vmax=1.0)
        ax1.set_xticks(range(3))
        ax1.set_yticks(range(3))
        ax1.set_xticklabels(['Math', 'Reading', 'Writing'])
        ax1.set_yticklabels(['Math', 'Reading', 'Writing'])
        for i in range(3):
            for j in range(3):
                ax1.text(j, i, f"{corr.iloc[i, j]:.2f}", ha='center', va='center', fontweight='bold')
        fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
        ax1.set_title("Matplotlib: plt.imshow(corr)", fontsize=11, fontweight='bold', color='#1e3a8a')

        # Right: Seaborn
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', vmin=0.7, vmax=1.0, ax=ax2,
                    xticklabels=['Math', 'Reading', 'Writing'], yticklabels=['Math', 'Reading', 'Writing'])
        ax2.set_title("Seaborn: sns.heatmap(corr, annot=True)", fontsize=11, fontweight='bold', color='#065f46')

    plt.suptitle(f"Side-by-Side Comparison: {chart_type.replace('_', ' ').title()}", fontsize=13, fontweight='bold', y=1.02)
    fig.tight_layout()
    return fig_to_base64(fig)


# ---------------------------------------------------------------------------
# 15. get_chart_explanation()
# ---------------------------------------------------------------------------
def get_chart_explanation(chart_type):
    """
    Returns an educational explanation package for students:
    - title
    - explanation
    - when_to_use
    - what_to_observe
    - limitations
    """
    explanations = {
        "histogram": {
            "title": "Histogram (Univariate Distribution)",
            "explanation": "A histogram groups continuous numeric data into consecutive bins and shows the count or density of students in each range.",
            "when_to_use": "Use when analyzing one continuous numerical variable (e.g. math score) to inspect the shape, spread, and central tendency of test results.",
            "what_to_observe": "Observe: (1) Shape (is it symmetric/bell-shaped or skewed?), (2) Peak/Mode (around 65-70 points), (3) Spread (range from near 0 to 100), and (4) Outliers on the left tail.",
            "limitations": "Bin size strongly influences visual appearance. It does not show individual data points or compare multiple sub-groups easily."
        },
        "bar_chart": {
            "title": "Bar Chart (Categorical vs Aggregated Metric)",
            "explanation": "A bar chart displays a summary statistic (such as average score) for distinct categories using rectangular bars with heights proportional to values.",
            "when_to_use": "Use when comparing average performance across categorical groups (e.g. comparing average math scores by lunch type or parental education).",
            "what_to_observe": "Observe: (1) Which group has the highest and lowest mean score, (2) The magnitude of the performance gap (e.g. standard lunch averages ~70 vs free/reduced ~59), and (3) Relative rankings.",
            "limitations": "Shows only the aggregate summary (mean); it hides internal group variance, spread, and sample size differences."
        },
        "line_chart": {
            "title": "Line Chart (Ordered Progression & Trends)",
            "explanation": "A line chart connects ordered data points with line segments to display progression, sequences, or ranked transitions.",
            "when_to_use": "Use when categories have a natural hierarchy or sequence (e.g. parental education from 'some high school' to 'master's degree') or score percentiles.",
            "what_to_observe": "Observe: (1) Direction of slope (positive upward slope confirms students whose parents attained higher degrees score higher), (2) Steepness of increments, and (3) Plateaus.",
            "limitations": "Implies continuity between points; inappropriate for unrelated unordered nominal categories (like arbitrary race groups)."
        },
        "box_plot": {
            "title": "Box Plot (5-Number Summary & Outlier Detection)",
            "explanation": "A box plot summarizes data using 5 key numbers: Minimum, 1st Quartile (Q1), Median (Q2), 3rd Quartile (Q3), and Maximum, with outliers marked as isolated points beyond 1.5 * IQR.",
            "when_to_use": "Use when comparing distributions, medians, and spreads across categories or identifying statistical outliers in student test scores.",
            "what_to_observe": "Observe: (1) The solid median line, (2) The Interquartile Range (box height = middle 50% of students), (3) Whiskers range, and (4) Red outlier dots below the lower whisker (e.g. score of 0 in math).",
            "limitations": "Hides multi-modal distributions (a bimodal distribution can produce an identical boxplot to a unimodal distribution)."
        },
        "scatter_plot": {
            "title": "Scatter Plot (Bivariate Correlation & Clusters)",
            "explanation": "A scatter plot positions individual observations on Cartesian coordinates to reveal relationships, correlations, and clusters between two numeric variables.",
            "when_to_use": "Use when investigating whether two exam subjects correlate (e.g. does a high reading score correspond to a high writing score?).",
            "what_to_observe": "Observe: (1) Direction of correlation (positive diagonal line), (2) Tightness along the red regression line (r = 0.95 indicates very strong correlation), and (3) Any unusual outliers.",
            "limitations": "Suffers from overplotting when points overlap; correlation does not imply direct causation."
        },
        "heatmap": {
            "title": "Correlation Heatmap (Multivariate Linear Relationships)",
            "explanation": "A heatmap displays a matrix of Pearson correlation coefficients (r) colored by intensity, where warm red indicates strong positive correlation and cool blue indicates negative correlation.",
            "when_to_use": "Use during initial multivariate EDA to evaluate pairwise relationships among all numerical variables simultaneously.",
            "what_to_observe": "Observe: (1) The diagonal is always 1.000, (2) Reading and Writing have the strongest link (r = 0.955), and (3) Math correlates moderately with Reading (r = 0.818) and Writing (r = 0.803).",
            "limitations": "Only evaluates linear associations; strong non-linear relationships can show an r close to 0."
        },
        "count_plot": {
            "title": "Count Plot (Univariate Categorical Frequency)",
            "explanation": "A count plot shows the raw count and percentage of observations falling into each distinct category of a qualitative variable.",
            "when_to_use": "Use when checking category balance, group representation, or potential class imbalance in survey cohorts.",
            "what_to_observe": "Observe: (1) Sample representation (e.g. 518 female vs 482 male students), (2) Heavily populated groups vs underrepresented cohorts (e.g. Group A is smallest ethnic group).",
            "limitations": "Shows only sample counts; gives no information regarding student performance or grades."
        },
        "pie_chart": {
            "title": "Pie Chart (Proportional Composition)",
            "explanation": "A circular statistical graphic divided into slices illustrating numerical proportion where the arc length of each slice represents its share of 100%.",
            "when_to_use": "Use only when displaying proportional parts of a whole with very few categories (typically 2 to 4).",
            "what_to_observe": "Observe: The dominant slice (e.g. 64.2% of students completed 'none' for test prep course vs 35.8% 'completed').",
            "limitations": "Human vision struggles to compare subtle angle variations; poor for categories with similar shares or more than 5 slices."
        },
        "pairplot": {
            "title": "Pair Plot (Comprehensive Multivariate Exploration)",
            "explanation": "A pair plot constructs an N x N matrix of subplots displaying univariate distributions along the diagonal and bivariate scatter plots on all off-diagonal panels.",
            "when_to_use": "Use to inspect all pairs of continuous variables simultaneously, optionally color-coded by categorical factors like gender or lunch.",
            "what_to_observe": "Observe: (1) Elliptical cluster shapes showing strong linear relationships, (2) Diagonal KDE distributions, and (3) Group separation (e.g. females shifted right in reading/writing).",
            "limitations": "Computationally intensive; scales quadratically (N^2 plots) as the number of features increases."
        }
    }
    return explanations.get(chart_type, {
        "title": chart_type.replace('_', ' ').title(),
        "explanation": "Interactive Exploratory Data Analysis visualization.",
        "when_to_use": "Use to explore patterns and distributions in your dataset.",
        "what_to_observe": "Look for central tendencies, spread, and group variations.",
        "limitations": "Review sample size and data distribution assumptions."
    })


# ---------------------------------------------------------------------------
# 16. generate_matplotlib_code()
# ---------------------------------------------------------------------------
def generate_matplotlib_code(chart_type, **kwargs):
    """
    Returns clean, beginner-friendly Python Matplotlib code for the specified chart.
    """
    col = kwargs.get('column') or 'math score'
    xcol = kwargs.get('x_column') or 'lunch'
    ycol = kwargs.get('y_column') or 'math score'

    codes = {
        "histogram": f"""import matplotlib.pyplot as plt
import pandas as pd

# Load dataset
df = pd.read_csv("StudentsPerformance.csv")

# 1. Create figure and axis
plt.figure(figsize=(8, 5))

# 2. Plot histogram with explicit bins and borders
plt.hist(df['{col}'], bins=15, color='#3b82f6', edgecolor='white', alpha=0.7)

# 3. Add mean and median reference lines
mean_val = df['{col}'].mean()
median_val = df['{col}'].median()
plt.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {{mean_val:.1f}}')
plt.axvline(median_val, color='green', linestyle=':', linewidth=2, label=f'Median: {{median_val:.1f}}')

# 4. Configure labels, title, and grid
plt.title("Distribution of {col.title()} (Matplotlib)", fontsize=13, fontweight='bold')
plt.xlabel("{col.title()} (0 - 100)", fontsize=11)
plt.ylabel("Student Count", fontsize=11)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.4)

plt.tight_layout()
plt.show()""",

        "bar_chart": f"""import matplotlib.pyplot as plt
import pandas as pd

# Load dataset
df = pd.read_csv("StudentsPerformance.csv")

# 1. Calculate group means using pandas
means = df.groupby('{xcol}')['{ycol}'].mean().sort_values(ascending=False)

# 2. Plot bars manually
plt.figure(figsize=(8, 5))
bars = plt.bar(means.index, means.values, color='#3b82f6', edgecolor='#1e3a8a', width=0.55)

# 3. Annotate values on top of bars
for bar in bars:
    y_val = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, y_val + 1, f"{{y_val:.1f}}",
             ha='center', va='bottom', fontweight='bold')

plt.title("Average {ycol.title()} by {xcol.title()} (Matplotlib)", fontsize=13, fontweight='bold')
plt.xlabel("{xcol.title()}", fontsize=11)
plt.ylabel("Average {ycol.title()}", fontsize=11)
plt.ylim(0, 105)
plt.grid(axis='y', linestyle='--', alpha=0.4)

plt.tight_layout()
plt.show()""",

        "line_chart": f"""import matplotlib.pyplot as plt
import pandas as pd

# Load dataset
df = pd.read_csv("StudentsPerformance.csv")

# 1. Group by category and compute mean
means = df.groupby('{xcol}')['{ycol}'].mean()

# 2. Plot line with markers
plt.figure(figsize=(9, 5))
plt.plot(means.index, means.values, marker='o', color='#0284c7', linewidth=2.5, markersize=8)

# 3. Label each data point
for x, y in zip(means.index, means.values):
    plt.text(x, y + 0.8, f"{{y:.1f}}", ha='center', fontweight='bold', color='#0369a1')

plt.title("Score Progression (Matplotlib)", fontsize=13, fontweight='bold')
plt.xlabel("{xcol.title()}", fontsize=11)
plt.ylabel("Mean {ycol.title()}", fontsize=11)
plt.grid(True, linestyle='--', alpha=0.4)
plt.xticks(rotation=20)

plt.tight_layout()
plt.show()""",

        "box_plot": f"""import matplotlib.pyplot as plt
import pandas as pd

# Load dataset
df = pd.read_csv("StudentsPerformance.csv")

# 1. Prepare grouped data as a list of series
categories = df['{xcol}'].unique()
data_groups = [df[df['{xcol}'] == cat]['{ycol}'] for cat in categories]

# 2. Draw box plot
plt.figure(figsize=(8, 5))
plt.boxplot(data_groups, tick_labels=categories, patch_artist=True,
            boxprops=dict(facecolor='#93c5fd', color='#1e40af'))

plt.title("{ycol.title()} Distribution by {xcol.title()} (Matplotlib)", fontsize=13, fontweight='bold')
plt.xlabel("{xcol.title()}", fontsize=11)
plt.ylabel("{ycol.title()}", fontsize=11)
plt.grid(axis='y', linestyle='--', alpha=0.4)

plt.tight_layout()
plt.show()""",

        "scatter_plot": f"""import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Load dataset
df = pd.read_csv("StudentsPerformance.csv")

# 1. Create figure and scatter points
plt.figure(figsize=(8, 5))
plt.scatter(df['{xcol}'], df['{ycol}'], alpha=0.6, color='#2563eb', s=35)

# 2. Calculate and plot linear regression trendline
m, b = np.polyfit(df['{xcol}'], df['{ycol}'], 1)
x_vals = np.linspace(df['{xcol}'].min(), df['{xcol}'].max(), 100)
plt.plot(x_vals, m*x_vals + b, color='red', linestyle='--', linewidth=2, label=f'Trendline (slope: {{m:.2f}})')

r = df['{xcol}'].corr(df['{ycol}'])
plt.title(f"{ycol.title()} vs {xcol.title()} (Matplotlib | r = {{r:.2f}})", fontsize=13, fontweight='bold')
plt.xlabel("{xcol.title()}", fontsize=11)
plt.ylabel("{ycol.title()}", fontsize=11)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.4)

plt.tight_layout()
plt.show()""",

        "heatmap": """import matplotlib.pyplot as plt
import pandas as pd

# Load dataset and compute correlation matrix
df = pd.read_csv("StudentsPerformance.csv")
corr = df[['math score', 'reading score', 'writing score']].corr()

# 1. Plot using imshow
fig, ax = plt.subplots(figsize=(7, 5))
im = ax.imshow(corr, cmap='coolwarm', vmin=0.7, vmax=1.0)

# 2. Set tick labels and colorbar
labels = ['Math', 'Reading', 'Writing']
ax.set_xticks(range(len(labels)))
ax.set_yticks(range(len(labels)))
ax.set_xticklabels(labels)
ax.set_yticklabels(labels)
plt.colorbar(im, ax=ax, label='Pearson r')

# 3. Annotate cell values manually with loops
for i in range(len(labels)):
    for j in range(len(labels)):
        ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha='center', va='center', fontweight='bold')

plt.title("Correlation Heatmap (Matplotlib)", fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()""",

        "count_plot": f"""import matplotlib.pyplot as plt
import pandas as pd

# Load dataset
df = pd.read_csv("StudentsPerformance.csv")

# 1. Count occurrences using pandas value_counts()
counts = df['{col}'].value_counts()

# 2. Plot bar chart of counts
plt.figure(figsize=(8, 5))
bars = plt.bar(counts.index, counts.values, color='#4f46e5', width=0.55)

# 3. Annotate counts on top of bars
for bar in bars:
    h = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, h + 5, f"{{h}}", ha='center', fontweight='bold')

plt.title("Count of Students by {col.title()} (Matplotlib)", fontsize=13, fontweight='bold')
plt.xlabel("{col.title()}", fontsize=11)
plt.ylabel("Count", fontsize=11)
plt.grid(axis='y', linestyle='--', alpha=0.4)
plt.xticks(rotation=20)

plt.tight_layout()
plt.show()""",

        "pie_chart": f"""import matplotlib.pyplot as plt
import pandas as pd

# Load dataset
df = pd.read_csv("StudentsPerformance.csv")
counts = df['{col}'].value_counts()

# Plot pie chart with percentages
plt.figure(figsize=(6, 6))
plt.pie(counts.values, labels=counts.index, autopct='%1.1f%%', startangle=140,
        colors=['#38bdf8', '#34d399', '#f472b6', '#facc15', '#a78bfa'])

plt.title("Proportions of {col.title()} (Matplotlib)", fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()""",

        "pairplot": """# Matplotlib requires writing nested loops to generate a pair grid:
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("StudentsPerformance.csv")
cols = ['math score', 'reading score', 'writing score']

fig, axes = plt.subplots(3, 3, figsize=(8, 8))
for i, col_y in enumerate(cols):
    for j, col_x in enumerate(cols):
        ax = axes[i, j]
        if i == j:
            ax.hist(df[col_x], bins=15, color='#3b82f6', edgecolor='white')
        else:
            ax.scatter(df[col_x], df[col_y], alpha=0.4, s=15, color='#059669')
        if i == 2:
            ax.set_xlabel(col_x)
        if j == 0:
            ax.set_ylabel(col_y)

plt.suptitle("Pairwise Score Grid (Matplotlib)", y=1.02, fontweight='bold')
plt.tight_layout()
plt.show()"""
    }
    return codes.get(chart_type, "# Matplotlib code not available for this chart type.")


# ---------------------------------------------------------------------------
# 17. generate_seaborn_code()
# ---------------------------------------------------------------------------
def generate_seaborn_code(chart_type, **kwargs):
    """
    Returns clean, beginner-friendly Python Seaborn code for the specified chart.
    """
    col = kwargs.get('column') or 'math score'
    xcol = kwargs.get('x_column') or 'lunch'
    ycol = kwargs.get('y_column') or 'math score'
    hue = kwargs.get('hue') or 'gender'

    codes = {
        "histogram": f"""import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Load dataset
df = pd.read_csv("StudentsPerformance.csv")

# 1. Set clean visual theme
sns.set_theme(style="whitegrid")
plt.figure(figsize=(8, 5))

# 2. One line generates both bins and smooth KDE distribution curve!
sns.histplot(data=df, x='{col}', bins=15, kde=True, color='#2563eb')

plt.title("Distribution of {col.title()} with KDE (Seaborn)", fontsize=13, fontweight='bold')
plt.xlabel("{col.title()} (0 - 100)", fontsize=11)
plt.ylabel("Student Count", fontsize=11)

plt.tight_layout()
plt.show()""",

        "bar_chart": f"""import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Load dataset
df = pd.read_csv("StudentsPerformance.csv")

sns.set_theme(style="whitegrid")
plt.figure(figsize=(8, 5))

# Seaborn automatically calculates the mean and handles aggregation!
ax = sns.barplot(data=df, x='{xcol}', y='{ycol}', palette="Blues_d", errorbar=None)

# Optional: Add value labels above bars
for p in ax.patches:
    ax.annotate(f"{{p.get_height():.1f}}", (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='bottom', xytext=(0, 4), textcoords='offset points', fontweight='bold')

plt.title("Average {ycol.title()} by {xcol.title()} (Seaborn)", fontsize=13, fontweight='bold')
plt.xlabel("{xcol.title()}", fontsize=11)
plt.ylabel("Average {ycol.title()}", fontsize=11)
plt.ylim(0, 105)

plt.tight_layout()
plt.show()""",

        "line_chart": f"""import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Load dataset
df = pd.read_csv("StudentsPerformance.csv")

sns.set_theme(style="whitegrid")
plt.figure(figsize=(9, 5))

# Seaborn computes group means and connects ordered points with ease
sns.lineplot(data=df, x='{xcol}', y='{ycol}', marker='o', color='#0284c7', errorbar=None)

plt.title("Score Progression by {xcol.title()} (Seaborn)", fontsize=13, fontweight='bold')
plt.xlabel("{xcol.title()}", fontsize=11)
plt.ylabel("{ycol.title()}", fontsize=11)
plt.xticks(rotation=20)

plt.tight_layout()
plt.show()""",

        "box_plot": f"""import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Load dataset
df = pd.read_csv("StudentsPerformance.csv")

sns.set_theme(style="whitegrid")
plt.figure(figsize=(8, 5))

# Pass dataframe directly with x and y column names
sns.boxplot(data=df, x='{xcol}', y='{ycol}', palette="Set2")

plt.title("{ycol.title()} Box Plot by {xcol.title()} (Seaborn)", fontsize=13, fontweight='bold')
plt.xlabel("{xcol.title()}", fontsize=11)
plt.ylabel("{ycol.title()}", fontsize=11)

plt.tight_layout()
plt.show()""",

        "scatter_plot": f"""import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Load dataset
df = pd.read_csv("StudentsPerformance.csv")

sns.set_theme(style="whitegrid")
plt.figure(figsize=(8, 5))

# One line handles scatter plotting, hue coloring, and automatic legend!
sns.scatterplot(data=df, x='{xcol}', y='{ycol}', hue='{hue}', alpha=0.7, s=40, palette='tab10')

plt.title("{ycol.title()} vs {xcol.title()} Colored by {hue.title()} (Seaborn)", fontsize=13, fontweight='bold')
plt.xlabel("{xcol.title()}", fontsize=11)
plt.ylabel("{ycol.title()}", fontsize=11)

plt.tight_layout()
plt.show()""",

        "heatmap": """import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Load dataset and compute correlation matrix
df = pd.read_csv("StudentsPerformance.csv")
corr = df[['math score', 'reading score', 'writing score']].corr()

plt.figure(figsize=(7, 5))

# Seaborn automatically formats text, annotations, colors, and colorbar in ONE call!
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".3f", vmin=0.7, vmax=1.0, linewidths=1.5)

plt.title("Correlation Heatmap (Seaborn)", fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()""",

        "count_plot": f"""import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Load dataset
df = pd.read_csv("StudentsPerformance.csv")

sns.set_theme(style="whitegrid")
plt.figure(figsize=(8, 5))

# Count plot directly tallies categorical values without manual groupby/value_counts!
sns.countplot(data=df, x='{col}', palette="mako")

plt.title("Student Counts by {col.title()} (Seaborn)", fontsize=13, fontweight='bold')
plt.xlabel("{col.title()}", fontsize=11)
plt.ylabel("Count", fontsize=11)
plt.xticks(rotation=20)

plt.tight_layout()
plt.show()""",

        "pie_chart": """# Note for Students:
# Seaborn intentionally DOES NOT have a `pieplot()` function!
# Seaborn's philosophy is rooted in statistical, Cartesian coordinate visualization.
# For circular proportional charts, data scientists use Matplotlib directly:
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("StudentsPerformance.csv")
counts = df['test preparation course'].value_counts()

plt.figure(figsize=(6, 6))
plt.pie(counts, labels=counts.index, autopct='%1.1f%%', colors=['#38bdf8', '#34d399'])
plt.title("Proportion of Students (Matplotlib)", fontsize=13, fontweight='bold')
plt.show()""",

        "pairplot": """import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Load dataset
df = pd.read_csv("StudentsPerformance.csv")

# Seaborn creates an entire 3x3 grid with distributions and scatter plots in ONE line!
g = sns.pairplot(
    df[['math score', 'reading score', 'writing score', 'gender']],
    hue='gender',
    palette='Set2'
)
g.fig.subplots_adjust(top=0.92)
g.fig.suptitle("Pairwise Score Grid by Gender (Seaborn)", fontweight='bold')

plt.show()"""
    }
    return codes.get(chart_type, "# Seaborn code not available for this chart type.")


# ---------------------------------------------------------------------------
# 18. create_dashboard_html()
# ---------------------------------------------------------------------------
def create_dashboard_html():
    """
    Returns the comprehensive, clean, single-page HTML interface
    with embedded CSS and vanilla JavaScript.
    No external JavaScript frameworks are used.
    """
    return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Interactive EDA Classroom Dashboard - Students' Performance</title>
    <style>
        :root {
            --primary: #2563eb;
            --primary-dark: #1d4ed8;
            --primary-light: #eff6ff;
            --secondary: #0f172a;
            --slate: #475569;
            --bg-body: #f8fafc;
            --bg-card: #ffffff;
            --border: #e2e8f0;
            --success: #16a34a;
            --warning: #d97706;
            --danger: #dc2626;
            --radius: 8px;
            --shadow: 0 1px 3px rgba(0,0,0,0.08), 0 1px 2px rgba(0,0,0,0.04);
            --shadow-md: 0 4px 6px -1px rgba(0,0,0,0.1), 0 2px 4px -1px rgba(0,0,0,0.06);
        }

        * { box-sizing: border-box; margin: 0; padding: 0; }

        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background-color: var(--bg-body);
            color: var(--secondary);
            line-height: 1.5;
            padding-bottom: 60px;
        }

        /* Top Header */
        header {
            background-color: #0f172a;
            color: #ffffff;
            padding: 1.25rem 2rem;
            border-bottom: 3px solid var(--primary);
        }
        .header-content {
            max-width: 1300px;
            margin: 0 auto;
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 1rem;
        }
        .header-title h1 {
            font-size: 1.45rem;
            font-weight: 700;
            letter-spacing: -0.02em;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        .header-title p {
            font-size: 0.88rem;
            color: #94a3b8;
            margin-top: 0.2rem;
        }
        .header-badges {
            display: flex;
            gap: 0.5rem;
            align-items: center;
        }
        .badge {
            font-size: 0.75rem;
            padding: 0.25rem 0.65rem;
            border-radius: 9999px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        .badge-blue { background-color: #1e3a8a; color: #93c5fd; }
        .badge-green { background-color: #064e3b; color: #6ee7b7; }

        /* Container */
        .container {
            max-width: 1300px;
            margin: 1.5rem auto;
            padding: 0 1.25rem;
        }

        /* Missing Dataset Alert */
        .alert-banner {
            background-color: #fef2f2;
            border: 1px solid #f87171;
            border-left: 5px solid var(--danger);
            padding: 1rem 1.25rem;
            border-radius: var(--radius);
            margin-bottom: 1.5rem;
            display: flex;
            align-items: flex-start;
            gap: 0.75rem;
        }
        .alert-banner h3 { color: #991b1b; font-size: 1rem; font-weight: 700; }
        .alert-banner p { color: #7f1d1d; font-size: 0.88rem; margin-top: 0.25rem; }

        /* Navigation Tabs */
        .nav-tabs {
            display: flex;
            gap: 0.35rem;
            background-color: #ffffff;
            padding: 0.4rem;
            border-radius: var(--radius);
            border: 1px solid var(--border);
            margin-bottom: 1.5rem;
            overflow-x: auto;
            box-shadow: var(--shadow);
        }
        .tab-btn {
            background: none;
            border: none;
            padding: 0.65rem 1.15rem;
            font-size: 0.9rem;
            font-weight: 600;
            color: var(--slate);
            border-radius: 6px;
            cursor: pointer;
            white-space: nowrap;
            transition: all 0.15s ease-in-out;
            display: flex;
            align-items: center;
            gap: 0.4rem;
        }
        .tab-btn:hover { background-color: var(--primary-light); color: var(--primary); }
        .tab-btn.active {
            background-color: var(--primary);
            color: #ffffff;
            box-shadow: 0 1px 2px rgba(37,99,235,0.3);
        }

        /* Tab Content Panels */
        .tab-pane { display: none; }
        .tab-pane.active { display: block; }

        /* Cards */
        .card {
            background: var(--bg-card);
            border-radius: var(--radius);
            border: 1px solid var(--border);
            box-shadow: var(--shadow);
            padding: 1.25rem;
            margin-bottom: 1.5rem;
        }
        .card-header {
            margin-bottom: 1rem;
            border-bottom: 1px solid var(--border);
            padding-bottom: 0.65rem;
        }
        .card-title {
            font-size: 1.12rem;
            font-weight: 700;
            color: var(--secondary);
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        .card-subtitle {
            font-size: 0.84rem;
            color: var(--slate);
            margin-top: 0.15rem;
        }

        /* Teacher Concept Banner */
        .concept-banner {
            background-color: #f0fdf4;
            border-left: 4px solid var(--success);
            padding: 1rem 1.25rem;
            border-radius: 4px;
            margin-bottom: 1.25rem;
            font-size: 0.9rem;
            color: #166534;
        }
        .concept-banner strong { color: #14532d; font-size: 0.95rem; }

        /* Metric Cards Grid */
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
            gap: 1rem;
            margin-bottom: 1.5rem;
        }
        .stat-card {
            background: #ffffff;
            border: 1px solid var(--border);
            border-radius: var(--radius);
            padding: 1rem;
            box-shadow: var(--shadow);
            text-align: center;
            border-top: 3px solid var(--primary);
        }
        .stat-label {
            font-size: 0.78rem;
            color: var(--slate);
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }
        .stat-val {
            font-size: 1.65rem;
            font-weight: 800;
            color: var(--secondary);
            margin-top: 0.25rem;
        }
        .stat-sub {
            font-size: 0.75rem;
            color: #64748b;
            margin-top: 0.2rem;
        }

        /* Interactive Grid (Controls on Left, Display on Right) */
        .interactive-layout {
            display: grid;
            grid-template-columns: 320px 1fr;
            gap: 1.25rem;
            align-items: start;
        }
        @media (max-width: 900px) {
            .interactive-layout { grid-template-columns: 1fr; }
        }

        /* Control Form Elements */
        .control-group {
            margin-bottom: 1.15rem;
        }
        .control-label {
            display: block;
            font-size: 0.85rem;
            font-weight: 700;
            color: var(--secondary);
            margin-bottom: 0.35rem;
        }
        .control-select, .control-btn {
            width: 100%;
            padding: 0.55rem 0.75rem;
            border-radius: 6px;
            border: 1px solid var(--border);
            font-size: 0.9rem;
            background-color: #ffffff;
            color: var(--secondary);
        }
        .control-select:focus {
            outline: none;
            border-color: var(--primary);
            box-shadow: 0 0 0 2px rgba(37,99,235,0.2);
        }

        .btn-group {
            display: flex;
            flex-direction: column;
            gap: 0.45rem;
        }
        .btn-option {
            background-color: #f8fafc;
            border: 1px solid var(--border);
            padding: 0.55rem 0.85rem;
            border-radius: 6px;
            font-size: 0.88rem;
            font-weight: 600;
            text-align: left;
            cursor: pointer;
            color: var(--slate);
            transition: all 0.15s ease;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }
        .btn-option:hover {
            background-color: var(--primary-light);
            color: var(--primary);
            border-color: #bfdbfe;
        }
        .btn-option.selected {
            background-color: #eff6ff;
            color: var(--primary);
            border-color: var(--primary);
            font-weight: 700;
        }

        /* Chart Display Canvas Card */
        .chart-display-card {
            background: #ffffff;
            border: 1px solid var(--border);
            border-radius: var(--radius);
            box-shadow: var(--shadow);
            padding: 1.25rem;
            text-align: center;
            min-height: 400px;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
        }
        .chart-img {
            max-width: 100%;
            height: auto;
            border-radius: 6px;
            box-shadow: 0 1px 2px rgba(0,0,0,0.05);
        }

        /* Educational Information Cards */
        .info-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 1rem;
            margin-top: 1.25rem;
        }
        .info-card {
            background: #ffffff;
            border: 1px solid var(--border);
            border-radius: var(--radius);
            padding: 1rem;
            font-size: 0.88rem;
        }
        .info-card.observation {
            background-color: #f8fafc;
            border-left: 4px solid var(--primary);
        }
        .info-card.when-to-use {
            background-color: #f8fafc;
            border-left: 4px solid var(--success);
        }
        .info-card.limitations {
            background-color: #f8fafc;
            border-left: 4px solid var(--warning);
        }
        .info-card h4 {
            font-size: 0.88rem;
            font-weight: 700;
            margin-bottom: 0.4rem;
            display: flex;
            align-items: center;
            gap: 0.35rem;
        }

        /* Code Viewer Tabs */
        .code-container {
            margin-top: 1.25rem;
            border: 1px solid var(--border);
            border-radius: var(--radius);
            overflow: hidden;
        }
        .code-header {
            background-color: #1e293b;
            color: #ffffff;
            padding: 0.5rem 0.85rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 0.82rem;
            font-weight: 600;
        }
        .code-tabs {
            display: flex;
            gap: 0.35rem;
        }
        .code-tab-btn {
            background: none;
            border: none;
            color: #94a3b8;
            padding: 0.3rem 0.75rem;
            font-size: 0.8rem;
            font-weight: 600;
            border-radius: 4px;
            cursor: pointer;
        }
        .code-tab-btn.active {
            background-color: var(--primary);
            color: #ffffff;
        }
        .copy-btn {
            background: rgba(255,255,255,0.15);
            border: none;
            color: #ffffff;
            padding: 0.25rem 0.6rem;
            font-size: 0.75rem;
            border-radius: 4px;
            cursor: pointer;
        }
        .copy-btn:hover { background: rgba(255,255,255,0.25); }
        pre.code-block {
            background-color: #0f172a;
            color: #f8fafc;
            padding: 1rem;
            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
            font-size: 0.82rem;
            line-height: 1.55;
            overflow-x: auto;
            margin: 0;
            white-space: pre;
        }

        /* Table Styles */
        .table-responsive {
            overflow-x: auto;
            margin-top: 0.5rem;
        }
        table.data-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.85rem;
            text-align: left;
        }
        table.data-table th, table.data-table td {
            padding: 0.65rem 0.85rem;
            border-bottom: 1px solid var(--border);
        }
        table.data-table th {
            background-color: #f8fafc;
            font-weight: 700;
            color: var(--secondary);
            border-top: 1px solid var(--border);
        }
        table.data-table tr:hover {
            background-color: #f1f5f9;
        }

        /* Comparison Table */
        table.comp-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.86rem;
        }
        table.comp-table th, table.comp-table td {
            padding: 0.75rem 0.85rem;
            border: 1px solid var(--border);
            vertical-align: top;
        }
        table.comp-table th {
            background-color: #f1f5f9;
            font-weight: 700;
        }
        table.comp-table tr:nth-child(even) { background-color: #f8fafc; }

        /* Side-by-Side Code Comparison Grid */
        .side-by-side-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1rem;
            margin-top: 1rem;
        }
        @media (max-width: 850px) {
            .side-by-side-grid { grid-template-columns: 1fr; }
        }

        /* Spinner */
        .spinner {
            border: 3px solid rgba(0, 0, 0, 0.1);
            width: 36px;
            height: 36px;
            border-radius: 50%;
            border-left-color: var(--primary);
            animation: spin 0.8s linear infinite;
            margin-bottom: 0.75rem;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
    </style>
</head>
<body>

    <!-- Header -->
    <header>
        <div class="header-content">
            <div class="header-title">
                <h1>🎓 Single-File Interactive EDA Dashboard</h1>
                <p>Teacher's Exploratory Data Analysis Environment &bull; Students' Performance in Exams</p>
            </div>
            <div class="header-badges">
                <span class="badge badge-blue">Flask + Pandas</span>
                <span class="badge badge-green">Matplotlib & Seaborn</span>
            </div>
        </div>
    </header>

    <div class="container">

        <!-- Missing Dataset Alert -->
        <div id="missing-alert" class="alert-banner" style="display: none;">
            <div style="font-size: 1.5rem;">⚠️</div>
            <div>
                <h3>Dataset Missing: StudentsPerformance.csv Not Found</h3>
                <p id="missing-msg">Please place <strong>StudentsPerformance.csv</strong> in the same directory as <strong>eda_dashboard.py</strong> and reload this page.</p>
            </div>
        </div>

        <!-- Navigation Tabs -->
        <nav class="nav-tabs">
            <button class="tab-btn active" onclick="switchTab('overview')">📊 1. Overview</button>
            <button class="tab-btn" onclick="switchTab('univariate')">📈 2. Univariate Analysis</button>
            <button class="tab-btn" onclick="switchTab('bivariate')">📉 3. Bivariate Analysis</button>
            <button class="tab-btn" onclick="switchTab('multivariate')">🧬 4. Multivariate Analysis</button>
            <button class="tab-btn" onclick="switchTab('mpl-vs-sns')">⚡ 5. Matplotlib vs Seaborn</button>
            <button class="tab-btn" onclick="switchTab('cheatsheet')">📖 6. Graph Comparison</button>
        </nav>

        <!-- ============================================================= -->
        <!-- TAB 1: OVERVIEW -->
        <!-- ============================================================= -->
        <div id="tab-overview" class="tab-pane active">
            
            <!-- What is EDA concept box -->
            <div class="concept-banner">
                <strong>💡 Classroom Lesson: What is Exploratory Data Analysis (EDA)?</strong>
                <p style="margin-top: 0.25rem;">
                    Exploratory Data Analysis (EDA) is the foundational data science workflow of investigating datasets to discover patterns, spot anomalies, test hypotheses, and verify assumptions with summary statistics and visualizations before building any machine learning models.
                </p>
            </div>

            <!-- Stats Metric Cards -->
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-label">Total Students</div>
                    <div id="stat-students" class="stat-val">--</div>
                    <div class="stat-sub">Sample size (Rows)</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Total Columns</div>
                    <div id="stat-cols" class="stat-val">--</div>
                    <div class="stat-sub">5 Categorical, 3 Numeric</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Avg Math Score</div>
                    <div id="stat-math" class="stat-val">--</div>
                    <div class="stat-sub">Range: 0 - 100</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Avg Reading Score</div>
                    <div id="stat-reading" class="stat-val">--</div>
                    <div class="stat-sub">Range: 17 - 100</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Avg Writing Score</div>
                    <div id="stat-writing" class="stat-val">--</div>
                    <div class="stat-sub">Range: 10 - 100</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Missing Values</div>
                    <div id="stat-missing" class="stat-val" style="color: var(--success);">0</div>
                    <div class="stat-sub">Clean dataset</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Duplicate Rows</div>
                    <div id="stat-duplicates" class="stat-val" style="color: var(--success);">0</div>
                    <div class="stat-sub">Unique records</div>
                </div>
            </div>

            <!-- Score Five-Number Summary Table -->
            <div class="card">
                <div class="card-header">
                    <div class="card-title">📐 Score Descriptive Statistics (5-Number Summary)</div>
                    <div class="card-subtitle">Measures of central tendency (mean, median) and dispersion (std, IQR) for exam subjects.</div>
                </div>
                <div class="table-responsive">
                    <table class="data-table" id="stats-table">
                        <thead>
                            <tr>
                                <th>Subject Feature</th>
                                <th>Count</th>
                                <th>Mean</th>
                                <th>Std Dev</th>
                                <th>Min</th>
                                <th>25% (Q1)</th>
                                <th>50% (Median)</th>
                                <th>75% (Q3)</th>
                                <th>Max</th>
                            </tr>
                        </thead>
                        <tbody id="stats-tbody">
                            <tr><td colspan="9" style="text-align: center;">Loading summary statistics...</td></tr>
                        </tbody>
                    </table>
                </div>
            </div>

            <!-- Dataset Preview Table -->
            <div class="card">
                <div class="card-header">
                    <div class="card-title">📋 Raw Dataset Preview (First 10 Observations)</div>
                    <div class="card-subtitle">Inspect raw values, feature types, and casing from StudentsPerformance.csv.</div>
                </div>
                <div class="table-responsive">
                    <table class="data-table" id="preview-table">
                        <thead id="preview-thead"></thead>
                        <tbody id="preview-tbody">
                            <tr><td colspan="8" style="text-align: center;">Loading dataset preview...</td></tr>
                        </tbody>
                    </table>
                </div>
            </div>

        </div>

        <!-- ============================================================= -->
        <!-- TAB 2: UNIVARIATE ANALYSIS -->
        <!-- ============================================================= -->
        <div id="tab-univariate" class="tab-pane">
            <div class="concept-banner">
                <strong>📈 Classroom Lesson: Univariate Analysis</strong>
                <p style="margin-top: 0.25rem;">
                    Univariate analysis inspects ONE variable at a time. For numerical variables, we examine distribution shape, central tendency, and outliers (Histogram, Box Plot). For categorical variables, we examine frequency counts and proportions (Count Plot, Pie Chart).
                </p>
            </div>

            <div class="interactive-layout">
                <!-- Controls -->
                <div class="card">
                    <div class="card-header">
                        <div class="card-title">⚙️ Variable Selector</div>
                        <div class="card-subtitle">Choose a column to explore</div>
                    </div>

                    <div class="control-group">
                        <label class="control-label" for="uni-col-select">Select Column:</label>
                        <select id="uni-col-select" class="control-select" onchange="onUnivariateColChange()">
                            <optgroup label="Numerical Variables (Scores)">
                                <option value="math score" selected>math score</option>
                                <option value="reading score">reading score</option>
                                <option value="writing score">writing score</option>
                            </optgroup>
                            <optgroup label="Categorical Variables">
                                <option value="gender">gender</option>
                                <option value="race/ethnicity">race/ethnicity</option>
                                <option value="parental level of education">parental level of education</option>
                                <option value="lunch">lunch</option>
                                <option value="test preparation course">test preparation course</option>
                            </optgroup>
                        </select>
                    </div>

                    <div class="control-group">
                        <label class="control-label">Select Graph Type:</label>
                        <div id="uni-chart-btns" class="btn-group">
                            <!-- Populated dynamically based on data type -->
                        </div>
                    </div>
                </div>

                <!-- Display Area -->
                <div>
                    <div class="chart-display-card" id="uni-chart-container">
                        <div class="spinner"></div>
                        <p style="color: var(--slate); font-size: 0.9rem;">Rendering interactive chart...</p>
                    </div>

                    <!-- Explanations & Observation Cards -->
                    <div class="info-grid">
                        <div class="info-card when-to-use">
                            <h4>🎯 When to Use This Graph</h4>
                            <p id="uni-when-to-use">Loading practical guidelines...</p>
                        </div>
                        <div class="info-card observation">
                            <h4>🔍 Teacher's Observation Guide</h4>
                            <p id="uni-what-to-observe">Loading key patterns to observe...</p>
                        </div>
                        <div class="info-card limitations">
                            <h4>⚠️ Limitations & Pitfalls</h4>
                            <p id="uni-limitations">Loading caveats...</p>
                        </div>
                    </div>

                    <!-- Code Box with Tabs -->
                    <div class="code-container">
                        <div class="code-header">
                            <div class="code-tabs">
                                <button class="code-tab-btn active" onclick="switchCodeTab('uni', 'mpl')">Matplotlib Code</button>
                                <button class="code-tab-btn" onclick="switchCodeTab('uni', 'sns')">Seaborn Code</button>
                            </div>
                            <button class="copy-btn" onclick="copyCode('uni')">📋 Copy Code</button>
                        </div>
                        <pre class="code-block" id="uni-code-mpl"># Loading Matplotlib Code...</pre>
                        <pre class="code-block" id="uni-code-sns" style="display: none;"># Loading Seaborn Code...</pre>
                    </div>
                </div>
            </div>
        </div>

        <!-- ============================================================= -->
        <!-- TAB 3: BIVARIATE ANALYSIS -->
        <!-- ============================================================= -->
        <div id="tab-bivariate" class="tab-pane">
            <div class="concept-banner">
                <strong>📉 Classroom Lesson: Bivariate Analysis</strong>
                <p style="margin-top: 0.25rem;">
                    Bivariate analysis examines relationships between TWO variables. It determines whether changes in one variable correspond with changes in another. Examples: Bar chart of average math score by lunch type, scatter plot of reading vs writing score, or boxplot of scores by gender.
                </p>
            </div>

            <div class="interactive-layout">
                <!-- Controls -->
                <div class="card">
                    <div class="card-header">
                        <div class="card-title">⚙️ Pair Controls</div>
                        <div class="card-subtitle">Configure X and Y variables</div>
                    </div>

                    <div class="control-group">
                        <label class="control-label" for="bi-chart-select">Bivariate Graph Type:</label>
                        <select id="bi-chart-select" class="control-select" onchange="onBivariateTypeChange()">
                            <option value="bar_chart" selected>Bar Chart (Average Score by Category)</option>
                            <option value="scatter_plot">Scatter Plot (Numeric vs Numeric + Trendline)</option>
                            <option value="box_plot">Box Plot by Category (Compare Distributions)</option>
                            <option value="line_chart">Line Chart (Ordered Progression Trend)</option>
                        </select>
                    </div>

                    <div class="control-group">
                        <label class="control-label" for="bi-x-select">X Variable:</label>
                        <select id="bi-x-select" class="control-select" onchange="loadBivariateChart()">
                            <!-- Populated dynamically -->
                        </select>
                    </div>

                    <div class="control-group">
                        <label class="control-label" for="bi-y-select">Y Variable (Numerical Score):</label>
                        <select id="bi-y-select" class="control-select" onchange="loadBivariateChart()">
                            <option value="math score" selected>math score</option>
                            <option value="reading score">reading score</option>
                            <option value="writing score">writing score</option>
                        </select>
                    </div>

                    <div class="control-group" id="bi-hue-group" style="display: none;">
                        <label class="control-label" for="bi-hue-select">Optional Grouping (Hue):</label>
                        <select id="bi-hue-select" class="control-select" onchange="loadBivariateChart()">
                            <option value="">None (Single Color)</option>
                            <option value="gender">gender</option>
                            <option value="lunch">lunch</option>
                            <option value="test preparation course">test preparation course</option>
                        </select>
                    </div>
                </div>

                <!-- Display Area -->
                <div>
                    <div class="chart-display-card" id="bi-chart-container">
                        <div class="spinner"></div>
                        <p style="color: var(--slate); font-size: 0.9rem;">Rendering bivariate chart...</p>
                    </div>

                    <div class="info-grid">
                        <div class="info-card when-to-use">
                            <h4>🎯 When to Use This Graph</h4>
                            <p id="bi-when-to-use">Loading practical guidelines...</p>
                        </div>
                        <div class="info-card observation">
                            <h4>🔍 Teacher's Observation Guide</h4>
                            <p id="bi-what-to-observe">Loading key patterns to observe...</p>
                        </div>
                        <div class="info-card limitations">
                            <h4>⚠️ Limitations & Pitfalls</h4>
                            <p id="bi-limitations">Loading caveats...</p>
                        </div>
                    </div>

                    <!-- Code Box with Tabs -->
                    <div class="code-container">
                        <div class="code-header">
                            <div class="code-tabs">
                                <button class="code-tab-btn active" onclick="switchCodeTab('bi', 'mpl')">Matplotlib Code</button>
                                <button class="code-tab-btn" onclick="switchCodeTab('bi', 'sns')">Seaborn Code</button>
                            </div>
                            <button class="copy-btn" onclick="copyCode('bi')">📋 Copy Code</button>
                        </div>
                        <pre class="code-block" id="bi-code-mpl"># Loading Matplotlib Code...</pre>
                        <pre class="code-block" id="bi-code-sns" style="display: none;"># Loading Seaborn Code...</pre>
                    </div>
                </div>
            </div>
        </div>

        <!-- ============================================================= -->
        <!-- TAB 4: MULTIVARIATE ANALYSIS -->
        <!-- ============================================================= -->
        <div id="tab-multivariate" class="tab-pane">
            <div class="concept-banner">
                <strong>🧬 Classroom Lesson: Multivariate Analysis</strong>
                <p style="margin-top: 0.25rem;">
                    Multivariate analysis evaluates interactions among THREE OR MORE variables at once. In student performance, scores in reading, writing, and math are heavily interconnected and influenced by demographics and preparation courses.
                </p>
            </div>

            <div class="interactive-layout">
                <!-- Controls -->
                <div class="card">
                    <div class="card-header">
                        <div class="card-title">⚙️ Multivariate Presets</div>
                        <div class="card-subtitle">Choose a multi-variable technique</div>
                    </div>

                    <div class="btn-group">
                        <button class="btn-option selected" id="multi-btn-heatmap" onclick="loadMultivariate('heatmap')">
                            <span>🌡️ Correlation Heatmap</span>
                            <span style="font-size: 0.75rem;">Matrix</span>
                        </button>
                        <button class="btn-option" id="multi-btn-pairplot" onclick="loadMultivariate('pairplot')">
                            <span>🧩 Pair Plot (All Subjects)</span>
                            <span style="font-size: 0.75rem;">3x3 Grid</span>
                        </button>
                    </div>

                    <div class="control-group" id="multi-hue-group" style="margin-top: 1.25rem; display: none;">
                        <label class="control-label" for="multi-hue-select">Color Pair Plot by (Hue):</label>
                        <select id="multi-hue-select" class="control-select" onchange="loadMultivariate('pairplot')">
                            <option value="">None (Uniform Color)</option>
                            <option value="gender" selected>gender</option>
                            <option value="lunch">lunch</option>
                            <option value="test preparation course">test preparation course</option>
                        </select>
                    </div>
                </div>

                <!-- Display Area -->
                <div>
                    <div class="chart-display-card" id="multi-chart-container">
                        <div class="spinner"></div>
                        <p style="color: var(--slate); font-size: 0.9rem;">Rendering multivariate analysis...</p>
                    </div>

                    <div class="info-grid">
                        <div class="info-card when-to-use">
                            <h4>🎯 When to Use This Graph</h4>
                            <p id="multi-when-to-use">Loading practical guidelines...</p>
                        </div>
                        <div class="info-card observation">
                            <h4>🔍 Teacher's Observation Guide</h4>
                            <p id="multi-what-to-observe">Loading key patterns to observe...</p>
                        </div>
                        <div class="info-card limitations">
                            <h4>⚠️ Limitations & Pitfalls</h4>
                            <p id="multi-limitations">Loading caveats...</p>
                        </div>
                    </div>

                    <!-- Code Box with Tabs -->
                    <div class="code-container">
                        <div class="code-header">
                            <div class="code-tabs">
                                <button class="code-tab-btn active" onclick="switchCodeTab('multi', 'mpl')">Matplotlib Code</button>
                                <button class="code-tab-btn" onclick="switchCodeTab('multi', 'sns')">Seaborn Code</button>
                            </div>
                            <button class="copy-btn" onclick="copyCode('multi')">📋 Copy Code</button>
                        </div>
                        <pre class="code-block" id="multi-code-mpl"># Loading Matplotlib Code...</pre>
                        <pre class="code-block" id="multi-code-sns" style="display: none;"># Loading Seaborn Code...</pre>
                    </div>
                </div>
            </div>
        </div>

        <!-- ============================================================= -->
        <!-- TAB 5: MATPLOTLIB VS SEABORN -->
        <!-- ============================================================= -->
        <div id="tab-mpl-vs-sns" class="tab-pane">
            
            <div class="concept-banner">
                <strong>⚡ Classroom Lesson: Matplotlib vs Seaborn Comparison</strong>
                <p style="margin-top: 0.25rem;">
                    <strong>Matplotlib</strong> is the low-level, imperative engine: you control every line, axis, and tick, but you must manually compute group means and format legends. 
                    <strong>Seaborn</strong> is the high-level, declarative statistical wrapper built on Matplotlib: it accepts pandas DataFrames directly, automates statistical estimation (like KDE curves, error bars, and correlations), and provides attractive defaults with concise one-line syntax.
                </p>
            </div>

            <!-- Chart Switcher Buttons -->
            <div class="card">
                <div class="card-header">
                    <div class="card-title">🔍 Side-by-Side Visual & Code Comparison</div>
                    <div class="card-subtitle">Select a chart type to inspect both implementations side-by-side</div>
                </div>
                <div style="display: flex; gap: 0.5rem; flex-wrap: wrap;">
                    <button class="tab-btn active" id="comp-btn-histogram" onclick="loadComparison('histogram')">📊 1. Histogram</button>
                    <button class="tab-btn" id="comp-btn-bar_chart" onclick="loadComparison('bar_chart')">📊 2. Bar Chart</button>
                    <button class="tab-btn" id="comp-btn-box_plot" onclick="loadComparison('box_plot')">📦 3. Box Plot</button>
                    <button class="tab-btn" id="comp-btn-scatter_plot" onclick="loadComparison('scatter_plot')">🎯 4. Scatter Plot</button>
                    <button class="tab-btn" id="comp-btn-heatmap" onclick="loadComparison('heatmap')">🌡️ 5. Heatmap</button>
                </div>
            </div>

            <!-- Side-by-side Image Container -->
            <div class="chart-display-card" id="comp-chart-container" style="min-height: 380px;">
                <div class="spinner"></div>
                <p style="color: var(--slate); font-size: 0.9rem;">Rendering side-by-side comparison...</p>
            </div>

            <!-- Side-by-Side Code Boxes -->
            <div class="side-by-side-grid">
                <div class="code-container" style="margin-top: 0;">
                    <div class="code-header">
                        <span>🐍 Matplotlib Implementation (Imperative)</span>
                        <button class="copy-btn" onclick="copyRawCode('comp-code-mpl')">📋 Copy</button>
                    </div>
                    <pre class="code-block" id="comp-code-mpl"># Loading Matplotlib Code...</pre>
                </div>
                <div class="code-container" style="margin-top: 0;">
                    <div class="code-header">
                        <span>✨ Seaborn Implementation (Declarative)</span>
                        <button class="copy-btn" onclick="copyRawCode('comp-code-sns')">📋 Copy</button>
                    </div>
                    <pre class="code-block" id="comp-code-sns"># Loading Seaborn Code...</pre>
                </div>
            </div>

            <!-- Teacher's Philosophy Breakdown -->
            <div class="card" style="margin-top: 1.5rem;">
                <div class="card-header">
                    <div class="card-title">📖 Core Philosophical Differences Explained for Beginners</div>
                </div>
                <div class="table-responsive">
                    <table class="comp-table">
                        <thead>
                            <tr>
                                <th>Comparison Dimension</th>
                                <th>Matplotlib</th>
                                <th>Seaborn</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr>
                                <td><strong>Abstraction Level</strong></td>
                                <td>Low-level. Full granular control over every visual element (canvas, axes, spines, ticks).</td>
                                <td>High-level. Focuses on statistical visualization with minimal boilerplate code.</td>
                            </tr>
                            <tr>
                                <td><strong>Data Structure Handling</strong></td>
                                <td>Expects raw arrays, NumPy arrays, or Python lists. Manual grouping with pandas required.</td>
                                <td>Directly accepts tidy pandas DataFrames using column names as strings (<code>data=df, x='col'</code>).</td>
                            </tr>
                            <tr>
                                <td><strong>Statistical Calculation</strong></td>
                                <td>Manual. You must calculate means, standard deviations, and regression trendlines yourself.</td>
                                <td>Automatic. Natively computes KDE curves, error bars, confidence intervals, and linear regressions.</td>
                            </tr>
                            <tr>
                                <td><strong>Default Aesthetics</strong></td>
                                <td>Basic, functional, classic appearance with plain white background.</td>
                                <td>Sophisticated, publication-ready themes (<code>whitegrid</code>, <code>darkgrid</code>) and harmonious color palettes.</td>
                            </tr>
                            <tr>
                                <td><strong>When to Choose</strong></td>
                                <td>Custom application layouts, fine-tuning subplots, non-standard plots, or embedding in GUI apps.</td>
                                <td>Fast exploratory data analysis (EDA), statistical relationships, hue groupings, and pair grids.</td>
                            </tr>
                        </tbody>
                    </table>
                </div>
            </div>

        </div>

        <!-- ============================================================= -->
        <!-- TAB 6: GRAPH COMPARISON GUIDE -->
        <!-- ============================================================= -->
        <div id="tab-cheatsheet" class="tab-pane">
            <div class="concept-banner">
                <strong>📖 Classroom Cheatsheet: The Complete EDA Graph Comparison Guide</strong>
                <p style="margin-top: 0.25rem;">
                    Use this teacher's reference table to select the correct chart type based on research questions, variable types, and analytical goals.
                </p>
            </div>

            <div class="card">
                <div class="card-header">
                    <div class="card-title">📊 EDA Graph Selection Matrix</div>
                    <div class="card-subtitle">Comprehensive guide covering purpose, data types, and limitations for all 9 core EDA graphs.</div>
                </div>
                <div class="table-responsive">
                    <table class="comp-table">
                        <thead>
                            <tr>
                                <th>Graph Name</th>
                                <th>Primary Purpose</th>
                                <th>When to Use</th>
                                <th>Example Question (Students Dataset)</th>
                                <th>Required Data Types</th>
                                <th>Limitations & Caveats</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr>
                                <td><strong>Histogram</strong></td>
                                <td>Examine distribution shape, frequency density, and skewness of a single continuous feature.</td>
                                <td>First step in univariate analysis to check normality, central mode, and tail spread.</td>
                                <td><em>"Are student math scores normally distributed or left-skewed?"</em></td>
                                <td>1 Numerical (Continuous)</td>
                                <td>Bin width selection heavily alters visual interpretation; hides individual observations.</td>
                            </tr>
                            <tr>
                                <td><strong>Box Plot</strong></td>
                                <td>Display 5-number summary (Min, Q1, Median, Q3, Max) and isolate statistical outliers (1.5 &times; IQR).</td>
                                <td>Comparing medians and spreads across groups or auditing data anomalies.</td>
                                <td><em>"Do students who completed prep courses have higher median reading scores?"</em></td>
                                <td>1 Numerical, or 1 Categorical + 1 Numerical</td>
                                <td>Can hide multi-modal (bimodal) distributions; non-technical audiences may misinterpret quartiles.</td>
                            </tr>
                            <tr>
                                <td><strong>Bar Chart</strong></td>
                                <td>Compare aggregated statistical summaries (mean, total, median) across discrete categories.</td>
                                <td>Comparing performance differences between distinct groups.</td>
                                <td><em>"What is the average writing score by parental education level?"</em></td>
                                <td>1 Categorical (X) + 1 Numerical (Y)</td>
                                <td>Hides sample size and internal group distribution/spread; requires y-axis starting at zero to avoid distortion.</td>
                            </tr>
                            <tr>
                                <td><strong>Count Plot</strong></td>
                                <td>Display the raw frequency count and percentage breakdown for each unique categorical class.</td>
                                <td>Checking category balance, class representation, or survey demographics.</td>
                                <td><em>"How many students belong to each race/ethnic group in the sample?"</em></td>
                                <td>1 Categorical</td>
                                <td>Only reflects counts; provides zero information about scores, grades, or underlying relationships.</td>
                            </tr>
                            <tr>
                                <td><strong>Pie Chart</strong></td>
                                <td>Show relative proportions of a whole summing to 100%.</td>
                                <td>Displaying simple percentage shares with very few categories (2 to 4).</td>
                                <td><em>"What percentage of students completed the test preparation course?"</em></td>
                                <td>1 Categorical (Proportions)</td>
                                <td>Human vision struggles to compare angles and slice areas accurately; illegible with &gt;5 categories.</td>
                            </tr>
                            <tr>
                                <td><strong>Scatter Plot</strong></td>
                                <td>Plot individual coordinate pairs to reveal correlation, linearity, and clusters between two variables.</td>
                                <td>Evaluating bivariate numeric relationships and testing for collinearity.</td>
                                <td><em>"Is there a strong positive correlation between reading score and writing score?"</em></td>
                                <td>2 Numerical (Optional Categorical Hue)</td>
                                <td>Suffers from severe overplotting with large datasets; correlation does not imply causation.</td>
                            </tr>
                            <tr>
                                <td><strong>Line Chart</strong></td>
                                <td>Display continuous progression, sequential trends, or ordered transitions across ranked groups.</td>
                                <td>Tracking trends where the x-axis has a natural order (time, education rank, or percentiles).</td>
                                <td><em>"How do scores steadily climb as parental education rises from high school to master's degree?"</em></td>
                                <td>1 Ordered/Sequential X + 1 Numerical Y</td>
                                <td>Falsely implies continuous transition if applied to unordered, arbitrary nominal categories.</td>
                            </tr>
                            <tr>
                                <td><strong>Correlation Heatmap</strong></td>
                                <td>Provide an annotated color matrix of pairwise Pearson correlation coefficients (r).</td>
                                <td>Multivariate screening to detect redundant collinear features and strong predictors.</td>
                                <td><em>"Which pair of subjects exhibits the highest linear correlation?"</em></td>
                                <td>Multiple Numerical</td>
                                <td>Only measures linear relationships; non-linear dependencies can register near zero.</td>
                            </tr>
                            <tr>
                                <td><strong>Pair Plot</strong></td>
                                <td>Generate an N &times; N grid showing all pairwise scatter plots and univariate distributions simultaneously.</td>
                                <td>Comprehensive initial multivariate exploration across all continuous features.</td>
                                <td><em>"How do math, reading, and writing interact, and do genders separate into distinct clusters?"</em></td>
                                <td>All Numerical Features (+ Categorical Hue)</td>
                                <td>Computationally expensive; scales quadratically (N&sup2; subplots) and becomes cluttered if &gt;6 features.</td>
                            </tr>
                        </tbody>
                    </table>
                </div>
            </div>

        </div>

    </div>

    <!-- ============================================================= -->
    <!-- JAVASCRIPT LOGIC (Plain Vanilla JS) -->
    <!-- ============================================================= -->
    <script>
        // State variables
        let appSummary = null;
        let activeCodeTabs = { uni: 'mpl', bi: 'mpl', multi: 'mpl' };
        let currentUniChart = 'histogram';
        let currentMultiPreset = 'heatmap';
        let currentCompChart = 'histogram';

        // Initialize on DOM load
        document.addEventListener('DOMContentLoaded', () => {
            fetchSummary();
        });

        // Tab Switching
        function switchTab(tabId) {
            document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
            document.querySelectorAll('.tab-pane').forEach(pane => pane.classList.remove('active'));

            const targetBtn = Array.from(document.querySelectorAll('.tab-btn')).find(b => b.getAttribute('onclick').includes(tabId));
            if (targetBtn) targetBtn.classList.add('active');

            const targetPane = document.getElementById('tab-' + tabId);
            if (targetPane) targetPane.classList.add('active');

            // Trigger chart load on tab switch if needed
            if (tabId === 'univariate' && !document.getElementById('uni-chart-img')) {
                onUnivariateColChange();
            } else if (tabId === 'bivariate' && !document.getElementById('bi-chart-img')) {
                onBivariateTypeChange();
            } else if (tabId === 'multivariate' && !document.getElementById('multi-chart-img')) {
                loadMultivariate('heatmap');
            } else if (tabId === 'mpl-vs-sns' && !document.getElementById('comp-chart-img')) {
                loadComparison('histogram');
            }
        }

        // Fetch Dataset Summary from Flask API
        async function fetchSummary() {
            try {
                const res = await fetch('/api/summary');
                const data = await res.json();

                if (data.status === 'error') {
                    document.getElementById('missing-alert').style.display = 'flex';
                    document.getElementById('missing-msg').innerText = data.message;
                    return;
                }

                appSummary = data;
                populateOverview(data);
                populateDropdowns(data);
                onUnivariateColChange();
                onBivariateTypeChange();
            } catch (err) {
                console.error("Failed to fetch summary:", err);
                document.getElementById('missing-alert').style.display = 'flex';
            }
        }

        // Populate Overview Tab Cards and Tables
        function populateOverview(data) {
            document.getElementById('stat-students').innerText = data.total_students.toLocaleString();
            document.getElementById('stat-cols').innerText = data.total_columns;
            document.getElementById('stat-math').innerText = data.avg_math_score;
            document.getElementById('stat-reading').innerText = data.avg_reading_score;
            document.getElementById('stat-writing').innerText = data.avg_writing_score;
            document.getElementById('stat-missing').innerText = data.total_missing;
            document.getElementById('stat-duplicates').innerText = data.duplicate_count;

            // Five-number summary statistics
            const tbodyStats = document.getElementById('stats-tbody');
            tbodyStats.innerHTML = '';
            for (const [col, stats] of Object.entries(data.descriptive_stats)) {
                const tr = document.createElement('tr');
                tr.innerHTML = `
                    <td><strong>${col.toUpperCase()}</strong></td>
                    <td>${stats.count}</td>
                    <td>${stats.mean}</td>
                    <td>${stats.std}</td>
                    <td>${stats.min}</td>
                    <td>${stats.q25}</td>
                    <td><strong style="color: var(--primary);">${stats.median}</strong></td>
                    <td>${stats.q75}</td>
                    <td>${stats.max}</td>
                `;
                tbodyStats.appendChild(tr);
            }

            // Preview Table
            const thead = document.getElementById('preview-thead');
            const tbodyPreview = document.getElementById('preview-tbody');
            thead.innerHTML = '';
            tbodyPreview.innerHTML = '';

            const headerTr = document.createElement('tr');
            data.columns.forEach(col => {
                const th = document.createElement('th');
                th.innerText = col;
                headerTr.appendChild(th);
            });
            thead.appendChild(headerTr);

            data.preview.forEach(row => {
                const tr = document.createElement('tr');
                data.columns.forEach(col => {
                    const td = document.createElement('td');
                    td.innerText = row[col];
                    tr.appendChild(td);
                });
                tbodyPreview.appendChild(tr);
            });
        }

        // Populate dropdown options
        function populateDropdowns(data) {
            const biX = document.getElementById('bi-x-select');
            biX.innerHTML = '';
            
            const optCat = document.createElement('optgroup');
            optCat.label = "Categorical Variables";
            data.categorical_columns.forEach(c => {
                const opt = document.createElement('option');
                opt.value = c;
                opt.innerText = c;
                optCat.appendChild(opt);
            });
            biX.appendChild(optCat);

            const optNum = document.createElement('optgroup');
            optNum.label = "Numerical Variables";
            data.numeric_columns.forEach(c => {
                const opt = document.createElement('option');
                opt.value = c;
                opt.innerText = c;
                optNum.appendChild(opt);
            });
            biX.appendChild(optNum);
        }

        // Handle Univariate Column Change
        function onUnivariateColChange() {
            const col = document.getElementById('uni-col-select').value;
            const isNumeric = appSummary && appSummary.numeric_columns.includes(col);
            const btnGroup = document.getElementById('uni-chart-btns');
            btnGroup.innerHTML = '';

            if (isNumeric) {
                btnGroup.innerHTML = `
                    <button class="btn-option selected" id="uni-btn-histogram" onclick="setUnivariateChart('histogram')">
                        <span>📊 Histogram & KDE</span>
                        <span style="font-size: 0.75rem;">Distribution</span>
                    </button>
                    <button class="btn-option" id="uni-btn-box_plot" onclick="setUnivariateChart('box_plot')">
                        <span>📦 Box Plot</span>
                        <span style="font-size: 0.75rem;">5-Number Summary</span>
                    </button>
                `;
                currentUniChart = 'histogram';
            } else {
                btnGroup.innerHTML = `
                    <button class="btn-option selected" id="uni-btn-count_plot" onclick="setUnivariateChart('count_plot')">
                        <span>📊 Count Plot</span>
                        <span style="font-size: 0.75rem;">Frequencies</span>
                    </button>
                    <button class="btn-option" id="uni-btn-pie_chart" onclick="setUnivariateChart('pie_chart')">
                        <span>🥧 Pie Chart</span>
                        <span style="font-size: 0.75rem;">Proportions</span>
                    </button>
                `;
                currentUniChart = 'count_plot';
            }
            loadUnivariateChart();
        }

        function setUnivariateChart(chartType) {
            currentUniChart = chartType;
            document.querySelectorAll('#uni-chart-btns .btn-option').forEach(b => b.classList.remove('selected'));
            const activeBtn = document.getElementById('uni-btn-' + chartType);
            if (activeBtn) activeBtn.classList.add('selected');
            loadUnivariateChart();
        }

        // Load Univariate Chart via API
        async function loadUnivariateChart() {
            const col = document.getElementById('uni-col-select').value;
            const container = document.getElementById('uni-chart-container');
            container.innerHTML = `<div class="spinner"></div><p style="color: var(--slate); font-size: 0.9rem;">Rendering ${currentUniChart.replace('_', ' ')}...</p>`;

            try {
                const res = await fetch(`/api/chart?chart_type=${currentUniChart}&column=${encodeURIComponent(col)}`);
                const data = await res.json();

                if (data.status === 'success') {
                    container.innerHTML = `<img src="${data.image}" class="chart-img" id="uni-chart-img" alt="${data.title}">`;
                    document.getElementById('uni-when-to-use').innerText = data.when_to_use;
                    document.getElementById('uni-what-to-observe').innerText = data.what_to_observe;
                    document.getElementById('uni-limitations').innerText = data.limitations;
                    document.getElementById('uni-code-mpl').innerText = data.mpl_code;
                    document.getElementById('uni-code-sns').innerText = data.sns_code;
                } else {
                    container.innerHTML = `<div style="color: var(--danger);">Error: ${data.message}</div>`;
                }
            } catch (err) {
                container.innerHTML = `<div style="color: var(--danger);">Failed to load chart.</div>`;
            }
        }

        // Handle Bivariate Type Change
        function onBivariateTypeChange() {
            const chartType = document.getElementById('bi-chart-select').value;
            const hueGroup = document.getElementById('bi-hue-group');
            const xSelect = document.getElementById('bi-x-select');

            if (chartType === 'scatter_plot') {
                hueGroup.style.display = 'block';
                // Default X to reading score for scatter
                if (appSummary && appSummary.numeric_columns.length > 1) {
                    xSelect.value = 'reading score';
                }
            } else {
                hueGroup.style.display = 'none';
                if (chartType === 'line_chart') {
                    xSelect.value = 'parental level of education';
                } else if (appSummary && appSummary.categorical_columns.length > 0) {
                    xSelect.value = 'lunch';
                }
            }
            loadBivariateChart();
        }

        // Load Bivariate Chart via API
        async function loadBivariateChart() {
            const chartType = document.getElementById('bi-chart-select').value;
            const xCol = document.getElementById('bi-x-select').value;
            const yCol = document.getElementById('bi-y-select').value;
            const hue = document.getElementById('bi-hue-select').value;

            const container = document.getElementById('bi-chart-container');
            container.innerHTML = `<div class="spinner"></div><p style="color: var(--slate); font-size: 0.9rem;">Rendering ${chartType.replace('_', ' ')}...</p>`;

            let url = `/api/chart?chart_type=${chartType}&x_column=${encodeURIComponent(xCol)}&y_column=${encodeURIComponent(yCol)}`;
            if (chartType === 'scatter_plot' && hue) {
                url += `&hue=${encodeURIComponent(hue)}`;
            } else if (chartType === 'box_plot') {
                url = `/api/chart?chart_type=box_plot&column=${encodeURIComponent(yCol)}&category_column=${encodeURIComponent(xCol)}`;
            }

            try {
                const res = await fetch(url);
                const data = await res.json();

                if (data.status === 'success') {
                    container.innerHTML = `<img src="${data.image}" class="chart-img" id="bi-chart-img" alt="${data.title}">`;
                    document.getElementById('bi-when-to-use').innerText = data.when_to_use;
                    document.getElementById('bi-what-to-observe').innerText = data.what_to_observe;
                    document.getElementById('bi-limitations').innerText = data.limitations;
                    document.getElementById('bi-code-mpl').innerText = data.mpl_code;
                    document.getElementById('bi-code-sns').innerText = data.sns_code;
                } else {
                    container.innerHTML = `<div style="color: var(--danger);">Error: ${data.message}</div>`;
                }
            } catch (err) {
                container.innerHTML = `<div style="color: var(--danger);">Failed to load chart.</div>`;
            }
        }

        // Load Multivariate Preset
        async function loadMultivariate(preset) {
            currentMultiPreset = preset;
            document.querySelectorAll('#tab-multivariate .btn-option').forEach(b => b.classList.remove('selected'));
            const activeBtn = document.getElementById('multi-btn-' + preset);
            if (activeBtn) activeBtn.classList.add('selected');

            const hueGroup = document.getElementById('multi-hue-group');
            if (preset === 'pairplot') {
                hueGroup.style.display = 'block';
            } else {
                hueGroup.style.display = 'none';
            }

            const container = document.getElementById('multi-chart-container');
            container.innerHTML = `<div class="spinner"></div><p style="color: var(--slate); font-size: 0.9rem;">Rendering multivariate ${preset}...</p>`;

            let url = `/api/chart?chart_type=${preset}`;
            if (preset === 'pairplot') {
                const hue = document.getElementById('multi-hue-select').value;
                if (hue) url += `&hue=${encodeURIComponent(hue)}`;
            }

            try {
                const res = await fetch(url);
                const data = await res.json();

                if (data.status === 'success') {
                    container.innerHTML = `<img src="${data.image}" class="chart-img" id="multi-chart-img" alt="${data.title}">`;
                    document.getElementById('multi-when-to-use').innerText = data.when_to_use;
                    document.getElementById('multi-what-to-observe').innerText = data.what_to_observe;
                    document.getElementById('multi-limitations').innerText = data.limitations;
                    document.getElementById('multi-code-mpl').innerText = data.mpl_code;
                    document.getElementById('multi-code-sns').innerText = data.sns_code;
                } else {
                    container.innerHTML = `<div style="color: var(--danger);">Error: ${data.message}</div>`;
                }
            } catch (err) {
                container.innerHTML = `<div style="color: var(--danger);">Failed to load chart.</div>`;
            }
        }

        // Load Side-by-Side Matplotlib vs Seaborn Comparison
        async function loadComparison(chartType) {
            currentCompChart = chartType;
            document.querySelectorAll('#tab-mpl-vs-sns .tab-btn').forEach(b => b.classList.remove('active'));
            const activeBtn = document.getElementById('comp-btn-' + chartType);
            if (activeBtn) activeBtn.classList.add('active');

            const container = document.getElementById('comp-chart-container');
            container.innerHTML = `<div class="spinner"></div><p style="color: var(--slate); font-size: 0.9rem;">Generating side-by-side comparison...</p>`;

            try {
                const res = await fetch(`/api/comparison?chart_type=${chartType}`);
                const data = await res.json();

                if (data.status === 'success') {
                    container.innerHTML = `<img src="${data.image}" class="chart-img" id="comp-chart-img" alt="Comparison">`;
                    document.getElementById('comp-code-mpl').innerText = data.mpl_code;
                    document.getElementById('comp-code-sns').innerText = data.sns_code;
                } else {
                    container.innerHTML = `<div style="color: var(--danger);">Error: ${data.message}</div>`;
                }
            } catch (err) {
                container.innerHTML = `<div style="color: var(--danger);">Failed to load comparison.</div>`;
            }
        }

        // Switch Code View Tab between Matplotlib and Seaborn
        function switchCodeTab(section, lib) {
            activeCodeTabs[section] = lib;
            const container = document.getElementById(section + '-chart-container').parentElement;
            const btns = container.querySelectorAll('.code-tab-btn');
            btns[0].classList.toggle('active', lib === 'mpl');
            btns[1].classList.toggle('active', lib === 'sns');

            document.getElementById(section + '-code-mpl').style.display = lib === 'mpl' ? 'block' : 'none';
            document.getElementById(section + '-code-sns').style.display = lib === 'sns' ? 'block' : 'none';
        }

        // Copy Code to Clipboard
        function copyCode(section) {
            const lib = activeCodeTabs[section];
            const codeElem = document.getElementById(`${section}-code-${lib}`);
            navigator.clipboard.writeText(codeElem.innerText).then(() => {
                alert("Python code copied to clipboard!");
            });
        }

        function copyRawCode(elemId) {
            const text = document.getElementById(elemId).innerText;
            navigator.clipboard.writeText(text).then(() => {
                alert("Code snippet copied to clipboard!");
            });
        }
    </script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# 19. create_flask_routes()
# ---------------------------------------------------------------------------
def create_flask_routes(app):
    """
    Registers all routes for the single-file Flask EDA application:
    - GET / : Serves the main HTML dashboard
    - GET /api/summary : Returns dataset summary metrics and stats
    - GET /api/chart : Generates base64 chart images + educational info + code
    - GET /api/comparison : Generates side-by-side comparison for Matplotlib vs Seaborn
    """

    @app.route('/')
    def index():
        global DF, LOAD_ERROR
        # If dataset was not loaded at startup, attempt reloading once
        if DF is None:
            DF, LOAD_ERROR = load_dataset(CSV_PATH)
        return render_template_string(create_dashboard_html())

    @app.route('/api/summary')
    def api_summary():
        global DF, LOAD_ERROR
        if DF is None:
            DF, LOAD_ERROR = load_dataset(CSV_PATH)
        if DF is None:
            return jsonify({
                "status": "error",
                "message": LOAD_ERROR or "StudentsPerformance.csv could not be located in the project directory."
            }), 404

        summary = get_dataset_summary(DF)
        summary["status"] = "success"
        return jsonify(summary)

    @app.route('/api/chart')
    def api_chart():
        global DF, LOAD_ERROR
        if DF is None:
            DF, LOAD_ERROR = load_dataset(CSV_PATH)
        if DF is None:
            return jsonify({
                "status": "error",
                "message": "Dataset not found. Please place StudentsPerformance.csv in the same folder as eda_dashboard.py."
            }), 404

        chart_type = request.args.get('chart_type', 'histogram')
        column = request.args.get('column', 'math score')
        x_col = request.args.get('x_column', 'lunch')
        y_col = request.args.get('y_column', 'math score')
        category_col = request.args.get('category_column', None)
        hue = request.args.get('hue', None)

        try:
            image_data = None
            if chart_type == 'histogram':
                image_data = create_histogram(DF, column)
            elif chart_type == 'bar_chart':
                image_data = create_bar_chart(DF, x_col, y_col)
            elif chart_type == 'line_chart':
                image_data = create_line_chart(DF, x_col, y_col)
            elif chart_type == 'box_plot':
                image_data = create_box_plot(DF, column=column if not category_col else y_col, category_column=category_col)
            elif chart_type == 'scatter_plot':
                image_data = create_scatter_plot(DF, x_col, y_col, hue=hue)
            elif chart_type == 'heatmap':
                image_data = create_heatmap(DF)
            elif chart_type == 'count_plot':
                image_data = create_count_plot(DF, column)
            elif chart_type == 'pie_chart':
                image_data = create_pie_chart(DF, column)
            elif chart_type == 'pairplot':
                image_data = create_pairplot(DF, hue=hue)
            else:
                return jsonify({"status": "error", "message": f"Unsupported chart type '{chart_type}'."}), 400

            explanation = get_chart_explanation(chart_type)
            mpl_code = generate_matplotlib_code(chart_type, column=column, x_column=x_col, y_column=y_col, hue=hue)
            sns_code = generate_seaborn_code(chart_type, column=column, x_column=x_col, y_column=y_col, hue=hue)

            return jsonify({
                "status": "success",
                "chart_type": chart_type,
                "title": explanation["title"],
                "explanation": explanation["explanation"],
                "when_to_use": explanation["when_to_use"],
                "what_to_observe": explanation["what_to_observe"],
                "limitations": explanation["limitations"],
                "image": image_data,
                "mpl_code": mpl_code,
                "sns_code": sns_code
            })
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 400

    @app.route('/api/comparison')
    def api_comparison():
        global DF, LOAD_ERROR
        if DF is None:
            DF, LOAD_ERROR = load_dataset(CSV_PATH)
        if DF is None:
            return jsonify({"status": "error", "message": "Dataset not found."}), 404

        chart_type = request.args.get('chart_type', 'histogram')
        try:
            image_data = create_side_by_side_comparison(DF, chart_type)
            mpl_code = generate_matplotlib_code(chart_type)
            sns_code = generate_seaborn_code(chart_type)
            return jsonify({
                "status": "success",
                "chart_type": chart_type,
                "image": image_data,
                "mpl_code": mpl_code,
                "sns_code": sns_code
            })
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 400


# ---------------------------------------------------------------------------
# 20. main()
# ---------------------------------------------------------------------------
def main():
    """
    Main entry point for running the EDA Dashboard application.
    """
    global DF, LOAD_ERROR
    print("=" * 70)
    print("[INFO] Interactive EDA Classroom Dashboard - Single-File Edition")
    print("=" * 70)

    # Attempt loading the CSV dataset
    DF, LOAD_ERROR = load_dataset(CSV_PATH)
    if DF is not None:
        print(f"[OK] Dataset successfully loaded from: '{CSV_PATH}'")
        print(f"     Total Students: {len(DF)} | Total Features: {len(DF.columns)}")
        print(f"     Math Avg: {DF['math score'].mean():.2f} | Reading Avg: {DF['reading score'].mean():.2f} | Writing Avg: {DF['writing score'].mean():.2f}")
    else:
        print(f"[WARNING] {LOAD_ERROR}")
        print("          The web server will still start, and display an error banner in the browser.")

    # Initialize Flask app
    app = Flask(__name__)
    create_flask_routes(app)

    host = os.environ.get("FLASK_HOST", "127.0.0.1")
    port = int(os.environ.get("FLASK_PORT", 5000))

    print("-" * 70)
    print(f"[SERVER] Dashboard is starting on http://{host}:{port}/")
    print("         Open this link in your web browser for live classroom demonstration.")
    print("         Press CTRL+C in this terminal to stop the server.")
    print("=" * 70)

    app.run(host=host, port=port, debug=False)


if __name__ == '__main__':
    main()
