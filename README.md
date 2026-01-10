# Projet_Barca
# FC Barcelona Performance Analysis 2024-2025

## Research Question

**How do key players (Pedri, Raphinha, Lamine Yamal, and Iñigo Martínez) impact FC Barcelona's performance, and which factors best predict match victories?**

This project analyzes FC Barcelona's 2024-2025 La Liga season through two complementary approaches:
1. **Team Analysis**: Predicting match outcomes using machine learning models
2. **Key Player Analysis**: Identifying the most impactful players using clustering and custom metrics

## Setup

### Prerequisites
- Python 3.11
- Conda (recommended for Nuvolos) or pip

### Clone the repository
```bash
git clone https://github.com/Athithsko/Projet_Barca.git
cd Projet_Barca
```

### Create Environment

```bash 
# Using Conda (recommended)
conda env create -f environment.yml
conda activate Projet_Barca

# Or using pip
pip install -r requirement.txt

```

```bash 
## Usage


python Main.py
ls Graphics_Output/ # to be sure That the Main.py did launch correctly

```
**Expected output**: 
- Team performance analysis with ML model comparisons
- Key player rankings with impact scores
- Visualizations comparing model predictions vs actual results

## Project Structure

```
## Project Structure

Projet_Barca/
├── main.py                    # Entry point
├── README.md                  # This file
├── PROPOSAL.md                # Project proposal
├── AI_USAGE.md                # AI tools disclosure
├── environment.yml            # Conda dependencies
├── requirements.txt           # Pip dependencies
├── Src/
│   ├── Team_Data_Loader.py    # Data loading and preprocessing
│   ├── Analysis/
│   │   ├── Analysis_team.py   # Team exploratory analysis
│   │   └── Analysis_key_player.py  # Player metrics creation
│   ├── ML/
│   │   ├── ML_Team.py         # Victory prediction models
│   │   └── ML_Key_Player.py   # K-means clustering analysis
│   └── Graphics/
│       ├── Visu_Team.py       # Team visualizations
│       └── Visu_Key.py        # Player visualizations
├── Data_set/
│   ├── ProjetBarca.csv        # Team match data (38 matches)
│   └── Key_players.csv        # Player statistics
├── Graphics_Output/           # Output plots 
│   
└── notebooks/                 # Exploration (optional)
```
## Methodology

### Data Sources
- **Team Data**: 38 La Liga matches with metrics including xG, xGA, possession, goals, and opponent information
- **Player Data**: Individual statistics for 4 key players (Pedri, Raphinha, Lamine Yamal, Iñigo Martínez)

### Team Analysis Pipeline
1. **Data Preprocessing**: Clean data, convert formats, create target variables (Victory/Defeat/Draw)
2. **Exploratory Analysis**: Home vs Away performance, xG efficiency, correlation analysis
3. **ML Models**: Compare Logistic Regression, Random Forest, and Gradient Boosting
4. **Temporal Split**: Train on first 30 matches, test on last 8 matches

### Key Player Analysis Pipeline
1. **Advanced Metrics Creation**: Per-90-minute statistics, efficiency ratios, impact scores
2. **Player Role Classification**: Automatic role assignment based on performance profile
3. **K-Means Clustering**: Unsupervised grouping with elbow method optimization
4. **Ranking System**: Custom scoring combining offensive, defensive, passing, and dribbling dimensions

### Features Used

**Team Prediction Model:**
- `Equipe_type`: Team composition (with/without star players)
- `xG`, `xGA`: Expected goals for the team and against
- `Poss`: Possession percentage
- `Venue`: Home (1) or Away (0)
- `Opponent_tier`: Manual classification (1=Elite to 4=Relegation)

**Player Clustering:**
- Scoring: Goals/90, xG/90, Conversion Rate
- Playmaking: Assists/90, Progressive Passes/90
- Defensive: Tackles, Interceptions, Ball Recoveries
- Dribbling: Successful Take-Ons, Progressive Carries

## Results

### Team Analysis

We evaluated three models using temporal splitting (Train: 30 matches, Test: 8 matches) and 5-fold Cross-Validation to assess robustness.

| Model | Accuracy | F1-Score | ROC-AUC (Test) | ROC-AUC (CV Mean) |
|-------|----------|----------|----------------|-------------------|
| Logistic Regression | 0.750 | 0.833 | 0.714 | 0.930 (+/- 0.098) |
| Random Forest | 0.750 | 0.857 | 0.286 | 0.857 (+/- 0.179) |
| Gradient Boosting | 0.875 | 0.933 | 0.286 | 0.703 (+/- 0.213) |

``` Best model: Gradient Boosting (Accuracy) vs Logistic Regression (AUC Stability) ```

**Key Findings:**
- **Accuracy vs Robustness**: While Gradient Boosting achieved the highest accuracy (87.5%) on the test set, Logistic Regression demonstrated better separation capability (Best ROC-AUC: 0.714).
- **Cross-Validation**: The 5-fold cross-validation reveals that Logistic Regression is actually the most robust model on average (AUC 0.930), while tree-based models suffer more variance due to the small dataset size.
- **Predictive Factors**: xG Efficiency and xGA Efficiency proved to be the most consistent predictors of victory.

### Player Analysis
| Player | Role | Impact Score | Ml Ranking score |
|--------|------|--------------|---------|
| Lamine Yamal | Playmaker | 70.2 | 0.339 |
| Raphinha | Goalscorer | 54.6 | 0.370 |
| Pedri  | Creative Midfielder | 51.6 | 0.298 |
| Iñigo  | Defender | 56.1 | 0.341 |

## Key Metrics Explained

### Overall Impact Score (0-100)

A composite metric inspired by FIFA player ratings, **weighted by player role** to reflect their tactical responsibilities:

#### **Role-Specific Weight Distribution**

| Role | Offensive | Dribbling | Passing | Defensive | Stamina |
|------|-----------|-----------|---------|-----------|---------|
| **Goalscorer** | 40% | 20% | 20% | 10% | 10% |
| **Playmaker** | 30% | 40% | 30% | 10% | 10% |
| **Creative Midfielder** | 5% | 30% | 30% | 20% | 15% |
| **Defender** | 5% | 15% | 30% | 30% | 20% |
| **Default** | 25% | 25% | 20% | 20% | 10% |

### xG Efficiency
```xG_efficiency = Goals Scored / Expected Goals (xG)```
- Value > 1.0: Overperforming (clinical finishing)
- Value < 1.0: Underperforming (missed chances)

## Reproducibility

```All random operations use random_state=42 for reproducibility:```

- Logistic Regression, Random Forest, and Gradient Boosting models (ML_Team.py)
- K-Means clustering (ML_Key_Player.py)

```Note: The train/test split uses a temporal approach (first 30 matches for training, last 8 for testing) rather than random splitting, which is more appropriate for time-series sports data.```

## Requirements

- Python 3.11
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn

## Author

Athithiyan Rajeswaran

Data Science and Advanced Programming 2025-2026 - Final Project - Projet_Barca

## References

- Dataset: FC Barcelona La Liga 2024-2025 season statistics
- Libraries: scikit-learn, pandas, matplotlib, seaborn
- Methodology: FBref for advanced football metrics definitions
