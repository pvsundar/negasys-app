# NEGASYS v5.0: Household Product Line Optimizer
## P.V. (Sundar) Balakrishnan
#### School of Business, University of Washington Bothell

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://negasys-app.streamlit.app)

**NEGASYS** (NEGotiated household ASsortment SYStem) is a decision-support tool for product-line optimization under dyadic household decision making.

## Live Application

**Access NEGASYS here:** https://negasys-app.streamlit.app

No installation required. Works on Windows, Mac, and Chromebook.

## Overview

Traditional product-line optimization assumes individual consumers. NEGASYS extends this to model **households as two-member units** (Partner H and Partner W) who jointly evaluate and choose products through negotiation.

## Key Features

- **Four Decision Rules**: Linear weighted, Symmetric Nash, Generalized Nash (Roth), Rawlsian minimum
- **Synthetic & Real Data**: Generate test data or upload conjoint utility files
- **Stage 0 Integration**: Assign status quo products from competitor set
- **Genetic Algorithm Optimization**: Find optimal product lines under capacity constraints
- **Multi-Rule Comparison**: Test all four rules and compare results
- **Export to CSV**: Download results for further analysis

## Decision Rules

| Rule | Formula | Interpretation |
|------|---------|----------------|
| **Linear Weighted** | αU_H + (1-α)U_W | Simple weighted average |
| **Symmetric Nash** | U_H × U_W | Both partners must benefit |
| **Generalized Nash** | U_H^α × U_W^(1-α) | Asymmetric bargaining power |
| **Rawlsian Min** | min{U_H, U_W} | Protect worst-off partner |

## Data Formats

### Utility File (.01b)
- Header: `num_consumers,num_attributes,levels_1,levels_2,...`
- Body: Part-worth utilities (one row per individual)
- Consecutive rows paired into households

### Competitor File (.01c)
- Header: `num_competitors,num_attributes,levels_1,levels_2,...`
- Body: Attribute codes for each competitor product

## Teaching Materials

This application accompanies the **SmartTVs 4'Us** teaching case. Contact the author for the complete teaching packet including:
- Case narrative
- Teaching note (instructor only)
- Student assignment
- Sample data files

## Author

**P.V. (Sundar) Balakrishnan**  
School of Business  
University of Washington Bothell

## References

- Nash, J.F. (1950). The bargaining problem. *Econometrica*, 18(2), 155-162.
- Roth, A.E. (1979). *Axiomatic models of bargaining*. Springer-Verlag.
- Balakrishnan, P.V. & Jacob, V.S. (1996). Genetic algorithms for product design. *Management Science*, 42(8), 1105-1117.

## License

For educational use only. Contact author for permissions.
