---
title: Teaching

view: 1

header:
  caption: ""
  image: ""
---

I teach finance and data science by having students **do** finance and data science. The courses below are built around real data, working code, and live decisions, not stylised textbook examples where everything fits and nothing breaks. Students run the analysis, watch it fail, and learn to say honestly what the evidence does and does not support. Rigour, intellectual humility, and the discipline to quantify uncertainty are not separate skills bolted on at the end. They are how the doing is structured.

The materials are open. The labs, the slides, the trading platform, and the companion textbooks are all on the public web so that anyone, student or otherwise, can try the same exercises in their own time.

## Current teaching at Ulster University

### FIN306: FinTech and Data Science

**Undergraduate core module** taught as an applied statistical science. Students fit models on real financial data, watch them overfit, fix them, and write up what they can and cannot conclude. The course confronts the replication crisis directly and asks students to be explicit about what their analyses establish.

**Open companion textbook:** *[Statistical Science for Finance: Rigorous Methods for Technology-Enabled Markets](https://quinfer.github.io/financial-data-science/)* (published May 2026). Covers return predictability, volatility modelling, factor models, and machine learning for cross-sectional analysis, with emphasis on signal versus noise, walk-forward validation, and intellectual humility about prediction in noisy markets. All labs, slides, and datasets are open access.

### MiniMBA: FinTech and Data Science

**Three-day intensive** (15 credits, level 7, BMG850) combining economic reasoning with validation-first data science. Not a course in building trading bots; a course in making **disciplined, evidence-based judgements** about models, data, and deployment.

**Programme site:** [quinfer.github.io/minimba](https://quinfer.github.io/minimba/)

**Day 1.** The economics of FinTech: cost puzzle, platform business models, the economics of prediction.  
**Day 2.** Responsible data science: validation discipline, overfitting, backtest reliability, walk-forward testing, survivorship bias, and an LLM-as-critic exercise.  
**Day 3.** AI in finance: a real case study (the UKFin+ project) on AI for regulatory compliance, lexical drift, the confidence trap, and graduated automation.

Slides, labs, and Colab notebooks are all served from the programme site.

### TickLab: a trading simulation platform for the classroom

**TickLab** is a web-based trading simulation platform I am building for teaching market microstructure and trading dynamics. It carries forward the pedagogy of the QUB **trading principles** course (described below) but on infrastructure I control, so the simulation can evolve with the curriculum: new missions, a Valuation Workbench, a tutorial player, and a Python order-book engine with market makers and liquidity and informed traders running over WebSocket.

In a TickLab session students assume distinct industry roles (market maker, liquidity trader, informed trader), and the prices and liquidity that emerge in the room are the consequence of the decisions they make. Concepts such as the law of one price, price formation, market efficiency, and event arbitrage stop being slide titles and become things that visibly happen because of what students just did.

**Live site:** [www.ticklab.co.uk](https://www.ticklab.co.uk) (in development).

## Earlier teaching at Queen's University Belfast (2010–2025)

At QUB I taught postgraduate courses in **algorithmic trading and investment** (López de Prado framework for ML in financial markets, with the full pipeline from feature engineering to backtesting and a heavy emphasis on avoiding overfitting and leakage), **financial econometrics** (applied statistical science, not a recipe book), and **trading principles** (a learning-by-doing course built around roughly 18 live trading simulations using UpTick and Bloomberg Terminal data). The trading principles course is the direct ancestor of TickLab.

### Teaching innovations from QUB

Two pedagogical projects I helped kick-start while at Queen's:

- **Q-RaP (Queen's Management School Remote analytics Platform)** was a high-performance cloud stack (RStudio Team on Azure) for open-science analytics, launched in 2021 for postgraduate teaching. It was retired when I moved to Ulster and is no longer documented as a standalone page here.

- **[Queen's Student Managed Fund (QSMF)](smf/)** is a real-money investment portfolio managed by students. Established in 2014 (virtual) and transitioned to real money in 2015 with alumni support. Leadership was shared with Alan Hanna and Aine Gallagher, who took the fund forward in later years as I stepped away to develop the FAIR (Finance and AI Research) lab. Grounded in experiential learning, professionalism, and corporate engagement. The archive includes annual reports and student testimonials.

## Open educational resources

- Quinn, Barry (2026). *Statistical Science for Finance: Rigorous Methods for Technology-Enabled Markets.* Open textbook and companion for FIN306, Ulster University Business School. [quinfer.github.io/financial-data-science](https://quinfer.github.io/financial-data-science/)
- Quinn, Barry (2026). *MiniMBA: FinTech and Data Science.* Programme materials (slides, labs, Colab notebooks). [quinfer.github.io/minimba](https://quinfer.github.io/minimba/)
- *TickLab.* Web-based trading simulation platform for teaching market microstructure. [www.ticklab.co.uk](https://www.ticklab.co.uk) (in development).

## Publications

- Quinn, Barry, Hanna, Alan, Gallagher, Aine (2018). Queen's Student Managed Fund: Investing in the Student Experience. *Reflections*, 27, pp. 16–17.
