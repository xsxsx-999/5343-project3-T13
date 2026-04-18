## Introduction
Designed for STATGR5243, this project establishes a robust framework for conducting data-driven A/B testing. We developed an interactive data-cleaning platform via ShinyApps and instrumented it with Google Analytics for precise telemetry collection. The core of the project culminates in a full-scale A/B test, transitioning from experimental design to the statistical analysis of real-world user data to evaluate platform efficiency.


## Project Structure
```
STATGR5243-Project3/
├── version_A/                  # Packaged source code for ShinyApp Version A
├── version_B/                  # Packaged source code for ShinyApp Version B
├── README.md                   # Project documentation
├── index.html                  # Reference code for webpage deployment
├── raw_data.csv                # Raw user data collected via Google Analytics
├── cleaning.ipynb              # Jupyter Notebook for cleaning and preprocessing raw data
├── clean_events_from_raw.csv   # Cleaned dataset based on events extracted from raw data
├── clean_users_from_raw.csv    # Cleaned dataset based on users extracted from raw data
└── ab_test.ipynb               # Main notebook containing the core A/B testing analysis
```

## Note for Course Submission

This project fulfills the advanced requirements for Project 3 by demonstrating:

* **Experimental Design:** A well-structured A/B testing setup with clearly justified research goals, hypotheses, and a strong methodology.
* **Data Collection & Quality:** Original, high-quality user data actively collected via Google Analytics, featuring well-organized datasets and thorough documentation of our data choices.
* **Statistical Analysis:** Strong statistical analysis applied to real-world data, ensuring correct interpretations of user performance metrics.
* **Results & Insights:** Insightful and well-supported conclusions regarding UI efficiency, accompanied by a meaningful discussion of the findings.
* **Code Reproducibility:** Well-documented and easily reproducible code, with clear explanations provided throughout our data cleaning and analysis notebooks.


## URL for AB test:

Launched URL: [https://xsxsx-999.github.io/random-url-for-AB-test-5243/](https://xsxsx-999.github.io/random-url-for-AB-test-5243/)

Repository: https://github.com/xsxsx-999/random-url-for-AB-test-5243

This URL Randomly split users to different versions of our App. 

### Direct URLs for 2 versions:

version A: https://xsxshiny.shinyapps.io/ab-test-version-a/

version B: https://xsxshiny.shinyapps.io/ab-test-version-b/
