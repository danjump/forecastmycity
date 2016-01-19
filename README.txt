Skeleton overview of project:

Data Analysis --------------------------------------------
Lots of raw data exploration in: 
data/

The data files used in the end were:
data/BEA-RegionalIncomeByIndustry/CA5N_2001_2013_MSA.csv
and
data/BEA-RegionalIncomeByIndustry/CA5_1969_2000_MSA.csv

Data preparation is done in:
analysis/prepare_regional_income_data.py

Time series fitting done in:
analysis/regional_income_fit.py

Results are then handled by:
analysis/generate_ranking.py

All of this generates a database containing results of financial 
projects and their rankings which will be used by the web app to plot
and display results.

Web App --------------------------------------------------
The web app uses flask to pull and plot results. I have apache running to serve the website on my server machine.

The main flask backend for this is in:
web_app/app/views.py

The simple html for this input and output are in:
web_app/app/templates/input.html
web_app/app/templates/output.html
