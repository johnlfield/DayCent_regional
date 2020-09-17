# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.6.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # DayCent regional simulation results post-processing
# This Jupyter Notebook is designed to facilitate post-processing and analysis of sets of raw DayCent results from a regional scale simulation. For more information, contact author [John Field](https://johnlfield.weebly.com/) at <john.L.field@gmail.com>
#
# ## DayCent background
# DayCent is a process-based model that simulates agro-ecosystem net primary production, soil organic matter dynamics, and nitrogen (M) cycling and trace gas emissions. DayCent is a daily-timestep version of the older CENTURY model. Both models were created and are currently maintained at the Colorado State University [Natural Resource Ecology Laboratory](https://www.nrel.colostate.edu/) (CSU-NREL), and source code is available upon request.  DayCent model homepage:  [https://www2.nrel.colostate.edu/projects/daycent/](https://www2.nrel.colostate.edu/projects/daycent/)
#
# ![Alt text](DayCent.png)
#
#
# ## Regional simulation workflow
# The primary spatial data inputs to DayCent are:
# * soil texture as a function of depth
# * historic daily weather (Tmin, Tmax, precip)
#
# Our DayCent spatial modeling workflow is based on a national-scale GIS database of current land use ([NLCD](https://www.mrlc.gov/national-land-cover-database-nlcd-2016)), soil ([SSURGO](https://www.nrcs.usda.gov/wps/portal/nrcs/detail/soils/survey/?cid=nrcs142p2_053627)), and weather ([NARR](https://www.ncdc.noaa.gov/data-access/model-data/model-datasets/north-american-regional-reanalysis-narr)) data layers housed at CSU-NREL. The python-based workflow consists of a collection of scripts that perform the following:
# 1. Selection of area to be simulated, specified based on current land cover and/or land biophysical factors (i.e., soil texutre, slope, land capability class rating, etc.)
# 2. Determination of individual unique DayCent model runs (i.e., **"strata"**) necessary to cover the heterogenity of soils and climate across the simulation area
# 3. Parallel execution of simulations on the CSU-NREL computing cluster
# 4. Results analysis and mapping (this routine)
#

# import the necessary modules
import constants
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly as py
import plotly.figure_factory as ff
from plotly.offline import init_notebook_mode, iplot
from plotly.graph_objs import *

# ## Loading data
# Individual DayCent strata are specified via a .csv format "runtable" file, which contains the following information:
# * unique identifier (strata_no)
# * ID for DayCent spin-up archive (runno)
# * SSURGO soil map unit ID (mukey_int)
# * NARR climate grid ID (gridx & gridy)
# * county FIPS code (fips)
# * DayCent-format schedule file to simulate (sch_file)
# * latitude of the county centroid, used to set perennial grass phenology (latitude)
# * for simulations on abandoned agricultural land, year of peak ag land extent (peak_year)
# * land area represented by that strata (tot_ha)
# The code below loads the relevant runtable to a Pandas dataframe.

runtable = "eastern_US_runtable_incl81.csv"
run_df = pd.read_csv(runtable, skiprows=[1])   # skip SQL datatype row
run_df

# Raw DayCent model output is spread across two files:
# * .lis files contain information related to per-area biomass harvest and soil carbon
# * year_summary.out contains per-area trace gas emissions
# Most DayCent outputs are in units of grams of carbon per meter squared (g C m-2), though some of the nitrogen flux results are reported on a per-hectare basis instead.
#
# The code below loads these raw results into Pandas dataframes, merges them, and performs basic unit converions to express the results in more familiar units of kg or Mg per hectare.

lis_df

# +
# loading data
lis_file = "X.lis"
# lis_file = '/Volumes/wcnr-network/Public/RubelScratch/jlf/results/2019-09-16,13.26__eastern_US_runtable_incl81__79__CBI_baseline/X.lis'
lis_df = pd.read_csv(lis_file, skiprows=[1])
ys_file = "year_summary.out"
# ys_file = '/Volumes/wcnr-network/Public/RubelScratch/jlf/results/2019-09-16,13.26__eastern_US_runtable_incl81__79__CBI_baseline/year_summary.out'
ys_df = pd.read_csv(ys_file, skiprows=[1])

# merging .lis and year_summary.out results
annual_df = pd.merge(lis_df, ys_df, on=['strata_no', 'crop', 'land_type', 'time'])

# unit conversions
annual_df['yield_Mg_ha'] = ((annual_df['crmvst'] * constants.g_m2_to_Mg_ha) / constants.C_concentration)
annual_df['dSOC_MgC_ha'] = (annual_df['d_somsc'] * constants.g_m2_to_Mg_ha)
annual_df['dN2ON_kgN_ha'] = (annual_df['N2Oflux'] * constants.g_m2_to_kg_ha)
annual_df['iN2ON_kgN_ha'] = ((0.0075 * annual_df['strmac(2)'] +
                               0.01 * annual_df['volpac'] +
                               0.01 * annual_df['NOflux']) * constants.g_m2_to_kg_ha)
annual_df['kgCH4_ox_ha'] = (annual_df['CH4'] * constants.g_m2_to_kg_ha)
annual_df['ghg_MgCO2e_ha'] = (annual_df['dSOC_MgC_ha'] * constants.C_to_CO2 * -1.0) + \
                               ((annual_df['dN2ON_kgN_ha'] + annual_df['iN2ON_kgN_ha']) *
                                constants.kg_ha_to_Mg_ha * constants.N_to_N2O * constants.N2O_GWP100_AR5) + \
                              (annual_df['kgCH4_ox_ha'] * constants.kg_ha_to_Mg_ha * constants.CH4_GWP100_AR5 * -1.0)

annual_df
# print("Full list of raw and unit-convered DayCent output variables included in results:")
# for col in results_df.columns:
#     print(col)
# -

# ## Data aggregation
# Calculating annual average per-ha results at the strata level (ignoring all years prior to the first year of switchgrass harvest in 2020):

strata_df = annual_df[annual_df['time'].between(2020, 2068, inclusive=True)][
        ['strata_no', 'land_type', 'yield_Mg_ha', 'dSOC_MgC_ha', 'dN2ON_kgN_ha', 'iN2ON_kgN_ha', 'kgCH4_ox_ha', 'ghg_MgCO2e_ha']].groupby(
            ['strata_no', 'land_type']).mean()
strata_df = strata_df.reset_index()  # simplifies future merge operations
strata_df

# Next, we combine the total production and impacts associated with each strata by multiplying the per-ha results with the area represented by each strata:

# +
area_df = pd.merge(strata_df, run_df, on='strata_no')  # re-associate FIPS codes and land areas with strata results

# calculate area totals
area_df['yield_Mg'] = area_df['yield_Mg_ha'] * area_df['tot_ha']
area_df['dSOC_MgC'] = area_df['dSOC_MgC_ha'] * area_df['tot_ha']
area_df['dN2ON_kgN'] = area_df['dN2ON_kgN_ha'] * area_df['tot_ha']
area_df['iN2ON_kgN'] = area_df['iN2ON_kgN_ha'] * area_df['tot_ha']
area_df['kgCH4_ox'] = area_df['kgCH4_ox_ha'] * area_df['tot_ha']
area_df['ghg_MgCO2e'] = area_df['ghg_MgCO2e_ha'] * area_df['tot_ha']
area_df
# -

# Finally, we aggregate these results to the county scale, and divide by the total area simulated for each county to calculate area-weighted results:

# +
county_df = area_df[['fips', 'tot_ha', 'yield_Mg', 'dSOC_MgC', 'dN2ON_kgN', 'iN2ON_kgN', 'kgCH4_ox',
                       'ghg_MgCO2e']].groupby('fips').sum()
county_df = county_df.reset_index()

county_df['yield_Mg_ha'] = county_df['yield_Mg'] / county_df['tot_ha']
county_df['dSOC_MgC_ha'] = county_df['dSOC_MgC'] / county_df['tot_ha']
county_df['dN2ON_kgN_ha'] = county_df['dN2ON_kgN'] / county_df['tot_ha']
county_df['iN2ON_kgN_ha'] = county_df['iN2ON_kgN'] / county_df['tot_ha']
county_df['kgCH4_ox_ha'] = county_df['kgCH4_ox'] / county_df['tot_ha']
county_df['ghg_MgCO2e_ha'] = county_df['ghg_MgCO2e'] / county_df['tot_ha']
county_df

# +
init_notebook_mode(connected=True)

scope = ''
# scope = ['MT', 'WY', 'CO', 'NM', 'TX', 'OK', 'KS', 'NE', 'SD', 'ND', 'MN',
#          'IA', 'MO', 'AR', 'LA', 'MS', 'IL', 'WI', 'MI', 'IN', 'OH', 'KY',
#          'TN', 'AL']

def fips_mapping(df, title, column_mapped, legend_title, linspacing, divergent=False, reverse=False):

    # use 'linspacing' parameters to a bin list, and specify rounding if values are small-ish
    bin_list = np.linspace(linspacing[0], linspacing[1], linspacing[2]).tolist()
    rounding = True
    if linspacing[1] < 10:
        rounding = False

    kwargs = {}
    if scope:
        kwargs['scope'] = scope

    if divergent:
        # convert matplotlib (r, g, b, x) tuple color format to 'rgb(r, g, b)' Plotly string format
        cmap = get_cmap('RdBu')  # or RdYlBu for better differentiation vs. missing data squares in tiling map
        custom_rgb_cmap = [cmap(x) for x in np.linspace(0, 1, (linspacing[2] + 1))]
        custom_plotly_cmap = []
        for code in custom_rgb_cmap:
            plotly_code = 'rgb({},{},{})'.format(code[0] * 255.0, code[1] * 255.0, code[2] * 255.0)
            custom_plotly_cmap.append(plotly_code)
        if reverse:
            custom_plotly_cmap.reverse()

        kwargs['state_outline'] = {'color': 'rgb(100,100,100)', 'width': 1.0}
        kwargs['colorscale'] = custom_plotly_cmap

    fig = ff.create_choropleth(fips=df['fips'],
                               values=df[column_mapped].tolist(),
                               binning_endpoints=bin_list,
                               round_legend_values=rounding,
                               county_outline={'color': 'rgb(255,255,255)', 'width': 0.25},
                               legend_title=legend_title,
                               title=title,
                               paper_bgcolor='rgba(0,0,0,0)',
                               plot_bgcolor='rgba(0,0,0,0)',
                               **kwargs)
    iplot(fig)


# -

fips_mapping(county_df, 'Abandoned land availability', 'tot_ha', '(ha)', (0, 100000, 21))

# ## Climate analysis
# Here's some initial exploratory code to parse a DayCent-format weather file and analyze inter-annual variability in growing-season temperatures and precipitation.

weather_file1 = "NARR_89_234.wth"
weather_df1 = pd.read_csv(weather_file1, sep='\t',
                         names=['DayOfMonth','Month', "Year", "DayOfYear", 'Tmax_C', 'Tmin_C', "Precip_cm"])
weather_file2 = "NARR_89_231.wth"
weather_df2 = pd.read_csv(weather_file2, sep='\t',
                         names=['DayOfMonth','Month', "Year", "DayOfYear", 'Tmax_C', 'Tmin_C', "Precip_cm"])
weather_df2

# +
seasonal_wth_df1 = weather_df1[weather_df1['Month'].isin([5, 6, 7, 8, 9])]
seasonal_wth_df1['Tavg_C'] = (seasonal_wth_df1['Tmin_C'] + seasonal_wth_df1['Tmax_C']) / 2.0
annunal_wth_df1 = seasonal_wth_df1.groupby('Year').agg({'Tmax_C': 'mean','Tavg_C': 'mean', 'Precip_cm': 'sum'})
annunal_wth_df1 = annunal_wth_df1.reset_index()

seasonal_wth_df2 = weather_df2[weather_df2['Month'].isin([5, 6, 7, 8, 9])]
seasonal_wth_df2['Tavg_C'] = (seasonal_wth_df2['Tmin_C'] + seasonal_wth_df2['Tmax_C']) / 2.0
annunal_wth_df2 = seasonal_wth_df2.groupby('Year').agg({'Tmax_C': 'mean','Tavg_C': 'mean', 'Precip_cm': 'sum'})
annunal_wth_df2 = annunal_wth_df2.reset_index()
annunal_wth_df2
# -

plt.plot(annunal_wth_df1.Year, annunal_wth_df1.Precip_cm)
plt.plot(annunal_wth_df2.Year, annunal_wth_df2.Precip_cm)
plt.title("Difference between two weather grid centroids, 100km apart")
plt.xlabel("Year")
plt.ylabel("May–Sept. total precipitation (cm)")

plt.scatter(annunal_wth_df1.Tavg_C, annunal_wth_df1.Precip_cm)
plt.title("Inter-annual variability in growing season weather")
plt.xlabel("May–Sept. average air temperature (C)")
plt.ylabel("May–Sept. total precipitation (cm)")
