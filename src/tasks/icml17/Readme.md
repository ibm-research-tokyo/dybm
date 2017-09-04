(C) Copyright IBM Corp. 2017

# Scripts for reproducing figures in the ICML 2017 paper

This directory contains scripts that can be used to reproduce the experimental results and figures reported in the following paper:

T. Osogami, H. Kajino, and T. Sekiyama, "Bidirecdtional learning for time-series models with hidden units," ICML 2017.

## Prerequisite

The scripts rely on __netCDF4__ and __xlrd__.

```
pip install netCDF4 xlrd
```

## Download datasets

We use three datasets.  Please download the datasets and place them under 'data/'.

### Sunspot

Download the "CSV (,)" file, monthly-sunspot-number-zurich-17.csv, from "Export" in the "Monthly sunspot number, Zurich, 1749-1983" page at https://datamarket.com/data/set/22t4/

### Price

Download the excel file from the "Weekly Retail Gasoline and Diesel Prices" page at https://www.eia.gov/dnav/pet/pet_pri_gnd_a_epm0_pte_dpgal_w.htm

`wget https://www.eia.gov/dnav/pet/xls/PET_PRI_GND_A_EPM0_PTE_DPGAL_W.xls`

### Climate

Download the file, air.mon.anom.nc, from the "NOAA Global Surface Temperature (NOAAGlobalTemp)" page at https://www.esrl.noaa.gov/psd/data/gridded/data.noaaglobaltemp.html

`wget ftp://ftp.cdc.noaa.gov/Datasets/noaaglobaltemp/air.mon.anom.nc`

We have used V4.0.0 in the paper, but only V4.0.1 is available as of June 27, 2017.

## Convert datasets into numpy format

`python convert.py`

## Run experiments

```
python run_icml17_fig3.py 6 0 false
python run_icml17_fig3.py 6 1 false
python run_icml17_fig3.py 6 1 true
python run_icml17_fig3.py 8 0 false
python run_icml17_fig3.py 8 1 false
python run_icml17_fig3.py 8 1 true
python run_icml17_fig4.py sunspot test 1000 30 0 0 0.0
python run_icml17_fig4.py sunspot test 1000 30 4 0 0.0
python run_icml17_fig4.py sunspot test 1000 30 4 2 0.25
python run_icml17_fig4.py sunspot test 1000 30 4 2 0.5
python run_icml17_fig4.py sunspot test 1000 30 4 2 1.0
python run_icml17_fig4.py price test 10000 3 0 0 0.0
python run_icml17_fig4.py price test 10000 3 4 0 0.0
python run_icml17_fig4.py price test 10000 3 4 2 0.25
python run_icml17_fig4.py price test 10000 3 4 2 0.5
python run_icml17_fig4.py price test 10000 3 4 2 1.0
python run_icml17_fig4.py climate test 1000 2 0 0 0.0
python run_icml17_fig4.py climate test 1000 2 4 0 0.0
python run_icml17_fig4.py climate test 1000 2 4 2 0.25
python run_icml17_fig4.py climate test 1000 2 4 2 0.5
python run_icml17_fig4.py climate test 1000 2 4 2 1.0
```

## Make figures

```
python plot_icml17_fig3.py
python plot_icml17_fig4.py
```
