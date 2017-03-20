# EnRoutePerformance
All scripts are tested through on a windows 8.1 machine, and a Mac OS 10.11.6 machine. Development screen shot of a Mac machine is also included.


I. Dependent Python Packages:
_______________________________
	$ conda install pymongo
	$ pip install descartes
	$ conda install basemap
	$ conda install shapely
	$ pip install geopandas
	$ pip install pylogit
——————————————————————————————-
II. Usage
1. To use the packages, please first unzip the files “DependentData.zip”, “DependentData2.zip” and “DependentData3.zip” to the CURRENT directory 
2. Then run “GetWx_ARTCC.py” directly to extract ARTCC-based weather info.
3. “.py” files are packages, we need to use “.ipynb” files to call those packages and do further analysis, i.e., use the “02_EDA_Clustering.ipynb” to visualise the trajectories, and further conduct clustering analysis; use “03_BuildMNL_DATA_WX.IPYNB” to build MNL dataset and merge with convective weather; use “04_MIT.ipynb” to map with MIT. Between the convective weather and MIT module, we need to use the linux machine to conduct the wind analysis. “08_RegressionAnalysis.ipynb” is the one to fit regression models based on the MNL dataset.

III. Note:
1. DependentData2.zip is actually the output of GetEDA.py. Since we need to build mongodb to test the full function of GetEDA.py, I didn't do so in the testing phase. Considering FAA is using Oracle database, you may want to change the mongodb part to oracle. Thus, I just provide the output of GetEDA.py for IAH --> BOS case.
2. GetRouteSelectionModel.py needs to be further optimised. The way it constructs the nominal route set is memory-intensive. I think the right way to do it is similar to what I did in the wind mining script.
3. .ipynb file is a python markdown language, and can be opened by jupyter notebook. Examples of opening it:
	$ cd EnRoutePerformance
	$ jupyter notebook
4. There might be some warning messages of pyplot or pandas. This is mainly caused by the different versions of python packages, so don’t worry about them.
5. The matching results (with Convective weather, wind and MIT) are stored separately into ‘MNL_DEP_ARR_YEAR.csv’, ‘New_DEP_ARR_YEAR.csv’ and “Final_DEP_ARR.csv” in the folder ‘/MNL’. Since wind mapping is on another machine, I also provide the three files in the folder ‘/MNL’.
6. There are still some parts under development in the Mapping MIT module. However, for the ‘Nominal’ type, those unfinished part will not affect the performance.