# spectral

A basic toolbox for performing spectral analyses. It is a translation of parts of the LTPDA Matlab toolbox that was used for the analysis of the ESA LISA Pathfinder mission.

* More details about LISA Pathfinder: https://sci.esa.int/web/lisa-pathfinder
* Source to the LTPDA toolbox: https://www.elisascience.org/ltpda/

The installation is as usual 
```python
python3 setup.py install
```

An example for a given time series `data` stored in a numpy array:

```python
fs = 1.0 # The sampling frequency of the time series is required

f, S, Se, ENBW, N = lpsd(data, fs, Jdes=200, win='nuttall4b', olap=10,order=0,DOPLOT=True,VERBOSE=False)
```
![Alt text](example/example.png?raw=true)
