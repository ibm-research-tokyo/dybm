rem (C) Copyright IBM Corp. 2015

@echo off
pushd %0\..

rem 
rem Run experiments
rem
if %1==1 (
   echo Running 1
   java -classpath lib\commons-math.jar;bin com.ibm.stdp.science.SingleExperiment3 SCIENCE 3 3 9
) else if %1==2 (
  java -classpath lib\commons-math.jar;bin com.ibm.stdp.science.MirrorExperiment3 SCIENCE 5 3 3 9
) else if %1==3 (
  java -classpath lib\commons-math.jar;bin com.ibm.stdp.science.EvolutionExperiment 3 3 9
) else if %1==4 (
  java -classpath lib\commons-math.jar;bin com.ibm.stdp.science.MusicianExperiment 3 3 9
)

rem
rem Analyze results
rem

if %1==1 (
   java -classpath bin com.ibm.stdp.science.AnalyzeSingleSCIENCE > python\single.py
   java -classpath bin com.ibm.stdp.science.WriteNetwork
) else if %1==2 (
  java -classpath bin com.ibm.stdp.science.AnalyzeMirrorSCIENCE > python\mirror.py
) else if %1==3 (
  java -classpath bin com.ibm.stdp.science.AnalyzeEvolution > python\evolution.py
) else if %1==4 (
  java -classpath bin com.ibm.stdp.science.AnalyzeMusician > python\musician.py
)

rem
rem Draw figures
rem

pushd python
if %1==1 (
   python MakeSingleFigure.py
   python MakeSingleNetwork.py
) else if %1==2 (
   python MakeMirrorFigure.py
   python MakeMirrorStepsFigure.py
) else if %1==3 (
   python MakeEvolutionFigure.py 
) else if %1==4 (
   python MakeMusicalNote.py	
)
cd ..

pause
