rem (C) Copyright IBM Corp. 2015
rem
rem Licensed under the Apache License, Version 2.0 (the "License"); you may
rem not use this file except in compliance with the License. You may obtain
rem a copy of the License at
rem
rem      http://www.apache.org/licenses/LICENSE-2.0
rem
rem Unless required by applicable law or agreed to in writing, software
rem distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
rem WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
rem License for the specific language governing permissions and limitations
rem under the License.

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
