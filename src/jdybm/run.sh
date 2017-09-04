#!/bin/bash
# (C) Copyright IBM Corp. 2017

# Run experiments
if [ $1 = "1" ]; then
    java -classpath lib/commons-math.jar:bin com.ibm.stdp.science.SingleExperiment3 SCIENCE 3 3 9
elif [ $1 = "2" ]; then
    java -classpath lib/commons-math.jar:bin com.ibm.stdp.science.MirrorExperiment3 SCIENCE 5 3 3 9
elif [ $1 = "3" ]; then
    java -classpath lib/commons-math.jar:bin com.ibm.stdp.science.EvolutionExperiment 3 3 9
elif [ $1 = "4" ]; then
    java -classpath lib/commons-math.jar:bin com.ibm.stdp.science.MusicianExperiment 3 3 9
fi

# Analyze results
if [ $1 = "1" ]; then
    java -classpath bin com.ibm.stdp.science.AnalyzeSingleSCIENCE > python/single.py
    java -classpath bin com.ibm.stdp.science.WriteNetwork
elif [ $1 = "2" ]; then
    java -classpath bin com.ibm.stdp.science.AnalyzeMirrorSCIENCE > python/mirror.py
elif [ $1 = "3" ]; then
    java -classpath bin com.ibm.stdp.science.AnalyzeEvolution > python/evolution.py
elif [ $1 = "4" ]; then
    java -classpath bin com.ibm.stdp.science.AnalyzeMusician > python/musician.py
fi

# Draw figures
cd python
if [ $1 = "1" ]; then
    python MakeSingleFigure.py
    python MakeSingleNetwork.py
elif [ $1 = "2" ]; then
    python MakeMirrorFigure.py
    python MakeMirrorStepsFigure.py
elif [ $1 = "3" ]; then
    python MakeEvolutionFigure.py
elif [ $1 = "4" ]; then
    python MakeMusicalNote.py
fi
cd ..
