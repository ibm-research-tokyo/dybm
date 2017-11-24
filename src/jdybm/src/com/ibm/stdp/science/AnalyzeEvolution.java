// (C) Copyright IBM Corp. 2015
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may
// not use this file except in compliance with the License. You may obtain
// a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// License for the specific language governing permissions and limitations
// under the License.


package com.ibm.stdp.science;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.util.Arrays;

import com.ibm.stdp.BitMapImage;
import com.ibm.stdp.Network;
import com.ibm.stdp.TimeSeries;

/*
 * Analysis of the results of the Evolution experiment
 * @author osogami
 */
public class AnalyzeEvolution {
	
    public static void main(String[] args) {
	String directory = "Results/Evolution/";
	
	BitMapImage image = new BitMapImage();
	final TimeSeries trainingData = image.getEvolution();
	
	System.out.println("x = dict()");
	
	File dir = new File(directory);
	File[] files = dir.listFiles();
	for(File file : files){
	    String filename = file.getName();
	    if(!filename.endsWith(".bin")){
		continue;
	    }
	    String number = filename.split("param")[0].split("NN")[1];
	    int n = Integer.parseInt(number);
	    String result = directory+filename;
	    
	    Network ann;
	    
	    try {
		FileInputStream fis = new FileInputStream(result);
		ObjectInputStream is = new ObjectInputStream(fis);
		ann = (Network) is.readObject();
		is.close();
	    } catch (IOException | ClassNotFoundException e) {
		e.printStackTrace();
		return;
	    }
	    
	    // prediction
	    ann.store();
	    TimeSeries prediction = ann.freeRun(trainingData.size()*2,0);		
	    System.out.println("x["+n+"] = "+prediction.toPython());
	    ann.restore();				
	}
    }
}
