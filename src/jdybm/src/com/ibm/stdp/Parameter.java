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

package com.ibm.stdp;

import java.io.Serializable;
import java.util.Arrays;

/**
 * Global parameters of a DyBM
 * @author osogami
 *
 */
public class Parameter implements Serializable {

    private static final long serialVersionUID = 1L;
    
    public int nRatesPositive;// = ratePositive.length;
    public int nRatesNegative;// = rateNegative.length;
    public double[] ratePositive;
    public double[] rateNegative;
    
    public int minDelay;
    public int maxDelay;
    
    public Parameter(int n, int m){
	nRatesNegative = n;
	rateNegative = new double[n];
	for(int i=0;i<n;i++){
	    rateNegative[i] = (i + 1.0) / (n + 1.0);
	}
	System.out.println("Setting mu = "+Arrays.toString(rateNegative));
	
	nRatesPositive = m;
	ratePositive = new double[m];
	for(int i=0;i<m;i++){
	    ratePositive[i] = (i + 1.0) / (m + 1.0);
	}
	System.out.println("Setting lambda = "+Arrays.toString(ratePositive));
	
	minDelay = 1;
	maxDelay = 8;
    }
	
    public class Bias {
	public static final double mean = 0.0;
	public static final double std = 0.1;
    }
    
    public class Weight {
	public class U {
	    public static final double mean = 0.0;
	    public static final double std = 0.1;
	}
	public class V {
	    public static final double mean = 0.0;
	    public static final double std = 0.1;			
	}
    }
    
    public int getMinDelay() {
	return minDelay;
    }
    
    public int getMaxDelay() {
	return maxDelay;
    }
    
    public void setMaxDelay(int d){
	maxDelay = d;
    }
    
    // learning rate is normalized by the temperature
    static double learningRate = 1;
    
    static double temperature = 1;
    
    public static final boolean keepHistory = false;
}
