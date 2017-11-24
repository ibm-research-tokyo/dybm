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
import java.util.logging.Logger;

/*
 * FIFO queue with integer representation
 * @author osogami
 */
public class IntFIFO implements FIFO, Serializable {
    
    /**
     * 
     */
    private static final long serialVersionUID = 1L;
    
    private static final Logger logger = Logger.getLogger("com.ibm.stdp.IntFIFO");
    
    /**
     * The length of the FIFO queue
     */
    private int length;
    
    /**
     * The value of the integer representation of the FIFO queue
     */
    private int value;
    
    /**
     * The space to store the backup of the value
     */
    private int valueBackup;
    
    /**
     * Pre-computed value of beta eligibility traces
     */
    private double[][] beta;
    
    private double valueToOffer;
    
    private double[] rate;
    
    /**
     * Constracter of the FIFO queue with length n
     * @param n
     */
    public IntFIFO(int n, double[] mu) {
	//
	// set the length
	//
	length = n;
	
	rate = mu;
	
	//
	// Pre-compute the values of beta eligibility traces
	//
	// N: the number of integer representation of a binary queue of length
	int N = (int) Math.pow(2, length);
	beta = new double[rate.length][N];
	for(int i=0;i<N;i++){
	    // integer to binary queue
	    BinaryFIFO queue = getBinaryQueue(i);
	    
	    // binary queue to beta
	    for(int k=0;k<rate.length;k++){
		beta[k][i] = queue.getBeta(k);
	    }
	}
	
	valueToOffer = Math.pow(2, length-1);
	
	init();
    }
    
    @Override
    public void init() {
	value = 0;
    }
    
    @Override
    public boolean offer(boolean x) {
	if(length==0 && CONNECT_ZERO){
	    return x;
	}
	
	// queue.poll
	int z = value % 2; 
	value /= 2;
	
	// queue.offer
	if(x){
	    value += valueToOffer;
	}
	
	return z==1;
    }
    
    @Override
    public double getBeta(int k){
	return beta[k][value];
    }
    
    @Override
    public void store() {
	valueBackup = value;
    }
    
    @Override
    public void restore() {
	value = valueBackup;
    }
    
    // set the values of the FIFO queue
    /**
     * Get the representation of the BinaryFIFO
     * corresponding to the IntFIFO of representation, m
     * @param m
     * @return BinaryFIFO
     */
    private BinaryFIFO getBinaryQueue(int m){
	BinaryFIFO queue = new BinaryFIFO(length,rate);
	for(int i=0;i<length;i++){
	    int b = m % 2;
	    boolean binary = (b==1);
	    queue.offer(binary);
	    m /= 2;
	}
	
	return queue;
    }
    
    @Override
    public String toString() {
	BinaryFIFO queue = getBinaryQueue(value);
	return queue.toString();
    }
}
