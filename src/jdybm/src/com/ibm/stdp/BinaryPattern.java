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

import java.util.Random;

/*
 * Binary pattern
 * @author osogami
 */
public class BinaryPattern {
    boolean[] pattern;
    int dimension_;
    
    public BinaryPattern(int dimension) {
	pattern = new boolean[dimension];
	dimension_ = dimension;
    }
    
    public BinaryPattern(int dimension, int data) {
	pattern = new boolean[dimension];
	dimension_ = dimension;
	setPattern(data);
    }
    
    public BinaryPattern(boolean[] data){
	pattern = data;
	dimension_ = data.length;
    }
    
    /** 
     * Set the pattern from integer representation
     * @param n
     */
    public void setPattern(int n) {
	boolean[] data = getBoolean(n);
	setPattern(data);		
    }
    
    /**
     * Set the pattern 
     * @param data
     */
    public void setPattern(boolean[] data) {
	pattern = data;
    }
    
    /**
     * Convert an integer to a boolean representation
     * @param n
     */
    private boolean[] getBoolean(int n){
	boolean[] data = new boolean[dimension_];
	for(int i=0;i<dimension_;i++){
	    data[i] = (n%2 == 1);
	    n /= 2;
	}
	return data;
    }
    
    /**
     * Duplicate the patterns by n times to make new pattern
     * @param n
     * @return
     */
    public BinaryPattern duplicate(int n) {
	boolean[] data = new boolean[dimension_*n];
	int k=0;
	for(int i=0;i<dimension_;i++){
	    for(int j=0;j<n;j++){
		data[k++] = pattern[i];
	    }
	}
	BinaryPattern bp = new BinaryPattern(data);
	return bp;
    }
    
    public String toString(){
	String str = "";
	for(int j=0;j<dimension_;j++){
	    if(pattern[j]){
		str += "X";
	    }else{
		str += ".";
	    }
	}
	return str;
    }
    
    public String toPython() {
	String str = "[";
	for(int j=dimension_-1;j>-1;j--){
	    if(pattern[j]){
		str += "1";
	    }else{
		str += "0";
	    }
	    str += ",";
	}
	str += "]";
	return str;
    }	
    
    public boolean getValue(int i) {
	return pattern[i];
    }
    
    public int getDimension(){
	return dimension_;
    }
    
    public boolean equals(Object o){
	if(o==this) return true;
	if(o==null) return false;
	if(!(o instanceof BinaryPattern)) return false;
	
	BinaryPattern bp = (BinaryPattern) o;
	if(bp.getDimension() != this.getDimension()) return false;
	for(int i=0;i<getDimension();i++){
	    if(bp.getValue(i)!=this.getValue(i)){		
		return false;
	    }
	}
	return true;
    }
    
    public int hashCode(){
	int code = 0;
	for(boolean b : pattern){
	    if(b){
		code += 1;
	    }
	}
	return code;
    }
    
    public void randomFlip(double p, long seed) {
	Random random = new Random(seed);
	for(int i=0;i<pattern.length;i++){
	    boolean flip = random.nextDouble() < p;
	    if(flip){
		pattern[i] = (pattern[i]==false);
	    }
	}
    }
    
    public void flip(int d) {
	pattern[d] = (pattern[d]==false);
    }
}
