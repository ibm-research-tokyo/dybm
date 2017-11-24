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

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Random;
import java.util.Set;

import org.apache.commons.math3.distribution.UniformIntegerDistribution;

/*
 * Time series
 * @author osogami
 */
public class TimeSeries {
    BinaryPattern[] series;
    int index;
    int indexBackup;
    int period_;
    int dimension_;
    
    /**
     * Prepare a time series of given period and dimension
     * @param period
     * @param dimension
     */
    public TimeSeries(int period, int dimension){
	period_ = period;
	dimension_ = dimension;
	series = new BinaryPattern[period];
	for(int i=0;i<period;i++){
	    series[i] = new BinaryPattern(dimension);
	}
	index = 0;
    }
    
    public TimeSeries(int[] data){
	// find the dimension of the time series
	dimension_ = 1;
	for(int x : data){
	    int dim = 1 + (int) Math.floor(Math.log(x)/Math.log(2));
	    if(dim>dimension_){
		dimension_ = dim;
	    }
	}
	
	// find the period of the time series
	period_ = data.length;
	
	// prepare a time series
	setTimeSeries(data);
	
	index = 0;
    }
    
    public TimeSeries(boolean[][] data) {
	period_ = data.length;
	if(data.length>0){
	    dimension_ = data[0].length;
	}
	setTimeSeries(data);
	index = 0;
    }
    
    public TimeSeries(String str){
	int[] data = StrToInt(str);
	
	// find the dimension of the time series
	dimension_ = 1;
	for(int x : data){
	    int dim = 1 + (int) Math.floor(Math.log(x)/Math.log(2));
	    if(dim>dimension_){
		dimension_ = dim;
	    }
	}
	
	// find the period of the time series
	period_ = data.length;
	
	// prepare a time series
	setTimeSeries(data);
	
	index = 0;
    }
    
    private int[] StrToInt(String str){
	List<Integer> intPattern = new ArrayList<Integer>();
	for(int i=0;i<str.length();i++){
	    int[] letter = BitMapFont.map.get(str.charAt(i));
	    for(int j=0;j<letter.length;j++){
		intPattern.add(letter[j]);
	    }
	}
	
	int[] array = new int[intPattern.size()];
	for(int i=0;i<array.length;i++){
	    array[i] = intPattern.get(i);
	}
	
	return array;
    }
    
    /**
     * Return the mirror image of the time series
     * @return
     */
    public TimeSeries mirror(){
	TimeSeries ts = new TimeSeries(period_,dimension_);
	// always start with a space
	ts.set(0, this.series[0]);
	for(int i=0;i<period_-1;i++){
	    ts.set(i+1, this.series[period_-i-1]);
	}
	return ts;
    }
    
    /**
     * Return the current pattern, and increment the index
     * @return
     */
    public BinaryPattern getPattern(){
	BinaryPattern data = series[index];
	index = (index+1) % period_;
	return data;
    }
    
    public BinaryPattern getPattern(int index){
	return series[index];
    }
    
    /**
     * Set the time series from integer representation
     * @param data
     */
    public void setTimeSeries(int[] data){
	series = new BinaryPattern[period_];
	for(int i=0;i<data.length;i++){
	    series[i] = new BinaryPattern(dimension_,data[i]);
	}
    }
    
    public void setTimeSeries(boolean[][] data){
	series = new BinaryPattern[period_];
	for(int i=0;i<data.length;i++){
	    series[i] = new BinaryPattern(data[i]);
	}
    }
    
    public TimeSeries duplicate(int n){
	TimeSeries ts = new TimeSeries(period_,dimension_*n);
	for(int i=0;i<period_;i++){
	    BinaryPattern bp = series[i].duplicate(n);
	    ts.set(i,bp);
	}
	return ts;
    }
    
    public void set(int i, BinaryPattern bp){
	series[i] = bp;
    }
    
    public int size() {
	return series.length;
    }
    
    public void store() {
	indexBackup = index;
    }
    
    public void restore() {
	index = indexBackup;
    }
    
    public void setIndex(int idx) {
	index = idx;
    }
    
    public int getDimension() {
	return dimension_;
    }
    
    public int getPeriod() {
	return period_;
    }
    
    public String toString(){
	String str = "";
	for(int i=0;i<period_;i++){
	    str += series[i].toString();
	    str += "\n";
	}
	return str;
    }
    
    public String toPython(){
	String str = "[";
	for(int i=0;i<period_;i++){
	    str += series[i].toPython();
	    str += ",";
	}
	str += "]";
	return str;
    }
    
    public boolean equals(Object o){
	if(o==this) return true;
	if(o==null) return false;
	if(!(o instanceof TimeSeries)) return false;
	
	TimeSeries ts = (TimeSeries) o;
	if(ts.size() != this.size()) return false;
	if(ts.getDimension() != this.getDimension()) return false;
	if(ts.getPeriod() != this.getPeriod()) return false;
	for(int i=0;i<size();i++){
	    if(!ts.getPattern(i).equals(this.getPattern(i))){
		return false;
	    }
	}
	return true;
    }
    
    public int hashCode(){
	int code = 0;
	for(BinaryPattern bp : series){
	    code += bp.hashCode();
	}
	return code;
    }
    
    public void randomFlip(int i) {
	long seed = 0L;
	Random random = new Random(seed);
	Set<Integer> randomNumbers = new HashSet<Integer>();
	while(randomNumbers.size()<i){
	    int r = random.nextInt(dimension_*period_);
	    randomNumbers.add(r);
	}
	for(int r : randomNumbers){
	    int d = r / period_;
	    int p = r % period_;
	    series[p].flip(d);
	}
    }
}
