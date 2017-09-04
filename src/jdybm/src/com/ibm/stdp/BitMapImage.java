// (C) Copyright IBM Corp. 2015

package com.ibm.stdp;

/*
 * Bitmap pattern of human evolution
 * @author osogami
 */
public class BitMapImage {
    
    static int[][] first = {
	{},
	{1,0},
	{6,5,2,0},
	{7,6,5,4,3,2},
	{8,7,6,5,4},
	{9,8,7,6,4,3,2,1,0},
	{10,9,8,7,6},
	{10,9,8,7,6,5,4},
	{11,10,9,8,7,6,5,4},
	{11,10,9,3,2,1},
	{11,10,9}
    };
    
    static int[][] second = {
	{},
	{2,1,0},
	{2},
	{10,9,8,7,6,5,2},
	{12,11,10,9,8,7,6,5,4,3},
	{13,12,11,10,9,8,7,6},
	{15,14,13,12,11,10,6,5,2,1,0},
	{15,14,13,10,4,3},
	{15,14,13,9,},
	{8}
    };
    
    static int[][] third = {
	{},
	{1},
	{3,2,0},
	{4,3,0},
	{13,12,11,10,9,8,7,6,5,4},
	{15,14,13,12,11,10,9,8,7,6},
	{17,16,15,14,13,12,11,10,9,7,6,5,4,3},
	{17,16,15,13,12,11,10,5,4,3,2,1,0},
	{17,16,15,10,9,8,7,0},
	{0}
    };
    
    static int[][] fourth = {
	{},
	{2},
	{3,2,1,0},
	{4,3},
	{5,4},
	{19,18,17,16,15,14,13,12,11,10,9,8,7,6,5},
	{19,18,17,15,14,13,12,11,10,9,8,7,6},
	{19,18,17,14,13,12,11,10,8,7,6,5,4,3,2,1,0},
	{11,10,0},
	{9,8,0}	
    };
    
    static int dimension = 20;
    
    public TimeSeries getEvolution(){
	boolean[][] data = new boolean[first.length+second.length+third.length+fourth.length][dimension];
	int row = 0;
	for(int i=0;i<first.length;i++){
	    for(int j=0;j<dimension;j++){
		data[row][j] = false;
	    }
	    for(int j=0;j<first[i].length;j++){
		data[row][first[i][j]] = true;
	    }
	    row++;
	}
	
	for(int i=0;i<second.length;i++){
	    for(int j=0;j<dimension;j++){
		data[row][j] = false;
	    }
	    for(int j=0;j<second[i].length;j++){
		data[row][second[i][j]] = true;
	    }
	    row++;
	}
	
	for(int i=0;i<third.length;i++){
	    for(int j=0;j<dimension;j++){
		data[row][j] = false;
	    }
	    for(int j=0;j<third[i].length;j++){
		data[row][third[i][j]] = true;
	    }
	    row++;
	}
	
	for(int i=0;i<fourth.length;i++){
	    for(int j=0;j<dimension;j++){
		data[row][j] = false;
	    }
	    for(int j=0;j<fourth[i].length;j++){
		data[row][fourth[i][j]] = true;
	    }
	    row++;
	}
	
	TimeSeries ts = new TimeSeries(data);
	return ts;
    }
}
