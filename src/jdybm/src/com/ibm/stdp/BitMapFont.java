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

import java.util.HashMap;

/*
 * Bitmap patterns of alphabets
 * @author osogami
 */
public class BitMapFont {
	private final static int[] A = {0,63,72,72,63};
	private final static int[] B = {0,127,73,73,54};
	private final static int[] C = {0,62,65,65,34};
	private final static int[] E = {0,127,73,73,65};
	private final static int[] I = {0,65,127,65};
	private final static int[] L = {0,127,1,1,1};
	private final static int[] M = {0,127,16,8,16,127};		
	private final static int[] N = {0,127,16,8,4,127};
	private final static int[] P = {0,127,72,72,48};
	private final static int[] R = {0,127,76,74,49};
	private final static int[] S = {0,49,73,73,70};
	private final static int[] T = {0,64,64,127,64,64};
	private final static int[] U = {0,126,1,1,126};
	
	private final static int[] i = {0,7175,7175,7175,8191,8191,8191,8191,7175,7175,7175};
	private final static int[] b = {0,7175,7175,7175,8191,8191,8191,8191,7399,7399,7399,7399,8191,8191,4030,4030,1820};
	private final static int[] m = {0,7175,7175,7175,8191,8191,8191,8191,8064,4064,1016,254,127,254,1016,4064,8064,8191,8191,8191,8191,7175,7175,7175};
	
	public final static HashMap<Character,int[]> map = new HashMap<Character,int[]>();
	static{
		map.put('A',A);
		map.put('B',B);
		map.put('C',C);
		map.put('E',E);
		map.put('I',I);
		map.put('L',L);
		map.put('M',M);
		map.put('N',N);
		map.put('P',P);
		map.put('S',S);
		map.put('T',T);
		map.put('U',U);
		map.put('R',R);

		map.put('i',i);
		map.put('b',b);
		map.put('m',m);
	}
}
