package ai;

import misc.Complex;
import misc.FFT;

public class Main {
	public static void main(String[] args) throws Exception{
		
		Brain b = new Brain(1, .3, .9, 2, 2, 2, 1);
		double[][] input = new double[1000000][2];
		double[][] output = new double[1000000][1];
		for(int i = 0; i < 1000000; i++){
			input[i][0] = Math.random();
			input[i][1] = 1;
			double x = input[i][0];
			if(i % 2 == 0){
				output[i][0] = x / 2 + 0.1 + 0.1;
			}else{
				output[i][0] = x / 2 + 0.1 - 0.1;
			}
		}
		double[][] testInput = new double[200][2];
		double[][] testOutput = new double[200][1];
		for(int i = 0; i < 200; i++){
			testInput[i][0] = Math.random();
			testInput[i][1] = 1;
			double x = testInput[i][0];
			testOutput[i][0] = x / 2 + 0.1;
		}
		b.setTrainCases(input);
		b.setTrainResults(output);
		b.train();
		b.setTestCases(testInput);
		b.setTestResults(testOutput);
		System.out.println("Average error is " + b.getErrorPercentage() + "%");
	}
}
