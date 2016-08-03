package ai;

import java.util.ArrayList;

import misc.DoubleFunction;

//Note: to test, add training cases + results, then run training method, then run test cases + results,
//then run getError method.
public class Brain {
	private int[] numNeurons;
	private int totalNeurons;
	private ArrayList<Neuron>[] neurons;
	private double learningRate = 0.1;
	private double momentum = 0.1;
	
	//Note: first array represents the number of the individual training case in the batch,
	//second array is the elements of the training cases.
	private double[][] trainCases;
	private double[][] trainResults;
	private double[][] testCases;
	private double[][] testResults;
	private int batchSize = 1;
	
	private int trainCase = 0;

	private int step = 0;
	
	public Brain(double learningRate, double momentum, int... num){
		this(1, learningRate, momentum, num);
	}
	public Brain(int batchSize, double learningRate, double momentum, int... num){
		this.numNeurons = new int[num.length];
		this.neurons = new ArrayList[num.length];
		this.momentum = momentum;
		this.learningRate = learningRate;
		this.batchSize = batchSize;
		totalNeurons = 0;
		for(int i = 0; i < neurons.length; i++){
			neurons[i] = new ArrayList<Neuron>();
			numNeurons[i] = num[i];
			for(int i2 = 0; i2 < num[i]; i2++){
				neurons[i].add(new Neuron(i, i2, learningRate, momentum, batchSize));
			}
			totalNeurons += num[i];
		}
		reset(num);
	}
	public ArrayList<Neuron>[] getNeurons(){
		return neurons;
	}
	public double[] getCurrentTrainInput(){
		return trainCases[trainCase];
	}
	public double[] getCurrentTrainResult(){
		return trainResults[trainCase];
	}
	public Neuron nextStep() throws Exception{
		Neuron n = null;
		if(step >= 3* totalNeurons){
			refresh();
			trainCase++;
			
			return nextStep();
		}
		
		if(step >= totalNeurons && step < 2 * totalNeurons){
			int num = (step) % totalNeurons;
			n = getNeuron(num, false);
			n.backProp();
		}else if(step >= 2 * totalNeurons && step < 3 * totalNeurons){
			int num = (step) % totalNeurons;
			n = getNeuron(num, true);
			n.calculateAdjustments();
			n.adjust();
		}else{
			if(step == 0){
				initializeTrainCase(trainCase);
			}
			n = getNeuron(step, true);
			n.forwardProp();
			if(neurons[neurons.length - 1].contains(n)){
				n.getError(getNthBatchIthNeuron(trainCase, n.getNumber(), trainResults));
			}
		}
		step++;
		return n;
	}
	private Neuron getNeuron(int number, boolean forward){
		ArrayList<Neuron>[] n;
		
		if(forward == true) n = neurons;
		else n = (ArrayList<Neuron>[]) reverse(neurons);
		
		int c = 0;
		for(int i = 0; i < n.length; i++){
			for(int i2 = 0; i2 < n[i].size(); i2++){
				if(c == number) {
					return n[i].get(i2);
				}
				c++;
			}
		}
		return null;
	}
	private ArrayList<?>[] reverse(ArrayList<?>[] aL){
		ArrayList<?>[] returnAL = new ArrayList<?>[aL.length];
		for(int i = aL.length - 1; i >= 0; i--){
			returnAL[aL.length - 1 - i] = 
					aL[i];
		}
		return returnAL;
	}
	
	public int[] getStructure(){
		int[] num = new int[neurons.length];
		for(int i = 0; i < neurons.length; i++){
			num[i] = neurons[i].size();
		}
		return num;
	}
	protected double generateWeight(){
		return Math.random();
	}
	public void setTrainCases(double[][] trainCases){
		this.trainCases = trainCases;
	}
	public void setTrainResults(double[][] trainResults){
		this.trainResults = trainResults;
	}
	public void train() throws Exception{
		for(int i = 0; i < Math.floor(trainCases.length / batchSize); i++){
			initializeTrainCase(i);
			forwardProp();
			checkTrainCase(i);
			backProp();
			adjustWeights();
			refresh();
		}
	}
	public void initializeTrainCase(int n){
		for(int i = 0; i < neurons[0].size(); i++){
			neurons[0].get(i).addToForwardInput(getNthBatchIthNeuron(n, i, trainCases));
		}
	}
	public void initializeTestCase(int n){
		for(int i = 0; i < neurons[0].size(); i++){
			neurons[0].get(i).addToForwardInput(getNthBatchIthNeuron(n, i, testCases));
		}
	}
	public void forwardProp(){
		for(int i = 0; i < neurons.length; i++){
			for(Neuron n: neurons[i]){
				n.forwardProp();
			}
		}
	}
	public void checkTrainCase(int n) throws Exception{
		for(int i = 0; i < neurons[neurons.length - 1].size(); i++){
			neurons[neurons.length - 1].get(i).getError(getNthBatchIthNeuron(n, i, trainResults));
		}
	}
	public void checkTestCase(int n) throws Exception{
		for(int i = 0; i < neurons[neurons.length - 1].size(); i++){
			neurons[neurons.length - 1].get(i).getError(getNthBatchIthNeuron(n, i, testResults));
		}
	}
	public void backProp(){
		for(int i = neurons.length - 1; i >= 0; i--){
			for(Neuron n: neurons[i]){
				n.backProp();
			}
		}
	}
	public void adjustWeights(){
		for(int i = 0; i < neurons.length; i++){
			for(Neuron n: neurons[i]){
				n.calculateAdjustments();
				n.adjust();
			}
		}
	}
	public void refresh(){
		for(int i = 0; i < neurons.length; i++){
			for(Neuron n: neurons[i]){
				n.refresh();
			}
		}
		step = 0;
	}
	public void reset(int... num){
		double[][] tempWeights = new double[0][0];
		for(int i = 0; i < neurons.length; i++){
			if(i >= 1){
				for(int i2 = 0; i2 < neurons[i].size(); i2++){
					for(int i3 = 0; i3 < neurons[i - 1].size(); i3++){
						neurons[i].get(i2).addParent(neurons[i - 1].get(i3), tempWeights[i3][i2]);
//						System.out.println("Layer " + i + " node " + i2 + " layer " + (i-1) + " node "
//											+ i3 + " have weights " + tempWeights[i3][i2]);
					}
				}
			}
			if(i <= neurons.length - 2){
				tempWeights = new double[num[i]][num[i + 1]];
				for(int i2 = 0; i2 < neurons[i].size(); i2++){
					for(int i3 = 0; i3 < neurons[i + 1].size(); i3++){
						tempWeights[i2][i3] = generateWeight();
						neurons[i].get(i2).addChild(neurons[i + 1].get(i3), tempWeights[i2][i3]);
					}
				}
			}
		}
		refresh();
	}
	public void reset(){
		reset(getStructure());
	}
	public void setTestCases(double[][] testCases){
		this.testCases = testCases;
	}
	public void setTestResults(double[][] testResults){
		this.testResults = testResults;
	}
	public double getErrorPercentage() throws Exception{
		double totalError = 0;
		refresh();
		for(int i = 0; i < Math.floor(testCases.length / batchSize); i++){
			initializeTestCase(i);
			forwardProp();
			for(int i2 = 0; i2 < neurons[neurons.length - 1].size(); i2++){
//				System.out.println(neurons[neurons.length - 1].get(i2).getError(testResults[i][i2]));
				totalError += neurons[neurons.length - 1].get(i2).getError(getNthBatchIthNeuron(i, i2, testResults));
			}
			refresh();
		}
//		System.out.println(testCases.length);
		return totalError * 100 / testCases.length;
	}
	/**
	 * 0-indexed. Returns the array associated with the batch for a single neuron (array of values
	 * corresponding to a single neuron)
	 * 
	 * @return
	 */
	public double[] getNthBatchIthNeuron(int n, int i, double[][] m){
		double[] returnBatch = new double[batchSize];
		for(int i2 = 0; i2 < batchSize; i2 ++){
			returnBatch[i2] = m[n * batchSize + i2][i];
		}
		return returnBatch;
	}
}
