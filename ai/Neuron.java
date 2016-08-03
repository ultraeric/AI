package ai;

import java.util.ArrayList;

import misc.DoubleFunction;

/**
 * A single node in a neural network.
 * Used to create simple neural networks.
 * 
 * @author ericemily
 *
 */
public class Neuron {
//	For utility, identifying neuron, debugging, etc.
	private int layer;
	private int number;
	
	private int batchSize = 1;
	private double learningRate = 0.3;
	private double[] forwardPropInput = new double[batchSize];
	private double[] backPropInput = new double[batchSize];
	
//	The backPropDerivError is the partial derivative of the error function with respect to
//	this neuron's activation value.
	private double[] backPropDerivError = new double[batchSize];
	
	private double[] activationVal = new double[batchSize];
	private double[] expected = new double[batchSize];
	private double momentum = 0.1;
	private double outputError = 0.0;
	private DoubleFunction activation = 
			new DoubleFunction((x) -> {return 1.0 / (1.0 + Math.pow(Math.E, - x));});
	private ArrayList<Neuron> children = new ArrayList<Neuron>();
	private ArrayList<Double> childWeights = new ArrayList<Double>();
	private ArrayList<Neuron> parents = new ArrayList<Neuron>();
	private ArrayList<Double> parentWeights = new ArrayList<Double>();
	private ArrayList<Double> childWeightAdjustors = new ArrayList<Double>();
	private ArrayList<Double> parentWeightAdjustors = new ArrayList<Double>();
	private ArrayList<Double> previousChildWeightAdjustors = new ArrayList<Double>();
	private ArrayList<Double> previousParentWeightAdjustors = new ArrayList<Double>();

	/**
	 * Constructor.
	 * 
	 * @param layer			The layer that this neuron is contained in.
	 * @param number		The index of the neuron in its layer.
	 * @param learningRate	The learning rate used in weight adjustment. 
	 * 							High values respond more extremely to the gradient descent.
	 * @param momentum		The momentum constant, which helps avoid local minima in gradient descent.
	 * 							High values result in a larger consideration of the direction of the 
	 * 							previous iteration of gradient descent.
	 */
	
	public Neuron(int layer, int number, double learningRate, double momentum){
		this(layer, number, learningRate, momentum, 1);
	}
	public Neuron(int layer, int number, double learningRate, double momentum, int batchSize){
		this.batchSize = batchSize;
		refresh();
		this.layer = layer;
		this.number = number;
		this.learningRate = learningRate;
		this.momentum = momentum;
	}
	
	/**
	 * Resets iteration-dependent fields. 
	 */
	public void refresh(){
		outputError = 0.0;
		forwardPropInput = new double[batchSize];
		backPropDerivError = new double[batchSize];
		activationVal = new double[batchSize];
		backPropInput = new double[batchSize];
		expected = new double[batchSize];
		childWeightAdjustors = new ArrayList<Double>();
		parentWeightAdjustors = new ArrayList<Double>();
		previousChildWeightAdjustors = new ArrayList<Double>();
		previousParentWeightAdjustors = new ArrayList<Double>();
	}
	public void setBatchSize(int n){
		batchSize = n;
		refresh();
	}
	public int getNumber(){
		return number;
	}
	public ArrayList<Double> getChildWeights(){
		return childWeights;
	}
	public double[] getActivation(){
		return activationVal;
	}
	public void addToForwardInput(double[] d){
		for(int i = 0; i < batchSize; i++){
			forwardPropInput[i] += d[i];
		}
	}
	public void addToBackInput(double[] d){
		for(int i = 0; i < batchSize; i++){
			backPropInput[i] += d[i];
		}
	}
	public Neuron addChildren(ArrayList<Neuron> children, ArrayList<Double> childWeights){
		this.children = children;
		this.childWeights = childWeights;
		return this;
	}
	public Neuron addChildren(Neuron[] children, double[] childWeights){
		for(Neuron n: children){ this.children.add(n); }
		for(double d: childWeights){ this.childWeights.add(d); }
		return this;
	}
	public Neuron addChild(Neuron child, double childWeight){
		this.children.add(child);
		this.childWeights.add(childWeight);
//		//System.out.println("Layer " + layer + " number " + number + " had a child added");
		return this;
	}
	public Neuron addParents(ArrayList<Neuron> parents, ArrayList<Double> parentWeights){
		this.parents = parents;
		this.parentWeights = parentWeights;
		return this;
	}
	public Neuron addParents(Neuron[] parents, double[] parentWeights){
		for(Neuron n: parents){ this.parents.add(n); }
		for(double d: parentWeights){ this.parentWeights.add(d); }
		return this;
	}
	public Neuron addParent(Neuron parent, double parentWeight){
		this.parents.add(parent);
		this.parentWeights.add(parentWeight);
		return this;
	}
	
	/**
	 * Calculates this neuron's activation value and propagates its values to its children using their
	 * corresponding weights.
	 * 
	 * @return	the activation value of this neuron.
	 */
	public double[] forwardProp(){
//		//System.out.println(forwardPropInput);
		//System.out.println("layer " + layer + " number " + number + " has forwardPropInput " + forwardPropInput);
		for(int i = 0; i < batchSize; i++){
			activationVal[i] = activation.eval(forwardPropInput[i]);
			//System.out.println("Activation value for layer " + layer + " number " + number + " is " + activationVal);
		}
		if(childWeights.size() == 0){
			return activationVal;
		}else{
			for(int i = 0; i < children.size(); i++){
				double[] valuesPassedToChild = new double[batchSize];
				for(int i2 = 0; i2 < batchSize; i2++){
					valuesPassedToChild[i2] = activationVal[i2] * childWeights.get(i);
				}
				children.get(i).addToForwardInput(valuesPassedToChild);
				//System.out.println("layer " + layer + " number " + number + " transferred " +
//										activationVal * childWeights.get(i) + " to layer " + (layer + 1) +
//										" node " + i);
			}
			return activationVal;
		}
	}
	
	/**
	 * If this neuron is an output neuron, returns the error of this single neuron.
	 * 
	 * @param expected		The expected value of the training/test case.
	 * @return				The error of the neuron.
	 * @throws Exception
	 */
	public double getError(){
		return outputError;
	}
	/**
	 * Returns the total least squares error of this batch.
	 * 
	 * @param expected
	 * @return
	 * @throws Exception
	 */
	public double getError(double[] expected) throws Exception{
		if(childWeights.size() == 0){
			this.expected = expected;
			outputError = 0;
			for(int i = 0; i < batchSize; i++){
				outputError += 0.5 * Math.pow(expected[i] - activationVal[i], 2.0);
			}
			return outputError;
		}else{
			throw new Exception("Error cannot be extracted: this is not an output node.");
		}
	}
//	public double getResults(double expected) throws Exception{
//		return 0.5 * Math.pow(Math.round(activationVal) - expected, 2.0);
//	}
	
	/**
	 * Gets the partial derivative of the error with respect to this activation value.
	 * @return	The partial derivative of the error with respect to this activation value.
	 */
	public double[] getDerivError(){
		return backPropDerivError;
	}
	
	/**
	 * Backpropagation step. Passes the gradient of the error with respect to this neuron's activation value
	 * back to its parents.
	 * 
	 * @return 	The partial derivative of the error with respect to this activation value.
	 */
	public double[] backProp(){
		backPropDerivError = new double[batchSize];
		if(childWeights.size() == 0){
			for(int i = 0; i < batchSize; i++){
				backPropDerivError[i] = -(expected[i] - activationVal[i]);
			}
		}else{
			backPropDerivError = backPropInput;
		}
		//System.out.println("backPropDerivError for layer " + layer + " number " + number + " is " + backPropDerivError);
		double[] backPropNeuronError = new double[batchSize];
		for(int i = 0; i < batchSize; i++){
			backPropNeuronError[i] = backPropDerivError[i] * activationVal[i] * (1.0 - activationVal[i]);
		}
		for(int i = 0; i < parents.size(); i++){
			double[] valuesPassedToParent = new double[batchSize];
			for(int i2 = 0; i2 < batchSize; i2++){
				valuesPassedToParent[i2] = parentWeights.get(i) * backPropNeuronError[i2];
			}
			parents.get(i).addToBackInput(valuesPassedToParent);
			//System.out.println(parentWeights.get(i) * backPropNeuronError + " sent to layer " + (layer - 1) + " number " + i);
		}
		return backPropDerivError;
	}
	
	/**
	 * Calculates and stores how much each weight should be adjusted for the next iteration of learning.
	 * Note that adjustments are normalized.
	 */
	public void calculateAdjustments(){
		//The sum of the adjustments that each training item in the batch provides to the weights
		//of the parents
		double totalAdjustment = 0.0;
		for(int i = 0; i < parents.size(); i++){
			for(int i2 = 0; i2 < batchSize; i2++){
				totalAdjustment += 
//						Multiply with single parent's activation value to account for fact that total
//						input is affected by only the weight of the single parent node that feeds into
//						said weight. In other words, a single forward propagation step includes
//						output of previous neuron -> weight -> input of next neuron -> activation value.
//						You must find the partial derivative of each step, backwards, in order to
//						backpropagate the error. 
						parents.get(i).getActivation()[i2] *
						activationVal[i2] * (1.0 - activationVal[i2]) 
						* backPropDerivError[i2] * (-learningRate);
			}
			//The normalized adjustments to the weights
			double normalizedAdjustment = totalAdjustment / batchSize;
			if(previousParentWeightAdjustors.size() != 0) { parentWeightAdjustors.add(normalizedAdjustment + previousParentWeightAdjustors.get(i) * momentum); }
			else { parentWeightAdjustors.add(normalizedAdjustment); }
		}
		//totalAdjustment is now reused for the children
		totalAdjustment = 0.0;
		for(int i = 0; i < children.size(); i++){
			for(int i2 = 0; i2 < batchSize; i2++){
				totalAdjustment += activationVal[i2] * children.get(i).getActivation()[i2] * (1.0 - children.get(i).getActivation()[i2])
						* children.get(i).getDerivError()[i2] * (-learningRate);
			}
			//The normalized adjustments to the weights
			double normalizedAdjustment = totalAdjustment / batchSize;
			if(previousChildWeightAdjustors.size() != 0){ childWeightAdjustors.add(normalizedAdjustment + previousChildWeightAdjustors.get(i) * momentum); }
			else { childWeightAdjustors.add(normalizedAdjustment); }
		}
//		//System.out.println("Calculating adjustments");
	}
	
	/**
	 * Reads the precalculated weight adjustment values and adjusts the weights accordingly. Also stores
	 * the adjustments to the weights for the momentum term in the next iteration.
	 */
	public void adjust(){
		previousParentWeightAdjustors = (ArrayList<Double>) parentWeightAdjustors.clone();
		previousChildWeightAdjustors = (ArrayList<Double>) childWeightAdjustors.clone();
		for(int i = 0; i < parents.size(); i++){
			double oldWeight = parentWeights.get(i);
			parentWeights.set(i, oldWeight + parentWeightAdjustors.get(i));
		}
		for(int i = 0; i < children.size(); i++){
			double oldWeight = childWeights.get(i);
			childWeights.set(i, oldWeight + childWeightAdjustors.get(i));
			//System.out.println("New weight from layer " + layer + " number " + number + " to layer " + 
//								(layer + 1) + " number " + i + " is " + childWeights.get(i));
		}
//		//System.out.println("Adjusted");
	}
}
