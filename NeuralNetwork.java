import java.util.ArrayList;
import java.util.List;

public class NeuralNetwork {
	Matrix inputHiddenWeights; // for the weights between the input and hidden layer.
	Matrix hiddenOutputWeights; // for the weights between the hidden and output layer.
	Matrix hiddenBias; // bias matrix for the hidden layer.
	Matrix outputBias; // bias matrix for the output layer.
	
	double learningRate = 0.1; // the learning rate during weight optimization.
	
	public NeuralNetwork(int input, int hidden, int output) {
		this.inputHiddenWeights = new Matrix(hidden, input);
		this.hiddenOutputWeights = new Matrix(output, hidden);
		
		this.hiddenBias = new Matrix(hidden,1);
		this.outputBias = new Matrix(output,1);
	}
	
	
	public ArrayList<Double> predict(double[] pred) {
		Matrix input = Matrix.fromArray(pred);
		Matrix hidden = Matrix.multiply(inputHiddenWeights, input);
		hidden.add(hiddenBias);
		hidden.sigmoid();
		
		Matrix output = Matrix.multiply(hiddenOutputWeights, hidden);
		output.add(outputBias);
		output.sigmoid();
		
		return output.toArray();
	}
	
	public void train(double[] x, double[] y) {
		Matrix input = Matrix.fromArray(x);
		Matrix hidden = Matrix.multiply(inputHiddenWeights, input);
		hidden.add(hiddenBias);
		hidden.sigmoid();
		
		Matrix output = Matrix.multiply(hiddenOutputWeights, hidden);
		output.add(outputBias);
		output.sigmoid();
		
		Matrix target = Matrix.fromArray(y);
		
		Matrix error = Matrix.subtract(target, output);
		Matrix gradient = output.derivativeSigmoid();
		gradient.multiply(error);
		gradient.multiply(learningRate);
		
		Matrix trainedHidden = Matrix.transpose(hidden);
		Matrix hiddenOutputDelta = Matrix.multiply(gradient, trainedHidden);
		
		hiddenOutputWeights.add(hiddenOutputDelta);
		outputBias.add(gradient);
		
		Matrix hiddenOutputTrained = Matrix.transpose(hiddenOutputWeights);
		Matrix hiddenErrors = Matrix.multiply(hiddenOutputTrained, error);
		
		Matrix hiddenGradient = hidden.derivativeSigmoid();
		hiddenGradient.multiply(hiddenErrors);
		hiddenGradient.multiply(learningRate);
		
		Matrix inputTrained = Matrix.transpose(input);
		Matrix inputHiddenDelta = Matrix.multiply(hiddenGradient, inputTrained);
		
		inputHiddenWeights.add(inputHiddenDelta);
		hiddenBias.add(hiddenGradient);
		
	}
	
	public void fit(double[][] x, double[][] y, int epochs) {
		for (int i = 0; i < epochs; i++) {
			int sampleN = (int)(Math.random() * x.length);
			this.train(x[sampleN], y[sampleN]);
		}
		
	}
	
}
