import java.util.ArrayList;
import java.util.List;

import javax.swing.JFrame;


public class Main {

	static double[][] input = {
			{0,0},{1,0},{0,1},{1,1}
	};
	
	static double[][] output = {
			{0.00},{1.00},{1.00},{0.00}
	};
	
	
	private static NeuralNetwork nn;
	
	public static void main(String[] args) {
		// TODO Auto-generated method stub
		
		nn = new NeuralNetwork(2,10,1);
		nn.fit(input,output, 100000);
		
		double[][] in = {{0,0},{0,1},{1,0},{1,1}};
		ArrayList<Double> out;
		for (double d[]:input) {
			out = nn.predict(d);
			System.out.println(out.toString());
		}
		
	}

}
