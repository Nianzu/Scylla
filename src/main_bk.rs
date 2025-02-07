// Add in Cargo.toml the dependency:
// [dependencies]
// rand = "0.8"

use rand::Rng;

#[derive(Debug)]
struct Layer {
    // The weights matrix: each row corresponds to one neuron’s weights.
    weights: Vec<Vec<f64>>,
    // Biases for each neuron.
    biases: Vec<f64>,
    // For storing values from the last forward pass (needed for backprop).
    last_input: Vec<f64>,
    last_z: Vec<f64>,
    last_output: Vec<f64>,
}

impl Layer {
    /// Create a new layer with given input and output sizes.
    fn new(input_size: usize, output_size: usize) -> Self {
        let mut rng = rand::thread_rng();
        let weights = (0..output_size)
            .map(|_| {
                (0..input_size)
                    .map(|_| rng.gen_range(-1.0..1.0))
                    .collect::<Vec<f64>>()
            })
            .collect::<Vec<Vec<f64>>>();
        let biases = (0..output_size)
            .map(|_| rng.gen_range(-1.0..1.0))
            .collect::<Vec<f64>>();
        Self {
            weights,
            biases,
            last_input: vec![],
            last_z: vec![],
            last_output: vec![],
        }
    }

    /// Sigmoid activation function.
    fn sigmoid(x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }

    /// Derivative of the sigmoid function, given σ(x).
    fn sigmoid_derivative(sigmoid_x: f64) -> f64 {
        sigmoid_x * (1.0 - sigmoid_x)
    }

    /// Forward pass: computes output = sigmoid(W * input + b)
    fn forward(&mut self, input: &Vec<f64>) -> Vec<f64> {
        self.last_input = input.clone();
        let mut z = vec![];
        let mut output = vec![];

        // For each neuron i in the layer:
        for (w_row, b) in self.weights.iter().zip(self.biases.iter()) {
            // Compute the weighted sum: z_i = sum_j (w_ij * x_j) + b_i
            let z_i: f64 = w_row.iter()
                .zip(input.iter())
                .map(|(w, x)| w * x)
                .sum::<f64>() + b;
            z.push(z_i);
            // Apply activation function (sigmoid)
            let a_i = Self::sigmoid(z_i);
            output.push(a_i);
        }
        self.last_z = z;
        self.last_output = output.clone();
        output
    }

    /// Backward pass:
    /// Given the gradient dL/d(output) from later layers (or the loss),
    /// compute gradients for weights and biases, update parameters,
    /// and return dL/d(input) for the previous layer.
    fn backward(&mut self, dL_dout: &Vec<f64>, learning_rate: f64) -> Vec<f64> {
        // 1. Compute dL/dz = dL/dout * sigmoid_derivative(z)
        // (Note: we stored last_output = sigmoid(z))
        let dL_dz: Vec<f64> = dL_dout.iter()
            .zip(self.last_output.iter())
            .map(|(&d, &a)| d * Self::sigmoid_derivative(a))
            .collect();

        // 2. Gradients for weights: dL/dw_ij = dL/dz_i * x_j,
        //    and for biases: dL/db_i = dL/dz_i.
        let input = &self.last_input;
        let mut dL_dw = vec![vec![0.0; input.len()]; self.weights.len()];
        let mut dL_db = vec![0.0; self.biases.len()];

        for i in 0..self.weights.len() {
            dL_db[i] = dL_dz[i];
            for j in 0..input.len() {
                dL_dw[i][j] = dL_dz[i] * input[j];
            }
        }

        // 3. Update weights and biases using gradient descent.
        for i in 0..self.weights.len() {
            for j in 0..input.len() {
                self.weights[i][j] -= learning_rate * dL_dw[i][j];
            }
            self.biases[i] -= learning_rate * dL_db[i];
        }

        // 4. Compute and return dL/d(input) for the previous layer:
        //    dL/dx_j = sum_i (dL/dz_i * w_ij)
        let mut dL_dinput = vec![0.0; input.len()];
        for j in 0..input.len() {
            let mut sum = 0.0;
            for i in 0..self.weights.len() {
                sum += dL_dz[i] * self.weights[i][j];
            }
            dL_dinput[j] = sum;
        }
        dL_dinput
    }
}

/// The Network struct is a container for multiple layers.
struct Network {
    layers: Vec<Layer>,
}

impl Network {
    /// Creates a new network given a slice of layer sizes.
    /// For example, &[2, 2, 1] creates a network with an input layer of size 2,
    /// one hidden layer of size 2, and an output layer of size 1.
    fn new(layer_sizes: &[usize]) -> Self {
        let mut layers = vec![];
        for i in 0..(layer_sizes.len() - 1) {
            layers.push(Layer::new(layer_sizes[i], layer_sizes[i + 1]));
        }
        Self { layers }
    }

    /// The network’s forward pass: sequentially pass the input through all layers.
    fn forward(&mut self, input: &Vec<f64>) -> Vec<f64> {
        let mut output = input.clone();
        for layer in self.layers.iter_mut() {
            output = layer.forward(&output);
        }
        output
    }

    /// The network’s backward pass: propagate the gradient from the output
    /// back through each layer (in reverse order).
    fn backward(&mut self, dL_dout: &Vec<f64>, learning_rate: f64) {
        let mut grad = dL_dout.clone();
        // Propagate through layers in reverse order.
        for layer in self.layers.iter_mut().rev() {
            grad = layer.backward(&grad, learning_rate);
        }
    }
}

fn main() {
    // --- XOR Training Data ---
    // Each tuple is (input, target)
    let training_data = vec![
        (vec![0.0, 0.0], vec![0.0]),
        (vec![0.0, 1.0], vec![1.0]),
        (vec![1.0, 0.0], vec![1.0]),
        (vec![1.0, 1.0], vec![0.0]),
    ];

    // --- Create the Network ---
    // For XOR we can use a network with 2 input neurons, one hidden layer (2 neurons), and 1 output neuron.
    let mut network = Network::new(&[2, 2, 1]);

    let learning_rate = 0.5;
    let epochs = 100000;

    // --- Training Loop ---
    // Here we perform stochastic gradient descent (one example per iteration).
    for epoch in 0..epochs {
        // For simplicity, cycle through the training data.
        let index = epoch % training_data.len();
        let (ref input, ref target) = training_data[index];

        // 1. Forward pass
        let output = network.forward(input);

        // 2. Compute the loss (Mean Squared Error)
        let loss = 0.5 * (output[0] - target[0]).powi(2);

        // 3. Compute gradient of loss with respect to network output.
        //    For MSE, dL/d(output) = (output - target)
        let dL_dout = vec![output[0] - target[0]];

        // 4. Backward pass: update weights layer by layer.
        network.backward(&dL_dout, learning_rate);

        if epoch % 1000 == 0 {
            println!("Epoch {}: loss = {}", epoch, loss);
        }
    }

    // --- Testing the Network ---
    println!("\nTrained network results on XOR problem:");
    for (input, target) in training_data.iter() {
        let output = network.forward(input);
        println!("Input: {:?}, target: {:?}, output: {:?}", input, target, output);
    }
}
