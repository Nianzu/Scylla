use rand::Rng;

// ------------------------------
// Utility functions
// ------------------------------

/// Sigmoid activation function.
fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// Derivative of sigmoid given its output a = sigmoid(x).
fn sigmoid_derivative(a: f64) -> f64 {
    a * (1.0 - a)
}

// ------------------------------
// Fully Connected (Dense) Layer
// ------------------------------

#[derive(Debug)]
struct Layer {
    // Each row in `weights` is a set of weights for one neuron.
    weights: Vec<Vec<f64>>,
    biases: Vec<f64>,
    // Values saved during forward pass (used during backprop).
    last_input: Vec<f64>,
    last_z: Vec<f64>,
    last_output: Vec<f64>,
}

impl Layer {
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

    /// Forward pass: computes z = W·x + b and applies sigmoid activation.
    fn forward(&mut self, input: &Vec<f64>) -> Vec<f64> {
        self.last_input = input.clone();
        let mut z = vec![];
        let mut output = vec![];
        for (w_row, b) in self.weights.iter().zip(self.biases.iter()) {
            let z_val: f64 = w_row
                .iter()
                .zip(input.iter())
                .map(|(w, x)| w * x)
                .sum::<f64>()
                + b;
            z.push(z_val);
            output.push(sigmoid(z_val));
        }
        self.last_z = z;
        self.last_output = output.clone();
        output
    }

    /// Backward pass: using the gradient dL/d(output), compute gradients, update weights, and return dL/d(input).
    fn backward(&mut self, dL_dout: &Vec<f64>, learning_rate: f64) -> Vec<f64> {
        // Chain-rule: dL/dz = dL/dout * sigmoid_derivative(z)
        let dL_dz: Vec<f64> = dL_dout
            .iter()
            .zip(self.last_output.iter())
            .map(|(&d, &a)| d * sigmoid_derivative(a))
            .collect();

        let input = &self.last_input;
        let mut dL_dw = vec![vec![0.0; input.len()]; self.weights.len()];
        let mut dL_db = vec![0.0; self.biases.len()];

        // Compute gradients for each weight and bias.
        for i in 0..self.weights.len() {
            dL_db[i] = dL_dz[i];
            for j in 0..input.len() {
                dL_dw[i][j] = dL_dz[i] * input[j];
            }
        }

        // Update parameters.
        for i in 0..self.weights.len() {
            for j in 0..input.len() {
                self.weights[i][j] -= learning_rate * dL_dw[i][j];
            }
            self.biases[i] -= learning_rate * dL_db[i];
        }

        // Compute dL/d(input): propagate gradient to previous layer.
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

// ------------------------------
// Convolutional Layer (for single–channel images)
// ------------------------------

struct ConvLayer {
    // Filters: each filter is a 2D matrix.
    filters: Vec<Vec<Vec<f64>>>, // shape: [num_filters][filter_height][filter_width]
    biases: Vec<f64>,             // one bias per filter
    filter_height: usize,
    filter_width: usize,
    num_filters: usize,
    // Save the input image and the computed feature maps for backprop.
    last_input: Vec<Vec<f64>>,
    last_output: Vec<Vec<Vec<f64>>>, // shape: [num_filters][out_height][out_width]
}

impl ConvLayer {
    fn new(num_filters: usize, filter_height: usize, filter_width: usize) -> Self {
        let mut rng = rand::thread_rng();
        let filters = (0..num_filters)
            .map(|_| {
                (0..filter_height)
                    .map(|_| {
                        (0..filter_width)
                            .map(|_| rng.gen_range(-1.0..1.0))
                            .collect::<Vec<f64>>()
                    })
                    .collect::<Vec<Vec<f64>>>()
            })
            .collect::<Vec<Vec<Vec<f64>>>>();
        let biases = (0..num_filters)
            .map(|_| rng.gen_range(-1.0..1.0))
            .collect::<Vec<f64>>();
        Self {
            filters,
            biases,
            filter_height,
            filter_width,
            num_filters,
            last_input: vec![],
            last_output: vec![],
        }
    }

    /// Forward pass: perform a “cross-correlation” of each filter with the input image,
    /// add the bias, and apply the sigmoid activation.
    fn forward(&mut self, input: &Vec<Vec<f64>>) -> Vec<Vec<Vec<f64>>> {
        self.last_input = input.clone();
        let input_height = input.len();
        let input_width = input[0].len();
        // No padding; stride = 1.
        let out_height = input_height - self.filter_height + 1;
        let out_width = input_width - self.filter_width + 1;
        let mut output = vec![];
        for f in 0..self.num_filters {
            let mut out_channel = vec![vec![0.0; out_width]; out_height];
            for i in 0..out_height {
                for j in 0..out_width {
                    let mut sum = 0.0;
                    for fi in 0..self.filter_height {
                        for fj in 0..self.filter_width {
                            sum += self.filters[f][fi][fj] * input[i + fi][j + fj];
                        }
                    }
                    sum += self.biases[f];
                    out_channel[i][j] = sigmoid(sum);
                }
            }
            output.push(out_channel);
        }
        self.last_output = output.clone();
        output
    }

    /// Backward pass:
    /// - Compute gradients for filters and biases using the gradient from later layers.
    /// - Update parameters.
    /// - Compute and return dL/d(input) for further propagation.
    fn backward(&mut self, dL_dout: &Vec<Vec<Vec<f64>>>, learning_rate: f64) -> Vec<Vec<f64>> {
        let input_height = self.last_input.len();
        let input_width = self.last_input[0].len();
        let out_height = input_height - self.filter_height + 1;
        let out_width = input_width - self.filter_width + 1;

        // Compute gradient with respect to z (before activation).
        let mut dL_dz = vec![vec![vec![0.0; out_width]; out_height]; self.num_filters];
        for f in 0..self.num_filters {
            for i in 0..out_height {
                for j in 0..out_width {
                    let a = self.last_output[f][i][j];
                    dL_dz[f][i][j] = dL_dout[f][i][j] * sigmoid_derivative(a);
                }
            }
        }

        // Gradients for filters and biases.
        let mut dL_dfilters = vec![vec![vec![0.0; self.filter_width]; self.filter_height]; self.num_filters];
        let mut dL_dbiases = vec![0.0; self.num_filters];

        for f in 0..self.num_filters {
            for i in 0..out_height {
                for j in 0..out_width {
                    let grad = dL_dz[f][i][j];
                    dL_dbiases[f] += grad;
                    for fi in 0..self.filter_height {
                        for fj in 0..self.filter_width {
                            dL_dfilters[f][fi][fj] += grad * self.last_input[i + fi][j + fj];
                        }
                    }
                }
            }
        }

        // Update filters and biases.
        for f in 0..self.num_filters {
            for fi in 0..self.filter_height {
                for fj in 0..self.filter_width {
                    self.filters[f][fi][fj] -= learning_rate * dL_dfilters[f][fi][fj];
                }
            }
            self.biases[f] -= learning_rate * dL_dbiases[f];
        }

        // Compute dL/d(input): distribute gradients back over the input.
        // (This is similar to a “full” convolution of dL/dz with the flipped filters.)
        let mut dL_dinput = vec![vec![0.0; input_width]; input_height];
        for f in 0..self.num_filters {
            for i in 0..out_height {
                for j in 0..out_width {
                    let grad = dL_dz[f][i][j];
                    for fi in 0..self.filter_height {
                        for fj in 0..self.filter_width {
                            // Flip the filter indices.
                            dL_dinput[i + fi][j + fj] += grad
                                * self.filters[f][self.filter_height - fi - 1]
                                       [self.filter_width - fj - 1];
                        }
                    }
                }
            }
        }
        dL_dinput
    }
}

// ------------------------------
// Flattening and Reshaping Helpers
// ------------------------------

/// Flattens the 3D output of a ConvLayer into a 1D vector (for a fully connected layer).
fn flatten(conv_output: &Vec<Vec<Vec<f64>>>) -> Vec<f64> {
    let mut flat = vec![];
    for channel in conv_output {
        for row in channel {
            for &val in row {
                flat.push(val);
            }
        }
    }
    flat
}

/// Reshapes a flat vector back into a 3D tensor with dimensions: (num_filters, out_height, out_width).
fn reshape_flat_to_conv(
    flat: &Vec<f64>,
    num_filters: usize,
    out_height: usize,
    out_width: usize,
) -> Vec<Vec<Vec<f64>>> {
    let mut conv_grad = vec![];
    let mut index = 0;
    for _ in 0..num_filters {
        let mut channel = vec![];
        for _ in 0..out_height {
            let mut row = vec![];
            for _ in 0..out_width {
                row.push(flat[index]);
                index += 1;
            }
            channel.push(row);
        }
        conv_grad.push(channel);
    }
    conv_grad
}

// ------------------------------
// ConvNet: Combining a ConvLayer and a Fully Connected Layer
// ------------------------------

struct ConvNet {
    conv_layer: ConvLayer,
    fc_layer: Layer,
    // We store the dimensions of the convolution output (needed to reshape gradients).
    conv_output_dims: (usize, usize, usize), // (num_filters, out_height, out_width)
}

impl ConvNet {
    /// Create a new ConvNet.
    /// - `input_height` and `input_width` are the dimensions of the input image.
    /// - The convolution layer uses `num_filters` filters of size (`filter_height`×`filter_width`).
    /// - The fully connected layer maps the flattened convolution output to `fc_output_size` outputs.
    fn new(
        input_height: usize,
        input_width: usize,
        num_filters: usize,
        filter_height: usize,
        filter_width: usize,
        fc_output_size: usize,
    ) -> Self {
        let conv_layer = ConvLayer::new(num_filters, filter_height, filter_width);
        let out_height = input_height - filter_height + 1;
        let out_width = input_width - filter_width + 1;
        let flattened_size = num_filters * out_height * out_width;
        let fc_layer = Layer::new(flattened_size, fc_output_size);
        Self {
            conv_layer,
            fc_layer,
            conv_output_dims: (num_filters, out_height, out_width),
        }
    }

    /// Forward pass: feed the image through the conv layer, flatten the result, then through the FC layer.
    fn forward(&mut self, input: &Vec<Vec<f64>>) -> Vec<f64> {
        let conv_out = self.conv_layer.forward(input);
        let flat = flatten(&conv_out);
        self.fc_layer.forward(&flat)
    }

    /// Backward pass: propagate gradients from the FC layer, reshape to conv dimensions, then back through the conv layer.
    fn backward(&mut self, dL_dout: &Vec<f64>, learning_rate: f64) {
        let dL_dflat = self.fc_layer.backward(dL_dout, learning_rate);
        let (num_filters, out_height, out_width) = self.conv_output_dims;
        let dL_dconv = reshape_flat_to_conv(&dL_dflat, num_filters, out_height, out_width);
        self.conv_layer.backward(&dL_dconv, learning_rate);
    }
}

// ------------------------------
// Main: Training on a Toy Problem (Extend to MNIST as desired)
// ------------------------------

fn main() {
    // For this toy problem, our input images are 5x5.
    // We’ll define a simple task: if the center pixel (position [2][2]) is high (>0.5), target = 1; otherwise, target = 0.
    let training_data = vec![
        // Example with a high center pixel.
        (
            vec![
                vec![0.0, 0.0, 0.0, 0.0, 0.0],
                vec![0.0, 0.0, 0.0, 0.0, 0.0],
                vec![0.0, 0.0, 0.9, 0.0, 0.0],
                vec![0.0, 0.0, 0.0, 0.0, 0.0],
                vec![0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            vec![1.0],
        ),
        // Example with a low center pixel.
        (
            vec![
                vec![0.0, 0.0, 0.0, 0.0, 0.0],
                vec![0.0, 0.0, 0.0, 0.0, 0.0],
                vec![0.0, 0.0, 0.1, 0.0, 0.0],
                vec![0.0, 0.0, 0.0, 0.0, 0.0],
                vec![0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            vec![0.0],
        ),
    ];

    // Create our ConvNet.
    // For 5×5 inputs, we use 1 convolution filter of size 3×3.
    // (The convolution output will be 3×3, so after flattening, the FC layer input size is 1*3*3 = 9.)
    // We then use a fully connected layer with 1 output (for binary classification).
    let mut net = ConvNet::new(5, 5, 1, 3, 3, 1);

    let learning_rate = 0.1;
    let epochs = 5000;

    for epoch in 0..epochs {
        // Cycle through training examples.
        let index = epoch % training_data.len();
        let (ref input, ref target) = training_data[index];

        // Forward pass.
        let output = net.forward(input);
        // Compute loss (Mean Squared Error).
        let loss = 0.5 * (output[0] - target[0]).powi(2);
        // Compute gradient of loss with respect to the output.
        let dL_dout = vec![output[0] - target[0]];
        // Backward pass.
        net.backward(&dL_dout, learning_rate);

        if epoch % 500 == 0 {
            println!("Epoch {}: loss = {}", epoch, loss);
        }
    }

    // Test the network on the training examples.
    println!("\nTrained network results:");
    for (i, (input, target)) in training_data.iter().enumerate() {
        let output = net.forward(input);
        println!("Example {}: target = {:?}, output = {:?}", i, target, output);
    }
}
