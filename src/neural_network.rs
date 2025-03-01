use rand::Rng;
use savefile::prelude::*;
use savefile_derive::Savefile;

#[derive(Savefile,Clone)]
struct Layer {
    weights: Vec<f32>,
    weight_width: usize,
    biases: Vec<f32>,

    last_input: Vec<f32>,
    last_output: Vec<f32>,
    dl_dinput: Vec<f32>,
}

impl Layer {
    fn new(input_size: usize, output_size: usize) -> Self {
        let mut rng = rand::rng();
        let weight_width = input_size;

        let weights = (0..(output_size * input_size)) // For each node in the current layer
            .map(|_| rng.random_range(-1.0..1.0))
            .collect::<Vec<f32>>();

        let biases = (0..output_size) // For each node
            .map(|_| rng.random_range(-1.0..1.0)) // Select a random number in our seed range
            .collect::<Vec<f32>>();

        let dl_dinput = vec![0.0; input_size];

        Self {
            weights,
            weight_width,
            biases,
            last_input: vec![],
            last_output: vec![],
            dl_dinput,
        }
    }

    // https://en.wikipedia.org/wiki/Sigmoid_function
    fn sigmoid(x: f32) -> f32 {
        1.0 / (1.0 + (-x).exp())
    }

    // https://math.stackexchange.com/questions/78575/derivative-of-sigmoid-function-sigma-x-frac11e-x
    fn sigmoid_derivative(sigmoid_x: f32) -> f32 {
        sigmoid_x * (1.0 - sigmoid_x)
    }

    fn forward(&mut self, input: &Vec<f32>) -> Vec<f32> {
        self.last_input = input.clone();
        let mut output = vec![];
        let num_neurons = self.biases.len();

        for i in 0..num_neurons {
            let start = i * self.weight_width;
            let end = start + self.weight_width;
            let neuron_weights = &self.weights[start..end];

            let z_i: f32 = neuron_weights
                .iter()
                .zip(input.iter())
                .map(|(w, x)| w * x)
                .sum::<f32>()
                + self.biases[i];

            let a_i = Self::sigmoid(z_i);
            output.push(a_i);
        }

        self.last_output = output.clone();
        output
    }

    fn backward(&mut self, dl_dout: &Vec<f32>, learning_rate: f32) -> &Vec<f32> {
        let num_neurons = self.biases.len();

        // dL/dz = dL/dout * sigmoid_derivative(z)
        let dl_dz: Vec<f32> = dl_dout
            .iter()
            .zip(self.last_output.iter())
            .map(|(&d, &a)| d * Self::sigmoid_derivative(a))
            .collect();

        // dL/db (Gradient for the bias) = dL/dz
        let dl_db = &dl_dz;

        // dL/dw (Gradient for the weight) = dL/dz * x for
        let input = &self.last_input;

        // Update weights and biases using gradient descent
        for i in 0..num_neurons {
            for j in 0..input.len() {
                unsafe {
                    self.weights[i * self.weight_width + j] -=
                        learning_rate * dl_dz.get_unchecked(i) * input.get_unchecked(j);
                }
            }
            unsafe {
                self.biases[i] -= learning_rate * dl_db.get_unchecked(i);
            }
        }

        // Compute and return dL/dx for previous layer
        // dL/dx = sum(dL/dz * w) for w in weights[i]
        for j in 0..input.len() {
            let mut sum = 0.0;
            for i in 0..num_neurons {
                unsafe {
                    sum += dl_dz.get_unchecked(i)
                        * self.weights.get_unchecked(i * self.weight_width + j);
                }
            }
            self.dl_dinput[j] = sum;
        }
        &self.dl_dinput
    }
}

#[derive(Savefile,Clone)]
pub struct Network {
    name: String,
    layers: Vec<Layer>,
}

impl Network {
    pub fn new(name_str: &str, layer_sizes: &[usize]) -> Self {
        let mut layers = vec![];
        let name: String = name_str.to_owned();
        for i in 0..(layer_sizes.len() - 1) {
            layers.push(Layer::new(layer_sizes[i], layer_sizes[i + 1]));
        }
        Self { name, layers }
    }

    pub fn forward(&mut self, input: &Vec<f32>) -> Vec<f32> {
        let mut output = input.clone();
        for layer in self.layers.iter_mut() {
            output = layer.forward(&output)
        }
        output
    }

    fn backward(&mut self, dl_dout: &Vec<f32>, learning_rate: f32) {
        let mut grad = dl_dout;
        for layer in self.layers.iter_mut().rev() {
            grad = layer.backward(&grad, learning_rate);
        }
    }

    pub fn train(
        &mut self,
        dataset: &Vec<Vec<f32>>,
        labels: &Vec<Vec<f32>>,
        learning_rate: f32,
    ) -> f32 {
        let mut loss_sum = 0.0;
        for i in 0..dataset.len() {
            let output = self.forward(&dataset[i]);
            let loss = rmse(&output, &labels[i]); // mean squared error
            loss_sum += loss;

            let dl_dout = network_gradient_loss(&output, &labels[i]);
            self.backward(&dl_dout, learning_rate);
        }
        loss_sum / dataset.len() as f32
    }

    pub fn validation_loss(&mut self, dataset: &Vec<Vec<f32>>, labels: &Vec<Vec<f32>>) -> f32 {
        let mut validation_loss_sum = 0.0;
        for i in 0..dataset.len() {
            let output = self.forward(&dataset[i]);
            let loss = rmse(&output, &labels[i]);
            validation_loss_sum += loss;
        }
        validation_loss_sum / dataset.len() as f32
    }

    pub fn accuracy(&mut self, dataset: &Vec<Vec<f32>>, labels: &Vec<Vec<f32>>) -> f32 {
        let mut accuracy = 0.0;
        for i in 0..dataset.len() {
            let output = self.forward(&dataset[i]);
            // https://stackoverflow.com/questions/53903318/what-is-the-idiomatic-way-to-get-the-index-of-a-maximum-or-minimum-floating-poin
            let predicted = output
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.total_cmp(b))
                .map(|(index, _)| index)
                .unwrap();
            let actual = labels[i]
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.total_cmp(b))
                .map(|(index, _)| index)
                .unwrap();
            if predicted == actual {
                accuracy += 1.0;
            }
        }
        accuracy / dataset.len() as f32
    }

    pub fn save(&mut self) {
        save_file("trained_network/".to_owned() + &self.name + ".bin", 0, self).unwrap();
    }

    pub fn load(path: &str) -> Network {
        load_file(path, 0).unwrap()
    }
}

fn rmse(current: &Vec<f32>, target: &Vec<f32>) -> f32 {
    let n = current.len();
    let sum_squared_error: f32 = current
        .iter()
        .zip(target.iter())
        .map(|(c, t)| (c - t).powi(2))
        .sum();
    (sum_squared_error / n as f32).sqrt()
}

// dl/dout (Gradient of loss for the network) = (output - target)
fn network_gradient_loss(predictions: &Vec<f32>, target: &Vec<f32>) -> Vec<f32> {
    predictions
        .into_iter()
        .zip(target.clone().into_iter())
        .map(|(o, t)| (o - t))
        .collect()
}
