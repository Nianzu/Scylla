use plotters::prelude::*;
use rand::Rng;
use std::{os::linux::net, process::Output, vec};

struct Layer {
    weights: Vec<Vec<f64>>,
    biases: Vec<f64>,

    last_input: Vec<f64>,
    last_output: Vec<f64>,
}

impl Layer {
    fn new(input_size: usize, output_size: usize) -> Self {
        let mut rng = rand::thread_rng();

        let weights = (0..output_size) // For each node in the current layer
            .map(|_| {
                (0..input_size) // And for each input to that node
                    .map(|_| rng.gen_range(-1.0..1.0)) // Select a random number in our seed range
                    .collect::<Vec<f64>>()
            })
            .collect::<Vec<Vec<f64>>>();

        let biases = (0..output_size) // For each node
            .map(|_| rng.gen_range(-1.0..1.0)) // Select a random number in our seed range
            .collect::<Vec<f64>>();

        Self {
            weights,
            biases,
            last_input: vec![],
            last_output: vec![],
        }
    }

    // https://en.wikipedia.org/wiki/Sigmoid_function
    fn sigmoid(x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }

    // https://math.stackexchange.com/questions/78575/derivative-of-sigmoid-function-sigma-x-frac11e-x
    fn sigmoid_derivative(sigmoid_x: f64) -> f64 {
        sigmoid_x * (1.0 - sigmoid_x)
    }

    fn forward(&mut self, input: &Vec<f64>) -> Vec<f64> {
        self.last_input = input.clone();
        let mut output = vec![];

        for (weights, bias) in self.weights.iter().zip(self.biases.iter()) {
            // z = sum(w * x) + bias for w,x in weights,input
            let z_i: f64 = weights
                .iter()
                .zip(input.iter())
                .map(|(w, x)| w * x)
                .sum::<f64>()
                + bias;

            let a_i = Self::sigmoid(z_i);
            output.push(a_i);
        }
        self.last_output = output.clone();
        output
    }

    fn backward(&mut self, dL_dout: &Vec<f64>, learning_rate: f64) -> Vec<f64> {
        // dL/dz = dL/dout * sigmoid_derivative(z)
        let dL_dz: Vec<f64> = dL_dout
            .iter()
            .zip(self.last_output.iter())
            .map(|(&d, &a)| d * Self::sigmoid_derivative(a))
            .collect();

        // dL/db (Gradient for the bias) = dL/dz
        let dL_db = dL_dz.to_vec();

        // dL/dw (Gradient for the weight) = dL/dz * x for
        let input = &self.last_input;
        let mut dL_dw = vec![vec![0.0; input.len()]; self.weights.len()];
        for i in 0..self.weights.len() {
            for j in 0..input.len() {
                dL_dw[i][j] = dL_dz[i] * input[j];
            }
        }

        // Update weights and biases using gradient descent
        for i in 0..self.weights.len() {
            for j in 0..input.len() {
                self.weights[i][j] -= learning_rate * dL_dw[i][j];
            }
            self.biases[i] -= learning_rate * dL_db[i];
        }

        // Compute and return dL/dx for previous layer
        // dL/dx = sum(dL/dz * w) for w in weights[i]
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

struct Network {
    layers: Vec<Layer>,
}

impl Network {
    fn new(layer_sizes: &[usize]) -> Self {
        let mut layers = vec![];
        for i in 0..(layer_sizes.len() - 1) {
            layers.push(Layer::new(layer_sizes[i], layer_sizes[i + 1]));
        }
        Self { layers }
    }

    fn forward(&mut self, input: &Vec<f64>) -> Vec<f64> {
        let mut output = input.clone();
        for layer in self.layers.iter_mut() {
            output = layer.forward(&output)
        }
        output
    }

    fn backward(&mut self, dL_dout: &Vec<f64>, learning_rate: f64) {
        let mut grad = dL_dout.clone();
        for layer in self.layers.iter_mut().rev() {
            grad = layer.backward(&grad, learning_rate);
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    //#########################################################################
    // Load data
    //#########################################################################

    // xor training data
    let training_data = vec![
        (vec![0.0, 0.0], vec![0.0]),
        (vec![0.0, 1.0], vec![1.0]),
        (vec![1.0, 0.0], vec![1.0]),
        (vec![1.0, 1.0], vec![0.0]),
    ];

    let mut network = Network::new(&[2, 2, 1]);
    let learning_rate = 0.5;
    let epochs = 100000;

    let mut losses: Vec<f64> = vec![];

    //#########################################################################
    // Training
    //#########################################################################

    for epoch in 0..epochs {
        let mut loss_sum = 0.0;
        for (ref input, ref target) in training_data.iter() {
            let output = network.forward(input);
            let loss = 0.5 * (output[0] - target[0]).powi(2); // mean squared error
            loss_sum += loss;
            // dl/dout (Gradient of loss for the network) = (output - target)
            let dL_dout = vec![output[0] - target[0]];
            network.backward(&dL_dout, learning_rate);
        }
        let loss_avg = loss_sum / training_data.len() as f64;
        losses.push(loss_avg);
        if epoch % 1000 == 0 {
            println!("Epoch {}: avg loss = {}", epoch, loss_avg);
        }
    }

    //#########################################################################
    // Testing
    //#########################################################################

    for (input, target) in training_data.iter() {
        let output = network.forward(input);
        println!(
            "Input: {:?}, target: {:?} output: {:?}",
            input, target, output
        );
    }

    //#########################################################################
    // Visualization
    //#########################################################################
    // Create a drawing area for the plot
    let root = BitMapBackend::new("line_plot.png", (640, 480)).into_drawing_area();
    root.fill(&WHITE)?;

    // Define the chart area and label axes
    let mut chart = ChartBuilder::on(&root)
        .caption("Loss over epochs", ("sans-serif", 50).into_font())
        .margin(20)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(0.0..losses.len() as f64, 0.0..0.2)?;
    // .build_cartesian_2d(0.0..losses.len() as f64, (0.0..0.2).log_scale())?;

    // Draw the mesh for the chart
    chart.configure_mesh().draw()?; // Define the data points for the line
    let line_data = vec![
        (0.0, 0.0),
        (1.0, 2.0),
        (2.0, 3.0),
        (3.0, 5.0),
        (4.0, 7.0),
        (5.0, 8.0),
    ];

    // Plot the line using the data points
    chart
        .draw_series(LineSeries::new(
            losses.iter().enumerate().map(|(i, &loss)| (i as f64, loss)),
            &RED,
        ))?
        .label("Training Loss")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));

    // Add legend to the chart
    chart
        .configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()?;

    // Save the plot
    root.present()?;

    Ok(())
}
