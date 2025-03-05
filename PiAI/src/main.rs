use rayon::prelude::*;

mod neural_network;

const WINDOW_SIZE: usize = 10;

const NETWORK: [u32; 5] = [WINDOW_SIZE as u32, 128, 256, 64, 1];
const ACTIVATION_FUNC_HIDDEN : usize = 3;
const ACTIVATION_FUNC_OUTPUT : usize = 0;
const INITIAL_WEIGHT_RANGE : (f32, f32) = (-1.0, 1.0);
const INITIAL_BIAS_RANGE : (f32, f32) = (-0.1, 0.1);

const POPULATION_SIZE: usize = 10000;


fn main() {
    let ascii_digits = b"3141592653589793238462643383279502884197169";
    let binary_digits: Vec<u8> = ascii_digits.iter().map(|c| c - b'0').collect();

    // Create the population
    let mut population: Vec<neural_network::NeuralNetwork> = Vec::new();
    for _ in 0..POPULATION_SIZE {
        population.push(neural_network::NeuralNetwork::new(Vec::from(&NETWORK[..]), ACTIVATION_FUNC_HIDDEN, ACTIVATION_FUNC_OUTPUT, INITIAL_WEIGHT_RANGE, INITIAL_BIAS_RANGE));
    }

    loop {
        // Run all the networks and get the best one
        // Parallelize the evaluation of the networks
        let scores: Vec<usize> = (0..POPULATION_SIZE).into_par_iter().map(|i| {
            run(&population[i], &binary_digits)
        }).collect();
        let best_score = scores.iter().max().unwrap();
        let best_network = scores.iter().position(|x| x == best_score).unwrap();


        // Print the best network
        println!("Best score: {}", best_score);

        // Clone the best to replace the population and mutate all of them except one
        population = vec![population[best_network].clone()];
        let new_population: Vec<neural_network::NeuralNetwork> = (1..POPULATION_SIZE).into_par_iter().map(|i| {
            let mut agent = population[0].clone();
            agent.mutate(0.05, 0.1);
            agent
        }).collect();
        population.extend(new_population);
    }
}


fn run(neural_network:&neural_network::NeuralNetwork, binary_digits:&Vec<u8>) -> usize {
    let mut next_digit_index:usize =  WINDOW_SIZE;
    loop {
        let window:Vec<u8> = Vec::from(&binary_digits[next_digit_index-WINDOW_SIZE..next_digit_index]);
        let normalized_window:Vec<f32> = window.iter().map(|x| *x as f32/10.0 as f32).collect();
        if !((neural_network.feed_forward(normalized_window)[0] * 10.0).round() as u8 == binary_digits[next_digit_index]) {
            break;
        }
        next_digit_index += 1;

        if next_digit_index == binary_digits.len() {
            panic!("The AI is very good at memorizing Pi");
        }
    }
    return next_digit_index-WINDOW_SIZE;
}