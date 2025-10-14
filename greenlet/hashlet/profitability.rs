// SPDX-License-Identifier: MIT
use std::env;
use std::process;
mod lib;
use lib::{check_profitable, collapsed_profitable_m53};
fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() != 4 {
        eprintln!("Usage: {} <target_price> <current_price> <volume>", args[0]);
        process::exit(1);
    }
    let target_price: u128 = args[1].parse().unwrap_or_else(|_| {
        eprintln!("Invalid target_price");
        process::exit(1);
    });
    let current_price: u128 = args[2].parse().unwrap_or_else(|_| {
        eprintln!("Invalid current_price");
        process::exit(1);
    });
    let volume: u128 = args[3].parse().unwrap_or_else(|_| {
        eprintln!("Invalid volume");
        process::exit(1);
    });
    let result = check_profitable(target_price, current_price, volume);
    let m53_result = collapsed_profitable_m53(194062501, volume, target_price, current_price);
    println!(
        "{{\"profitable\": {}, \"m53_passes\": {}, \"m53_reward\": {}}}",
        result, m53_result.0, m53_result.1
    );
}
