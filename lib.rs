// SPDX-License-Identifier: MIT
#![no_std]
pub const FEE_RATE: u128 = 30;
pub const MARTINGALE_FACTOR: u128 = 2;
pub const DIVISOR: u128 = 3;
pub const MOD_BITS: u128 = 256;
pub const MOD_SYM: u128 = 369;
pub const FLASH_FEE: u128 = 25;
pub const BURN_RATE: u128 = 50;
pub fn check_profitable(target_price: u128, current_price: u128, volume: u128) -> bool {
    let mut delta_p = if current_price > target_price {
        current_price - target_price
    } else {
        target_price - current_price
    };
    delta_p = delta_p * 10000 / target_price;
    if delta_p > 10000 {
        delta_p = 10000;
    }
    let s = volume * MARTINGALE_FACTOR;
    let flash_fee = s * FLASH_FEE / 10000;
    let total_fees = s * FEE_RATE / 10000 + flash_fee;
    let f = total_fees + total_fees * BURN_RATE / 100;
    let gross = delta_p * s / 10000;
    let adj_gross = gross * 93 / 100;
    adj_gross > f
}
pub fn collapsed_profitable_m53(
    p: u128,
    stake: u128,
    target_price: u128,
    current_price: u128,
) -> (bool, u128) {
    let mod_bits = p % MOD_BITS;
    let mod_sym = p % MOD_SYM;
    let risk_approx = (1u128 << mod_bits as u32) - 1;
    let sym_factor = mod_sym / DIVISOR;
    let risk_collapsed = risk_approx * sym_factor;
    let reward = risk_collapsed * stake / DIVISOR;
    let passes = check_profitable(target_price, current_price, reward);
    (passes, reward)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_check_profitable() {
        assert!(check_profitable(191710, 194062, 1000));
    }
    #[test]
    fn test_collapsed_profitable_m53() {
        let (passes, reward) = collapsed_profitable_m53(194062501, 1000, 191710, 194062);
        assert!(passes);
        assert!(reward > 0);
    }
}
