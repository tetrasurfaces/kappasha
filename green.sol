// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@openzeppelin/contracts/utils/math/SafeMath.sol";

contract green.sol {
    using SafeMath for uint256;

    uint256 public constant FEE_RATE = 30;  // 0.003 * 10^4 (scaled)
    uint256 public constant MARTINGALE_FACTOR = 2;
    uint256 public constant DIVISOR = 3;
    uint256 public constant MOD_BITS = 256;
    uint256 public constant MOD_SYM = 369;
    uint256 public constant FLASH_FEE = 25;  // 0.0025 * 10^4
    uint256 public constant BURN_RATE = 50;  // 0.5 * 100

    // Greenpaper: Δp * s > f (scaled prices: e.g., 191710 for 191.710)
    function checkProfitable(
        uint256 targetPrice,
        uint256 currentPrice,
        uint256 volume
    ) public pure returns (bool) {
        uint256 deltaP = ((currentPrice > targetPrice ? currentPrice - targetPrice : targetPrice - currentPrice)
                          .mul(10000).div(targetPrice));  // Δp * 10^4, cap 10000
        if (deltaP > 10000) deltaP = 10000;
        uint256 s = volume.mul(MARTINGALE_FACTOR);
        uint256 flashFee = s.mul(FLASH_FEE).div(10000);
        uint256 totalFees = s.mul(FEE_RATE).div(10000).add(flashFee);
        uint256 f = totalFees.add(totalFees.mul(BURN_RATE).div(100));
        uint256 gross = deltaP.mul(s).div(10000);  // Unscaled
        // Risk adj (skew/vol/funding approx 0.93 → mul 93/100)
        uint256 adjGross = gross.mul(93).div(100);
        return adjGross > f;
    }

    // M53 Collapse + Profit Check (TOC 45)
    function collapsedProfitableM53(
        uint256 p,  // Exponent, e.g., 194062501
        uint256 stake,
        uint256 targetPrice,
        uint256 currentPrice
    ) public pure returns (bool, uint256) {
        uint256 modBits = p % MOD_BITS;  // 165
        uint256 modSym = p % MOD_SYM;  // 235
        uint256 riskApprox = (1 << modBits) - 1;  // 2^modBits -1 (shift for pow)
        uint256 symFactor = modSym / DIVISOR;  // 78 (trunc; mirror frac via post-div)
        uint256 riskCollapsed = riskApprox.mul(symFactor);
        uint256 reward = riskCollapsed.mul(stake).div(DIVISOR);  // ~1.22e69
        // Use as scaled volume
        bool passes = checkProfitable(targetPrice, currentPrice, reward);
        return (passes, reward);
    }
}
