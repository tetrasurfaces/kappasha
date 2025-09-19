// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@openzeppelin/contracts/utils/math/SafeMath.sol";

contract Green {
    using SafeMath for uint256;

    uint256 public constant FEE_RATE = 30;
    uint256 public constant MARTINGALE_FACTOR = 2;
    uint256 public constant DIVISOR = 3;
    uint256 public constant MOD_BITS = 256;
    uint256 public constant MOD_SYM = 369;
    uint256 public constant FLASH_FEE = 25;
    uint256 public constant BURN_RATE = 50;

    function checkProfitable(
        uint256 targetPrice,
        uint256 currentPrice,
        uint256 volume
    ) public pure returns (bool) {
        uint256 deltaP = ((currentPrice > targetPrice ? currentPrice - targetPrice : targetPrice - currentPrice)
                          .mul(10000).div(targetPrice));
        if (deltaP > 10000) deltaP = 10000;
        uint256 s = volume.mul(MARTINGALE_FACTOR);
        uint256 flashFee = s.mul(FLASH_FEE).div(10000);
        uint256 totalFees = s.mul(FEE_RATE).div(10000).add(flashFee);
        uint256 f = totalFees.add(totalFees.mul(BURN_RATE).div(100));
        uint256 gross = deltaP.mul(s).div(10000);
        uint256 adjGross = gross.mul(93).div(100);
        return adjGross > f;
    }

    function collapsedProfitableM53(
        uint256 p,
        uint256 stake,
        uint256 targetPrice,
        uint256 currentPrice
    ) public pure returns (bool, uint256) {
        uint256 modBits = p % MOD_BITS;
        uint256 modSym = p % MOD_SYM;
        uint256 riskApprox = (1 << modBits) - 1;
        uint256 symFactor = modSym / DIVISOR;
        uint256 riskCollapsed = riskApprox.mul(symFactor);
        uint256 reward = riskCollapsed.mul(stake).div(DIVISOR);
        bool passes = checkProfitable(targetPrice, currentPrice, reward);
        return (passes, reward);
    }

    function mapSequenceToStake(uint256 index) public pure returns (uint256) {
        uint256 units = (index % 18) + 1;
        uint256 tens = (index / 18 % 9) + 1;
        uint256 digits = 1;
        uint256 temp = units;
        while (temp >= 10) {
            digits++;
            temp /= 10;
        }
        uint256 base = 10 ** digits;
        return tens * base + units;
    }
}
