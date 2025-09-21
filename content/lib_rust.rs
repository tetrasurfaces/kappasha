// Copyright (C) 2025 BlockChan Contributors
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.

use anchor_lang::prelude::*;
use anchor_spl::token::{self, Burn, MintTo, Token, TokenAccount, Transfer};
use pyth_sdk_solana::{load_price_account, PriceAccount};
use solana_program::{program::invoke, system_instruction, clock::Clock};
use std::cmp::max;

// Constants for program IDs and mints (fixed to 32 bytes)
const ORCA_WHIRLPOOL_PROGRAM_ID: Pubkey = Pubkey::new_from_array([
    0x6d, 0x8b, 0x6b, 0x1f, 0x6c, 0x8b, 0x8b, 0x7c, 0x3c, 0x1f, 0x7c, 0x3c, 0x1f, 0x7c,
    0x3c, 0x1f, 0x7c, 0x3c, 0x1f, 0x7c, 0x3c, 0x1f, 0x7c, 0x3c, 0x1f, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00,
]);
const ORCA_WHIRLPOOLS_CONFIG: Pubkey = Pubkey::new_from_array([
    0x2e, 0x1c, 0x7b, 0x8d, 0x1f, 0x6c, 0x8b, 0x8b, 0x7c, 0x3c, 0x1f, 0x7c, 0x3c, 0x1f,
    0x7c, 0x3c, 0x1f, 0x7c, 0x3c, 0x1f, 0x7c, 0x3c, 0x1f, 0x7c, 0x3c, 0x1f, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00,
]);
const LLLP_MINT: Pubkey = Pubkey::new_from_array([
    0x4b, 0x25, 0x7f, 0xe6, 0x1f, 0x6c, 0x8b, 0x8b, 0x7c, 0x3c, 0x1f, 0x7c, 0x3c, 0x1f,
    0x7c, 0x3c, 0x1f, 0x7c, 0x3c, 0x1f, 0x7c, 0x3c, 0x1f, 0x7c, 0x3c, 0x1f, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00,
]);
const USDT_MINT: Pubkey = Pubkey::new_from_array([
    0x4b, 0x25, 0x7f, 0xe6, 0x1f, 0x6c, 0x8b, 0x8b, 0x7c, 0x3c, 0x1f, 0x7c, 0x3c, 0x1f,
    0x7c, 0x3c, 0x1f, 0x7c, 0x3c, 0x1f, 0x7c, 0x3c, 0x1f, 0x7c, 0x3c, 0x1f, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00,
]);
const XAUT_MINT: Pubkey = Pubkey::new_from_array([
    0x4b, 0x25, 0x7f, 0xe6, 0x1f, 0x6c, 0x8b, 0x8b, 0x7c, 0x3c, 0x1f, 0x7c, 0x3c, 0x1f,
    0x7c, 0x3c, 0x1f, 0x7c, 0x3c, 0x1f, 0x7c, 0x3c, 0x1f, 0x7c, 0x3c, 0x1f, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00,
]);
const TOKEN_MINT: Pubkey = Pubkey::new_from_array([
    0x4a, 0x25, 0x7f, 0xe6, 0x1f, 0x6c, 0x8b, 0x8b, 0x7c, 0x3c, 0x1f, 0x7c, 0x3c, 0x1f,
    0x7c, 0x3c, 0x1f, 0x7c, 0x3c, 0x1f, 0x7c, 0x3c, 0x1f, 0x7c, 0x3c, 0x1f, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00,
]);

declare_id!("YourProtocolProgramId");

#[program]
pub mod surface_tension {
    use super::*;

    // Initialize a new MiniDex account with specified tick and token mint
    pub fn create_minidex(ctx: Context<CreateMiniDex>, tick: i64, token_mint: Pubkey) -> Result<()> {
        require!(token_mint == USDT_MINT || token_mint == XAUT_MINT, ErrorCode::InvalidToken);
        let mini_dex = &mut ctx.accounts.mini_dex;
        mini_dex.tick = tick;
        mini_dex.filled = 0;
        mini_dex.fees_collected = 0;
        mini_dex.token_mint = token_mint;
        mini_dex.is_buy = false;
        emit!(MiniDexCreated { token_mint });
        Ok(())
    }

    // Execute a flash loan swap with martingale-weighted amount
    pub fn flash_loan_swap(
        ctx: Context<FlashLoanSwap>,
        amount: u64,
        input_mint: Pubkey,
        output_mint: Pubkey,
        target_price: Option<f64>,
        martingale_factor: f64,
    ) -> Result<()> {
        require!(
            input_mint == USDT_MINT || input_mint == XAUT_MINT || input_mint == LLLP_MINT,
            ErrorCode::InvalidToken
        );
        require!(
            output_mint == USDT_MINT || output_mint == XAUT_MINT || output_mint == LLLP_MINT,
            ErrorCode::InvalidToken
        );
        // Fetch and validate Pyth price data
        let price_data = load_price_account(&ctx.accounts.pyth_price.data.borrow())?;
        let current_price = price_data.get_current_price()?.price as f64 / 1_000_000_000.0;
        require!(
            price_data.last_updated > Clock::get()?.unix_timestamp - 60,
            ErrorCode::StalePrice
        );
        if let Some(target) = target_price {
            require!(
                (current_price - target).abs() <= current_price * 0.01,
                ErrorCode::PriceDeviation
            );
        }
        // Transfer loan amount from vault
        let flash_loan_amount = amount;
        token::transfer(
            CpiContext::new(
                ctx.accounts.token_program.to_account_info(),
                Transfer {
                    from: ctx.accounts.vault_token.to_account_info(),
                    to: ctx.accounts.swapper_token_account.to_account_info(),
                    authority: ctx.accounts.vault.to_account_info(),
                },
            ),
            flash_loan_amount,
        )?;
        // Apply martingale factor and execute swap
        let adjusted_amount = (flash_loan_amount as f64 * martingale_factor).round() as u64;
        let swap_amount = jupiter_swap(&ctx, input_mint, output_mint, adjusted_amount, current_price)?;
        let flash_loan_fee = (swap_amount as f64 * 0.0025 * martingale_factor).round() as u64;
        ctx.accounts.mini_dex.fees_collected += flash_loan_fee;
        // Repay loan with fee
        token::transfer(
            CpiContext::new(
                ctx.accounts.token_program.to_account_info(),
                Transfer {
                    from: ctx.accounts.swapper_token_account.to_account_info(),
                    to: ctx.accounts.vault_token.to_account_info(),
                    authority: ctx.accounts.user.to_account_info(),
                },
            ),
            flash_loan_amount + flash_loan_fee,
        )?;
        // Mint LP tokens as reward
        let lp_tokens = (swap_amount as f64 * 0.005).round() as u64;
        token::mint_to(
            CpiContext::new(
                ctx.accounts.token_program.to_account_info(),
                MintTo {
                    mint: ctx.accounts.lllp_mint.to_account_info(),
                    to: ctx.accounts.swapper_token_account.to_account_info(),
                    authority: ctx.accounts.landlord_pool.to_account_info(),
                },
            ),
            lp_tokens,
        )?;
        ctx.accounts.mini_dex.filled += swap_amount;
        ctx.accounts.swapper_profile.unclaimed_harvest += flash_loan_fee;
        emit!(FlashLoanSwapExecuted {
            user: ctx.accounts.user.key(),
            input_mint,
            output_mint,
            amount: swap_amount,
            fee: flash_loan_fee,
        });
        // Taker benefit: Reclaim lamports for efficiency
        let taker_benefit = (swap_amount as f64 * 0.5).round() as u64;
        reclaim_lamports_and_mint_lllp(&ctx, taker_benefit)?;
        Ok(())
    }

    // Handle unified actions: swaps, limit orders, tick adjustments, greedy fills
    pub fn unified_action(
        ctx: Context<UnifiedAction>,
        action_type: u8,
        amount: u64,
        input_mint: Pubkey,
        output_mint: Pubkey,
        target_price: Option<f64>,
        is_one_sided: bool,
        is_short: bool,
        stream_quantity: Option<u64>,
        stream_interval: Option<u64>,
        max_duration: Option<u64>,
    ) -> Result<()> {
        require!(
            input_mint == USDT_MINT || input_mint == XAUT_MINT || input_mint == LLLP_MINT,
            ErrorCode::InvalidToken
        );
        require!(
            output_mint == USDT_MINT || output_mint == XAUT_MINT || output_mint == LLLP_MINT,
            ErrorCode::InvalidToken
        );
        let pool = &mut ctx.accounts.landlord_pool;
        let profile = &mut ctx.accounts.swapper_profile;
        let mini_dex = &mut ctx.accounts.mini_dex;
        let limit_queue = &mut ctx.accounts.limit_queue;
        // Set compute budget
        invoke(
            &solana_program::compute_budget::ComputeBudgetInstruction::set_compute_unit_price(1000),
            &[],
        )?;
        // Fetch and validate price
        let price_data = load_price_account(&ctx.accounts.pyth_price.data.borrow())?;
        let current_price = price_data.get_current_price()?.price as f64 / 1_000_000_000.0;
        require!(
            price_data.last_updated > Clock::get()?.unix_timestamp - 60,
            ErrorCode::StalePrice
        );
        if let Some(target) = target_price {
            require!(
                (current_price - target).abs() <= current_price * 0.01,
                ErrorCode::PriceDeviation
            );
        }
        let tick = price_to_tick(target_price.unwrap_or(current_price));
        let vol_multiplier = calculate_volatility(&ctx.accounts.pyth_price);
        let tick_range = (vol_multiplier * 1000.0).round() as i64;
        let _metadata = Metadata::from_account_info(&ctx.accounts.metadata_account.to_account_info())?;
        let swap_path = SwapPath {
            input_mint,
            output_mint,
            amount,
            current_price,
        };
        match action_type {
            0 | 1 => {
                let quantity = stream_quantity.unwrap_or(1);
                let interval = stream_interval.unwrap_or((60.0 / vol_multiplier).round() as u64);
                let chunk = amount / quantity;
                require!(chunk <= pool.balance / 10, ErrorCode::VolumeCap);
                if action_type == 1 && is_one_sided {
                    if (is_short && current_price > target_price.unwrap_or(current_price)) || (!is_short && current_price < target_price.unwrap_or(current_price)) {
                        limit_queue.orders.push(Order {
                            user: ctx.accounts.user.key(),
                            target_price: target_price.unwrap_or(current_price),
                            max_amount: amount,
                            filled: 0,
                            token_mint: output_mint,
                            stream_quantity: quantity,
                            stream_interval: interval,
                            max_duration,
                            created_at: Clock::get()?.unix_timestamp,
                            is_buy: !is_short,
                        });
                        emit!(OrderQueued {
                            user: ctx.accounts.user.key(),
                            amount,
                            target_price: target_price.unwrap_or(current_price),
                        });
                        return Ok(());
                    }
                }
                let batch_size = max(1, quantity.min(5));
                for batch in (0..quantity).step_by(batch_size as usize) {
                    let batch_amount = chunk * batch_size as u64;
                    let liquidity_amount = (batch_amount as f64 * 0.5).round() as u64;
                    token::transfer(
                        CpiContext::new(
                            ctx.accounts.token_program.to_account_info(),
                            Transfer {
                                from: ctx.accounts.vault_token.to_account_info(),
                                to: ctx.accounts.raydium_farm_account.to_account_info(),
                                authority: ctx.accounts.vault.to_account_info(),
                            },
                        ),
                        liquidity_amount,
                    )?;
                    let swap_amount = jupiter_swap(&ctx, input_mint, output_mint, batch_amount, current_price)?;
                    mini_dex.filled += swap_amount;
                    mini_dex.is_buy = !is_short;
                    let flash_loan_fee = (batch_amount as f64 * 0.0025).round() as u64;
                    mini_dex.fees_collected += flash_loan_fee;
                    let lp_tokens = (batch_amount as f64 * 0.005).round() as u64;
                    token::mint_to(
                        CpiContext::new(
                            ctx.accounts.token_program.to_account_info(),
                            MintTo {
                                mint: pool.lllp_mint.to_account_info(),
                                to: ctx.accounts.lllp_account.to_account_info(),
                                authority: ctx.accounts.landlord_pool.to_account_info(),
                            },
                        ),
                        lp_tokens,
                    )?;
                    let fee = if mini_dex.is_buy {
                        calculate_maker_rebate(batch_amount, current_price)?
                    } else {
                        calculate_swap_fee(batch_amount, current_price, pool.balance)?
                    };
                    mini_dex.fees_collected += fee;
                    profile.unclaimed_harvest += (fee as f64 * 0.5).round() as u64;
                    let burn_amount = (fee as f64 * 0.5).round() as u64;
                    token::burn(
                        CpiContext::new(
                            ctx.accounts.token_program.to_account_info(),
                            Burn {
                                mint: pool.lllp_mint.to_account_info(),
                                from: ctx.accounts.lllp_account.to_account_info(),
                                authority: ctx.accounts.landlord_pool.to_account_info(),
                            },
                        ),
                        burn_amount,
                    )?;
                    pool.total_lllp_burnt += burn_amount;
                    profile.burnt_lllp_contribution += burn_amount;
                    pool.burnt_lllp_fees += fee;
                    emit!(SwapFilled {
                        user: ctx.accounts.user.key(),
                        amount: swap_amount,
                        fees_collected: flash_loan_fee + fee,
                    });
                }
            }
            2 => {
                mini_dex.tick = if is_short { tick - 1 } else { tick + 1 };
            }
            3 => {
                let mut total_filled = 0;
                let mut total_fees = 0;
                let mut i = 0;
                while i < limit_queue.orders.len() {
                    let order = &mut limit_queue.orders[i];
                    let fill_ratio = ((current_price - order.target_price).abs() / order.target_price).min(1.0);
                    let fill_amount = (order.max_amount as f64 * fill_ratio).round() as u64;
                    order.filled += fill_amount;
                    total_filled += fill_amount;
                    let fee = if order.is_buy {
                        calculate_maker_rebate(fill_amount, current_price)?
                    } else {
                        calculate_swap_fee(fill_amount, current_price, pool.balance)?
                    };
                    total_fees += fee;
                    if order.filled >= order.max_amount {
                        limit_queue.orders.remove(i);
                    } else {
                        i += 1;
                    }
                }
                mini_dex.filled += total_filled;
                mini_dex.fees_collected += total_fees;
                profile.unclaimed_harvest += (total_fees as f64 * 0.5).round() as u64;
                let burn_amount = (total_fees as f64 * 0.5).round() as u64;
                token::burn(
                    CpiContext::new(
                        ctx.accounts.token_program.to_account_info(),
                        Burn {
                            mint: pool.lllp_mint.to_account_info(),
                            from: ctx.accounts.lllp_account.to_account_info(),
                            authority: ctx.accounts.landlord_pool.to_account_info(),
                        },
                    ),
                    burn_amount,
                )?;
                pool.total_lllp_burnt += burn_amount;
                profile.burnt_lllp_contribution += burn_amount;
                pool.burnt_lllp_fees += total_fees;
                emit!(GreedyLimitFilled {
                    user: ctx.accounts.user.key(),
                    total_filled,
                    total_fees,
                });
            }
            _ => return err!(ErrorCode::InvalidAction),
        }
        Ok(())
    }

    // Partially fill a queued order if price conditions are met
    pub fn execute_partial_fill(ctx: Context<ExecutePartialFill>, order_idx: usize) -> Result<()> {
        let limit_queue = &mut ctx.accounts.limit_queue;
        let mini_dex = &mut ctx.accounts.mini_dex;
        let profile = &mut ctx.accounts.swapper_profile;
        let order = limit_queue.orders.get_mut(order_idx).ok_or(ErrorCode::InvalidOrderIndex)?;
        let current_price = load_price_account(&ctx.accounts.pyth_price.data.borrow())?.get_current_price()?.price as f64;
        let should_execute = if order.is_buy {
            current_price <= order.target_price
        } else {
            current_price >= order.target_price
        };
        require!(should_execute, ErrorCode::PriceNotReached);
        let fill_ratio = ((current_price - order.target_price).abs() / order.target_price).min(1.0);
        let fill_amount = (order.max_amount as f64 * fill_ratio).round() as u64;
        order.filled += fill_amount;
        let clock = Clock::get()?;
        if let Some(max_duration) = order.max_duration {
            if clock.unix_timestamp - order.created_at > max_duration {
                reclaim_lamports_and_mint_lllp(&ctx, order.max_amount - order.filled)?;
                limit_queue.orders.remove(order_idx);
                emit!(OrderCancelled {
                    user: ctx.accounts.user.key(),
                    amount: order.max_amount - order.filled,
                });
                return Ok(());
            }
        }
        let available = 10; // Mocked supply
        let fill = fill_amount.min(available);
        order.filled += fill;
        let ix = Instruction {
            program_id: ctx.program_id,
            accounts: ctx.accounts.to_account_metas(None),
            data: UnifiedAction::new(0, fill, order.token_mint, order.token_mint, None, false, !order.is_buy, None, None, None).data(),
        };
        invoke(&ix, &ctx.accounts.to_account_infos())?;
        if order.filled >= order.max_amount {
            reclaim_lamports_and_mint_lllp(&ctx, (order.max_amount as f64 * 0.5).round() as u64)?;
            limit_queue.orders.remove(order_idx);
        }
        emit!(OrderFilled {
            user: ctx.accounts.user.key(),
            amount: fill,
            price: current_price,
        });
        Ok(())
    }

    // Cancel a queued order and reclaim lamports
    pub fn cancel_order(ctx: Context<CancelOrder>, order_idx: usize) -> Result<()> {
        let limit_queue = &mut ctx.accounts.limit_queue;
        let order = limit_queue.orders.get(order_idx).ok_or(ErrorCode::InvalidOrderIndex)?;
        require!(order.user == ctx.accounts.user.key(), ErrorCode::Unauthorized);
        reclaim_lamports_and_mint_lllp(&ctx, order.max_amount - order.filled)?;
        limit_queue.orders.remove(order_idx);
        emit!(OrderCancelled {
            user: ctx.accounts.user.key(),
            amount: order.max_amount - order.filled,
        });
        Ok(())
    }

    // Harvest accumulated fees and burnt share for user
    pub fn aggregate_harvest(ctx: Context<AggregateHarvest>) -> Result<()> {
        let profile = &mut ctx.accounts.swapper_profile;
        let pool = &mut ctx.accounts.landlord_pool;
        let harvest_amount = profile.unclaimed_harvest;
        let burnt_share = if pool.total_lllp_burnt > 0 {
            (pool.burnt_lllp_fees as f64 * profile.burnt_lllp_contribution as f64 / pool.total_lllp_burnt as f64).round() as u64
        } else {
            0
        };
        let total_harvest = harvest_amount + burnt_share;
        require!(total_harvest > 0, ErrorCode::NoHarvest);
        token::transfer(
            CpiContext::new(
                ctx.accounts.token_program.to_account_info(),
                Transfer {
                    from: ctx.accounts.vault_token.to_account_info(),
                    to: ctx.accounts.user_ata.to_account_info(),
                    authority: ctx.accounts.vault.to_account_info(),
                },
            ),
            total_harvest,
        )?;
        profile.unclaimed_harvest = 0;
        profile.burnt_lllp_contribution = 0;
        emit!(HarvestClaimed {
            user: ctx.accounts.user.key(),
            amount: total_harvest,
        });
        Ok(())
    }
}

// Account structures
#[derive(Accounts)]
pub struct CreateMiniDex<'info> {
    #[account(init, payer = user, space = 8 + 8 + 8 + 8 + 32 + 1)]
    pub mini_dex: Account<'info, MiniDex>,
    #[account(mut)]
    pub user: Signer<'info>,
    pub system_program: Program<'info, System>,
}

#[derive(Accounts)]
pub struct FlashLoanSwap<'info> {
    #[account(mut)]
    pub mini_dex: Account<'info, MiniDex>,
    #[account(mut)]
    pub landlord_pool: Account<'info, LandlordPool>,
    #[account(mut)]
    pub swapper_profile: Account<'info, SwapperProfile>,
    #[account(mut)]
    pub vault: AccountInfo<'info>,
    #[account(mut)]
    pub vault_token: Account<'info, TokenAccount>,
    #[account(mut)]
    pub swapper_token_account: Account<'info, TokenAccount>,
    #[account(mut)]
    pub lllp_mint: Account<'info, Mint>,
    pub pyth_price: AccountLoader<'info, PriceAccount>,
    pub token_program: Program<'info, Token>,
    #[account(mut)]
    pub user: Signer<'info>,
}

#[derive(Accounts)]
pub struct UnifiedAction<'info> {
    #[account(mut)]
    pub mini_dex: Account<'info, MiniDex>,
    #[account(mut)]
    pub landlord_pool: Account<'info, LandlordPool>,
    #[account(mut)]
    pub swapper_profile: Account<'info, SwapperProfile>,
    #[account(mut)]
    pub limit_queue: Account<'info, LimitQueue>,
    #[account(mut)]
    pub vault: AccountInfo<'info>,
    #[account(mut)]
    pub vault_token: Account<'info, TokenAccount>,
    #[account(mut)]
    pub raydium_farm_account: AccountInfo<'info>,
    #[account(mut)]
    pub lllp_account: Account<'info, TokenAccount>,
    pub pyth_price: AccountLoader<'info, PriceAccount>,
    pub token_program: Program<'info, Token>,
    #[account(mut)]
    pub user: Signer<'info>,
    pub metadata_account: AccountInfo<'info>,
}

#[derive(Accounts)]
pub struct ExecutePartialFill<'info> {
    #[account(mut)]
    pub mini_dex: Account<'info, MiniDex>,
    #[account(mut)]
    pub limit_queue: Account<'info, LimitQueue>,
    #[account(mut)]
    pub swapper_profile: Account<'info, SwapperProfile>,
    #[account(mut)]
    pub vault: AccountInfo<'info>,
    #[account(mut)]
    pub vault_token: Account<'info, TokenAccount>,
    #[account(mut)]
    pub raydium_farm_account: AccountInfo<'info>,
    #[account(mut)]
    pub lllp_mint: Account<'info, Mint>,
    #[account(mut)]
    pub landlord_pool: Account<'info, LandlordPool>,
    #[account(mut)]
    pub user_ata: Account<'info, TokenAccount>,
    pub pyth_price: AccountLoader<'info, PriceAccount>,
    pub token_program: Program<'info, Token>,
    pub system_program: Program<'info, System>,
    #[account(mut)]
    pub user: Signer<'info>,
}

#[derive(Accounts)]
pub struct CancelOrder<'info> {
    #[account(mut)]
    pub limit_queue: Account<'info, LimitQueue>,
    #[account(mut)]
    pub vault: AccountInfo<'info>,
    #[account(mut)]
    pub vault_token: Account<'info, TokenAccount>,
    #[account(mut)]
    pub raydium_farm_account: AccountInfo<'info>,
    #[account(mut)]
    pub lllp_mint: Account<'info, Mint>,
    #[account(mut)]
    pub landlord_pool: Account<'info, LandlordPool>,
    #[account(mut)]
    pub user_ata: Account<'info, TokenAccount>,
    pub token_program: Program<'info, Token>,
    pub system_program: Program<'info, System>,
    #[account(mut)]
    pub user: Signer<'info>,
}

#[derive(Accounts)]
pub struct AggregateHarvest<'info> {
    #[account(mut)]
    pub swapper_profile: Account<'info, SwapperProfile>,
    #[account(mut)]
    pub landlord_pool: Account<'info, LandlordPool>,
    #[account(mut)]
    pub vault: AccountInfo<'info>,
    #[account(mut)]
    pub vault_token: Account<'info, TokenAccount>,
    #[account(mut)]
    pub user_ata: Account<'info, TokenAccount>,
    pub token_program: Program<'info, Token>,
    #[account(mut)]
    pub user: Signer<'info>,
}

#[account]
pub struct MiniDex {
    pub tick: i64,
    pub filled: u64,
    pub fees_collected: u64,
    pub token_mint: Pubkey,
    pub is_buy: bool,
}

#[account]
pub struct LandlordPool {
    pub balance: u64,
    pub total_lllp_burnt: u64,
    pub burnt_lllp_fees: u64,
    pub lllp_mint: Pubkey,
}

#[account]
pub struct SwapperProfile {
    pub burnt_lllp_contribution: u64,
    pub unclaimed_harvest: u64,
    pub nft_keys: NftKeys,
}

#[account]
pub struct LimitQueue {
    pub orders: Vec<Order>,
}

#[derive(AnchorSerialize, AnchorDeserialize, Clone)]
pub struct Order {
    pub user: Pubkey,
    pub target_price: f64,
    pub max_amount: u64,
    pub filled: u64,
    pub token_mint: Pubkey,
    pub stream_quantity: u64,
    pub stream_interval: u64,
    pub max_duration: Option<u64>,
    pub created_at: i64,
    pub is_buy: bool,
}

#[derive(AnchorSerialize, AnchorDeserialize, Clone)]
pub struct NftKeys {
    pub unclaimed_harvest: u64,
}

#[derive(AnchorSerialize, AnchorDeserialize)]
pub struct Metadata {
    pub data: MetadataData,
}

#[derive(AnchorSerialize, AnchorDeserialize)]
pub struct MetadataData {
    pub uri: Option<String>,
}

#[derive(AnchorSerialize, AnchorDeserialize)]
pub struct SwapPath {
    pub input_mint: Pubkey,
    pub output_mint: Pubkey,
    pub amount: u64,
    pub current_price: f64,
}

#[event]
pub struct MiniDexCreated {
    pub token_mint: Pubkey,
}

#[event]
pub struct OrderQueued {
    pub user: Pubkey,
    pub amount: u64,
    pub target_price: f64,
}

#[event]
pub struct OrderFilled {
    pub user: Pubkey,
    pub amount: u64,
    pub price: f64,
}

#[event]
pub struct OrderCancelled {
    pub user: Pubkey,
    pub amount: u64,
}

#[event]
pub struct SwapFilled {
    pub user: Pubkey,
    pub amount: u64,
    pub fees_collected: u64,
}

#[event]
pub struct HarvestClaimed {
    pub user: Pubkey,
    pub amount: u64,
}

#[event]
pub struct GreedyLimitFilled {
    pub user: Pubkey,
    pub total_filled: u64,
    pub total_fees: u64,
}

#[error_code]
pub enum ErrorCode {
    #[msg("Price deviation exceeds allowed range")]
    PriceDeviation,
    #[msg("Invalid action type")]
    InvalidAction,
    #[msg("Invalid order index")]
    InvalidOrderIndex,
    #[msg("Price not reached for order execution")]
    PriceNotReached,
    #[msg("No harvest available to claim")]
    NoHarvest,
    #[msg("Volume exceeds pool capacity")]
    VolumeCap,
    #[msg("Unauthorized user")]
    Unauthorized,
    #[msg("Invalid token mint")]
    InvalidToken,
    #[msg("Stale price data")]
    StalePrice,
}

// Generic jupiter_swap to support multiple contexts
fn jupiter_swap<'info, T: Accounts<'info>>(
    ctx: &Context<T>,
    input_mint: Pubkey,
    output_mint: Pubkey,
    amount: u64,
    _current_price: f64,
) -> Result<u64> {
    // Placeholder: Simulate Jupiter aggregator swap
    // In production, this would call Jupiter's program with proper accounts
    let ix = anchor_lang::solana_program::instruction::Instruction {
        program_id: ctx.program_id,
        accounts: ctx.accounts.to_account_metas(None),
        data: UnifiedAction::new(0, amount, input_mint, output_mint, None, false, false, None, None, None).data(),
    };
    invoke(&ix, &ctx.accounts.to_account_infos())?;
    Ok(amount)
}

// Reclaim lamports and mint LLLP tokens for refunds or incentives
fn reclaim_lamports_and_mint_lllp<'info, T: Accounts<'info>>(ctx: &Context<T>, amount: u64) -> Result<()> {
    // Minimum rent-exempt balance for a basic account (~2,039,280 lamports)
    const MIN_RENT_EXEMPT: u64 = 2_039_280;
    // Calculate lamports to reclaim (mock: proportional to amount, capped at min rent-exempt)
    let lamports_to_reclaim = MIN_RENT_EXEMPT.min(amount);
    
    // Transfer lamports from vault to user
    invoke(
        &system_instruction::transfer(
            ctx.accounts.vault.key(),
            ctx.accounts.user.key(),
            lamports_to_reclaim,
        ),
        &[
            ctx.accounts.vault.to_account_info(),
            ctx.accounts.user.to_account_info(),
            ctx.accounts.system_program.to_account_info(),
        ],
    )?;

    // Mint LLLP tokens proportional to amount (e.g., 1% of amount as tokens)
    let lllp_to_mint = (amount as f64 * 0.01).round() as u64;
    if lllp_to_mint > 0 {
        token::mint_to(
            CpiContext::new(
                ctx.accounts.token_program.to_account_info(),
                MintTo {
                    mint: ctx.accounts.lllp_mint.to_account_info(),
                    to: ctx.accounts.user_ata.to_account_info(),
                    authority: ctx.accounts.landlord_pool.to_account_info(),
                },
            ),
            lllp_to_mint,
        )?;
        // Update pool balance
        ctx.accounts.landlord_pool.balance += lllp_to_mint;
    }

    Ok(())
}

fn calculate_swap_fee(amount: u64, price: f64, pool_balance: u64) -> Result<u64> {
    let fee_rate = 0.015;
    Ok(((amount as f64 * price * fee_rate) / pool_balance as f64).round() as u64)
}

fn calculate_maker_rebate(amount: u64, price: f64) -> Result<u64> {
    let rebate_rate = 0.005;
    Ok((amount as f64 * price * rebate_rate).round() as u64)
}

fn calculate_volatility(pyth_price: &AccountLoader<PriceAccount>) -> f64 {
    let price_data = load_price_account(&pyth_price.data.borrow()).unwrap();
    let prices = price_data.get_price_history(10).unwrap_or_default();
    if prices.is_empty() {
        return 0.01;
    }
    let mean = prices.iter().sum::<f64>() / prices.len() as f64;
    let variance = prices.iter().map(|&p| (p - mean).powi(2)).sum::<f64>() / prices.len() as f64;
    variance.sqrt() / mean
}

fn price_to_tick(price: f64) -> i64 {
    const TICK_BASE: f64 = 1.000001;
    const BASE_PRICE: f64 = 100.0;
    (price.ln() / BASE_PRICE.ln()).round() as i64
}
