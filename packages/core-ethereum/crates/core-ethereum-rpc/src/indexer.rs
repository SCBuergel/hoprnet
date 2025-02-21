use async_stream::stream;
use async_trait::async_trait;
use ethers::types::BlockNumber;
use ethers_providers::{JsonRpcClient, Middleware};
use futures::{Stream, TryStreamExt};
use log::debug;
use log::error;
use std::pin::Pin;

use crate::errors::{Result, RpcError::FilterIsEmpty};
use crate::rpc::RpcOperations;
use crate::{BlockWithLogs, HoprIndexerRpcOperations, Log, LogFilter};

#[async_trait]
impl<P: JsonRpcClient + 'static> HoprIndexerRpcOperations for RpcOperations<P> {
    async fn block_number(&self) -> Result<u64> {
        Ok(self.provider.get_block_number().await?.as_u64())
    }

    fn try_stream_logs<'a>(
        &'a self,
        start_block_number: u64,
        filter: LogFilter,
    ) -> Result<Pin<Box<dyn Stream<Item = BlockWithLogs> + Send + 'a>>> {
        if filter.is_empty() {
            return Err(FilterIsEmpty);
        }

        Ok(Box::pin(stream! {
            // On first iteration use the given block number as start
            let mut from_block = start_block_number;

            loop {
                match self.block_number().await {
                    Ok(latest_block) => {
                        // If on first iteration the start block is in the future, just set it to latest
                        if from_block == start_block_number && from_block > latest_block {
                            from_block = latest_block;
                        }

                        // This is a hard-failure on subsequent iterations which is unrecoverable
                        // (e.g. Anvil restart in the background when testing and `latest_block` jumps below `from_block`)
                        assert!(latest_block >= from_block, "indexer start block number is greater than the chain latest block number");

                        // Range is inclusive
                        let range_filter = ethers::types::Filter::from(filter.clone())
                            .from_block(BlockNumber::Number(from_block.into()))
                            .to_block(BlockNumber::Number(latest_block.into()));

                        if from_block != latest_block {
                            debug!("polling logs from blocks #{from_block} - #{latest_block}");
                        } else {
                             debug!("polling logs from block #{from_block}");
                        }

                        // The provider internally performs retries on timeouts and errors.
                        let mut retrieved_logs = self.provider.get_logs_paginated(&range_filter, self.cfg.logs_page_size);

                        let mut current_block_log = BlockWithLogs { block_id: from_block, ..Default::default()};
                        while let Ok(Some(log)) = retrieved_logs.try_next().await {
                            let log = Log::from(log);

                            // This assumes the logs are arriving ordered by blocks
                            if current_block_log.block_id != log.block_number {
                                debug!("completed {current_block_log}");
                                yield current_block_log;

                                current_block_log = BlockWithLogs::default();
                                current_block_log.block_id = log.block_number;
                            }

                            debug!("retrieved {log}");
                            current_block_log.logs.push(log);
                        }

                        debug!("retrieved complete {current_block_log}");

                        yield current_block_log;
                        from_block = latest_block + 1;
                    }
                    Err(e) => error!("failed to obtain current block number from chain: {e}")
                }

                futures_timer::Delay::new(self.cfg.expected_block_time).await;
            }
        }))
    }
}

#[cfg(test)]
mod test {
    use async_std::prelude::FutureExt;
    use std::time::Duration;

    use ethers::contract::EthEvent;
    use ethers_providers::Middleware;
    use futures::StreamExt;

    use bindings::hopr_channels::*;
    use bindings::hopr_token::{ApprovalFilter, HoprToken, TransferFilter};
    use core_crypto::keypairs::{ChainKeypair, Keypair};
    use core_ethereum_types::{create_anvil, ContractAddresses, ContractInstances};
    use log::debug;
    use utils_types::primitives::Address;

    use crate::client::native::SurfRequestor;
    use crate::client::{create_rpc_client_to_anvil, JsonRpcProviderClient, SimpleJsonRpcRetryPolicy};
    use crate::rpc::tests::mint_tokens;
    use crate::rpc::{RpcOperations, RpcOperationsConfig};
    use crate::{BlockWithLogs, HoprIndexerRpcOperations, LogFilter};

    async fn fund_channel<M: Middleware + 'static>(
        counterparty: Address,
        hopr_token: HoprToken<M>,
        hopr_channels: HoprChannels<M>,
    ) {
        hopr_token
            .approve(hopr_channels.address(), 1u128.into())
            .send()
            .await
            .unwrap()
            .await
            .unwrap();

        hopr_channels
            .fund_channel(counterparty.into(), 1u128)
            .send()
            .await
            .unwrap()
            .await
            .unwrap();
    }

    #[async_std::test]
    async fn test_should_get_block_number() {
        let anvil = create_anvil(Some(Duration::from_secs(1)));
        let chain_key_0 = ChainKeypair::from_secret(anvil.keys()[0].to_bytes().as_ref()).unwrap();

        let client = JsonRpcProviderClient::new(&anvil.endpoint(), SurfRequestor::default());

        let rpc = RpcOperations::new(client, &chain_key_0, Default::default(), SimpleJsonRpcRetryPolicy)
            .expect("failed to construct rpc");

        let b1 = rpc.block_number().await.expect("should get block number");
        async_std::task::sleep(Duration::from_secs(2)).await;
        let b2 = rpc.block_number().await.expect("should get block number");

        assert!(b2 > b1, "block number should increase");
    }

    #[async_std::test]
    async fn test_try_stream_logs_should_contain_all_logs_when_opening_channel() {
        let _ = env_logger::builder().is_test(true).try_init();

        let block_time = Duration::from_secs(1);

        let anvil = create_anvil(Some(block_time));
        let chain_key_0 = ChainKeypair::from_secret(anvil.keys()[0].to_bytes().as_ref()).unwrap();
        let chain_key_1 = ChainKeypair::from_secret(anvil.keys()[1].to_bytes().as_ref()).unwrap();

        // Deploy contracts
        let contract_instances = {
            let client = create_rpc_client_to_anvil(SurfRequestor::default(), &anvil, &chain_key_0);
            ContractInstances::deploy_for_testing(client, &chain_key_0)
                .await
                .expect("could not deploy contracts")
        };

        let tokens_minted_at = mint_tokens(contract_instances.token.clone(), 1000_u128, (&chain_key_0).into()).await;
        debug!("tokens were minted at block {tokens_minted_at}");

        let contract_addrs = ContractAddresses::from(&contract_instances);

        let cfg = RpcOperationsConfig {
            tx_polling_interval: Duration::from_millis(10),
            contract_addrs: contract_addrs.clone(),
            expected_block_time: block_time,
            ..RpcOperationsConfig::default()
        };

        let client = JsonRpcProviderClient::new(&anvil.endpoint(), SurfRequestor::default());

        let rpc =
            RpcOperations::new(client, &chain_key_0, cfg, SimpleJsonRpcRetryPolicy).expect("failed to construct rpc");

        let log_filter = LogFilter {
            address: vec![contract_addrs.token, contract_addrs.channels],
            topics: vec![
                TransferFilter::signature().into(),
                ApprovalFilter::signature().into(),
                ChannelOpenedFilter::signature().into(),
                ChannelBalanceIncreasedFilter::signature().into(),
            ],
        };

        debug!("{:#?}", contract_addrs);
        debug!("{:#?}", log_filter);

        // Spawn channel funding
        async_std::task::spawn(async move {
            fund_channel(
                chain_key_1.public().to_address(),
                contract_instances.token,
                contract_instances.channels,
            )
            .delay(block_time * 2)
            .await;
        });

        // Spawn stream
        let count_filtered_topics = log_filter.topics.len();
        let retrieved_logs = rpc
            .try_stream_logs(1, log_filter)
            .expect("must create stream")
            .skip_while(|b| futures::future::ready(b.len() != count_filtered_topics))
            .take(1)
            .collect::<Vec<BlockWithLogs>>()
            .timeout(Duration::from_secs(30))
            .await
            .expect("timeout"); // Everything must complete within 30 seconds

        // The last block must contain all 4 events
        let last_block_logs = retrieved_logs.last().unwrap().clone().logs;

        assert!(
            last_block_logs.iter().any(|log| log.address == contract_addrs.channels
                && log.topics.contains(&ChannelOpenedFilter::signature().0.into())),
            "must contain channel open"
        );
        assert!(
            last_block_logs.iter().any(|log| log.address == contract_addrs.channels
                && log
                    .topics
                    .contains(&ChannelBalanceIncreasedFilter::signature().0.into())),
            "must contain channel balance increase"
        );
        assert!(
            last_block_logs
                .iter()
                .any(|log| log.address == contract_addrs.token
                    && log.topics.contains(&ApprovalFilter::signature().0.into())),
            "must contain token approval"
        );
        assert!(
            last_block_logs
                .iter()
                .any(|log| log.address == contract_addrs.token
                    && log.topics.contains(&TransferFilter::signature().0.into())),
            "must contain token transfer"
        );
    }

    #[async_std::test]
    async fn test_try_stream_logs_should_contain_only_channel_logs_when_filtered_on_funding_channel() {
        let _ = env_logger::builder().is_test(true).try_init();

        let block_time = Duration::from_secs(1);

        let anvil = create_anvil(Some(block_time));
        let chain_key_0 = ChainKeypair::from_secret(anvil.keys()[0].to_bytes().as_ref()).unwrap();
        let chain_key_1 = ChainKeypair::from_secret(anvil.keys()[1].to_bytes().as_ref()).unwrap();

        // Deploy contracts
        let contract_instances = {
            let client = create_rpc_client_to_anvil(SurfRequestor::default(), &anvil, &chain_key_0);
            ContractInstances::deploy_for_testing(client, &chain_key_0)
                .await
                .expect("could not deploy contracts")
        };

        let tokens_minted_at = mint_tokens(contract_instances.token.clone(), 1000_u128, (&chain_key_0).into()).await;
        debug!("tokens were minted at block {tokens_minted_at}");

        let contract_addrs = ContractAddresses::from(&contract_instances);

        let cfg = RpcOperationsConfig {
            tx_polling_interval: Duration::from_millis(10),
            contract_addrs: contract_addrs.clone(),
            expected_block_time: block_time,
            ..RpcOperationsConfig::default()
        };

        let client = JsonRpcProviderClient::new(&anvil.endpoint(), SurfRequestor::default());

        let rpc =
            RpcOperations::new(client, &chain_key_0, cfg, SimpleJsonRpcRetryPolicy).expect("failed to construct rpc");

        let log_filter = LogFilter {
            address: vec![contract_addrs.channels],
            topics: vec![
                ChannelOpenedFilter::signature().into(),
                ChannelBalanceIncreasedFilter::signature().into(),
            ],
        };

        debug!("{:#?}", contract_addrs);
        debug!("{:#?}", log_filter);

        // Spawn channel funding
        async_std::task::spawn(async move {
            fund_channel(
                chain_key_1.public().to_address(),
                contract_instances.token,
                contract_instances.channels,
            )
            .delay(block_time * 2)
            .await;
        });

        // Spawn stream
        let count_filtered_topics = log_filter.topics.len();
        let retrieved_logs = rpc
            .try_stream_logs(1, log_filter)
            .expect("must create stream")
            .skip_while(|b| futures::future::ready(b.len() != count_filtered_topics))
            .take(1)
            .collect::<Vec<BlockWithLogs>>()
            .timeout(Duration::from_secs(30))
            .await
            .expect("timeout"); // Everything must complete within 30 seconds

        // The last block must contain all 2 events
        let last_block_logs = retrieved_logs.first().unwrap().clone().logs;

        assert!(
            last_block_logs.iter().any(|log| log.address == contract_addrs.channels
                && log.topics.contains(&ChannelOpenedFilter::signature().0.into())),
            "must contain channel open"
        );
        assert!(
            last_block_logs.iter().any(|log| log.address == contract_addrs.channels
                && log
                    .topics
                    .contains(&ChannelBalanceIncreasedFilter::signature().0.into())),
            "must contain channel balance increase"
        );
    }
}
