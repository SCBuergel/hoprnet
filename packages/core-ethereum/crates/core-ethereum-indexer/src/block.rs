use async_lock::RwLock;
use core_crypto::types::Hash;
use futures::{pin_mut, StreamExt};
use log::{debug, error, info};
use std::{collections::VecDeque, sync::Arc};

use core_ethereum_db::traits::HoprCoreEthereumDbActions;
use core_ethereum_rpc::{HoprIndexerRpcOperations, Log, LogFilter};
use core_ethereum_types::chain_events::SignificantChainEvent;
use utils_types::primitives::{Snapshot, U256};

use crate::{errors::CoreEthereumIndexerError, traits::ChainLogHandler};

#[cfg(all(feature = "prometheus", not(test)))]
use utils_metrics::metrics::{MultiCounter, MultiGauge, SimpleCounter, SimpleGauge};

use async_std::task::spawn;

#[cfg(all(feature = "prometheus", not(test)))]
lazy_static::lazy_static! {
    static ref METRIC_INDEXER_ERROR_COUNT: MultiCounter =
        MultiCounter::new(
            "hopr_indexer_errors_count",
            "Multicounter for provider errors in Indexer",
            &["type"]
    ).unwrap();
    static ref METRIC_INDEXER_UNCONFIRMED_BLOCK_COUNT: SimpleCounter =
        SimpleCounter::new(
            "hopr_indexer_processed_unconfirmed_blocks_count",
            "Number of processed unconfirmed blocks",
    ).unwrap();
    static ref METRIC_INDEXER_ANNOUNCEMENTS_COUNT: SimpleCounter =
        SimpleCounter::new(
            "hopr_indexer_processed_announcements_count",
            "Number of processed announcements",
    ).unwrap();
    static ref METRIC_INDEXER_OBSERVED_CHANNEL_STATUS: MultiGauge =
        MultiGauge::new(
            "core_indexer_channel_statuses",
            "Status of different channels",
            &["channel"]
        ).unwrap();
    static ref METRIC_INDEXER_CURRENT_BLOCK: SimpleGauge =
        SimpleGauge::new(
            "core_ethereum_gauge_indexer_block_number",
            "Current block number",
    ).unwrap();
}

fn log_comparator(left: &Log, right: &Log) -> std::cmp::Ordering {
    let blocks = left.block_number.cmp(&right.block_number);
    if blocks == std::cmp::Ordering::Equal {
        let tx_indices = left.tx_index.cmp(&right.tx_index);
        if tx_indices == std::cmp::Ordering::Equal {
            left.log_index.cmp(&right.log_index)
        } else {
            tx_indices
        }
    } else {
        blocks
    }
}

#[derive(Debug, Clone, Copy)]
pub struct IndexerConfig {
    /// Finalization chain length
    ///
    /// The number of blocks including and decreasing from the chain HEAD
    /// that the logs will be buffered for before being considered
    /// successfully joined to the chain.
    pub finalization: u64,
    /// The block at which the indexer should start
    ///
    /// It typically makes little sense to start indexing from the beginning
    /// of the chain, all that is sufficient is to start indexing since the
    /// relevant smart contracts were introduced into the chain.
    ///
    /// This value makes sure that indexing is relevant and as minimal as possible.
    pub start_block_number: u64,
    /// Fetch token transactions
    ///
    /// Whether the token transaction topics should also be fetched.
    pub fetch_token_transactions: bool,
}

impl Default for IndexerConfig {
    fn default() -> Self {
        Self {
            finalization: 8,
            start_block_number: 0,
            fetch_token_transactions: true,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Indexer<T, U, V>
where
    T: HoprIndexerRpcOperations + Send + 'static,
    U: ChainLogHandler + Send + 'static,
    V: HoprCoreEthereumDbActions + Send + Sync + 'static,
{
    rpc: Option<T>,
    db_processor: Option<U>,
    db: Arc<RwLock<V>>,
    cfg: IndexerConfig,
    egress: futures::channel::mpsc::UnboundedSender<SignificantChainEvent>,
}

impl<T, U, V> Indexer<T, U, V>
where
    T: HoprIndexerRpcOperations + Send + 'static,
    U: ChainLogHandler + Send + 'static,
    V: HoprCoreEthereumDbActions + Send + Sync + 'static,
{
    pub fn new(
        rpc: T,
        db_processor: U,
        db: Arc<RwLock<V>>,
        cfg: IndexerConfig,
        egress: futures::channel::mpsc::UnboundedSender<SignificantChainEvent>,
    ) -> Self {
        Self {
            rpc: Some(rpc),
            db_processor: Some(db_processor),
            db,
            cfg,
            egress,
        }
    }

    pub async fn start(&mut self) -> crate::errors::Result<()>
    where
        T: HoprIndexerRpcOperations + 'static,
        U: ChainLogHandler + 'static,
        V: HoprCoreEthereumDbActions + 'static,
    {
        if self.rpc.is_none() || self.db_processor.is_none() {
            return Err(CoreEthereumIndexerError::ProcessError(
                "indexer is already started".into(),
            ));
        }

        info!("Starting indexer...");

        let rpc = self.rpc.take().expect("rpc should be present");
        let db_processor = self.db_processor.take().expect("db_processor should be present");
        let db = self.db.clone();
        let tx_significant_events = self.egress.clone();

        let latest_block_in_db = self
            .db
            .read()
            .await
            .get_latest_block_number()
            .await?
            .map(|v| v as u64)
            .unwrap_or(self.cfg.start_block_number);

        info!("Latest saved block {:?}", latest_block_in_db);

        let mut topics = vec![];
        topics.extend(crate::constants::topics::announcement());
        topics.extend(crate::constants::topics::channel());
        topics.extend(crate::constants::topics::node_safe_registry());
        topics.extend(crate::constants::topics::network_registry());
        topics.extend(crate::constants::topics::ticket_price_oracle());
        if self.cfg.fetch_token_transactions {
            // TODO: Still needed?
            // Actively query for logs to prevent polling done by Ethers.js
            // that don't retry on failed attempts and thus makes the indexer
            // handle errors produced by internal Ethers.js provider calls
            topics.extend(crate::constants::topics::token());
        }

        let log_filter = LogFilter {
            address: db_processor.contract_addresses(),
            topics: topics.into_iter().map(Hash::from).collect(),
        };

        info!("Building indexer background process");
        let (tx, rx) = futures::channel::oneshot::channel::<()>();

        let (tx_proc, rx_proc) = futures::channel::mpsc::unbounded::<Log>();

        let finalization = self.cfg.finalization;
        spawn(async move {
            let mut tx = Some(tx);

            let mut block_stream = rpc
                .try_stream_logs(latest_block_in_db, log_filter)
                .expect("block stream should be constructible");

            let mut unconfirmed_events = VecDeque::<Vec<Log>>::new();
            let mut chain_head = 0;

            while let Some(block_with_logs) = block_stream.next().await {
                debug!("Processed block number: {}", block_with_logs.block_id);

                if !block_with_logs.logs.is_empty() {
                    // Assuming sorted and properly organized blocks,
                    // the following lines are just a sanity safety mechanism
                    let mut logs = block_with_logs.logs;
                    logs.sort_by(log_comparator);
                    unconfirmed_events.push_back(logs);
                }

                let current_block = block_with_logs.block_id;

                match rpc.block_number().await {
                    Ok(current_chain_block_number) => {
                        chain_head = current_chain_block_number;
                    }
                    Err(error) => {
                        error!("failed to fetch block number from RPC: {error}");
                        chain_head = chain_head.max(current_block);
                    }
                }

                if tx.is_some() {
                    let indexing_scope = chain_head - latest_block_in_db;
                    info!(
                        "Sync progress {:.2}% @ block {}",
                        (1f64 - ((chain_head - current_block) as f64 / (indexing_scope as f64))) * 100f64,
                        current_block
                    );

                    if current_block + finalization >= chain_head {
                        info!("Indexer sync successfully completed");
                        let _ = tx.take().expect("tx should be present").send(());
                    }
                }

                while let Some(logs) = unconfirmed_events.get(0) {
                    if let Some(log) = logs.get(0) {
                        if log.block_number + finalization <= current_block {
                            if let Err(error) = db
                                .write()
                                .await
                                .update_latest_block_number(log.block_number as u32)
                                .await
                            {
                                error!("failed to write the latest block number into the database: {error}");
                            }

                            #[cfg(all(feature = "prometheus", not(test)))]
                            {
                                METRIC_INDEXER_CURRENT_BLOCK.set(log.block_number as f64);
                            }

                            let bn = log.block_number;
                            if let Some(logs) = unconfirmed_events.pop_front() {
                                debug!("processing logs from block #{}: {:?}", bn, logs);

                                for log in logs.into_iter() {
                                    if let Err(error) = tx_proc.unbounded_send(log) {
                                        error!("failed to send and process logs: {error}")
                                    }
                                }
                            }

                            continue;
                        }
                    }

                    break;
                }
            }
        });

        spawn(async move {
            let rx = rx_proc;

            pin_mut!(rx);
            while let Some(log) = rx.next().await {
                let snapshot = Snapshot::new(
                    U256::from(log.block_number),
                    U256::from(log.tx_index), // TODO: unused, kept for ABI compatibility of DB
                    log.log_index,
                );

                let tx_hash = log.tx_hash;

                match db_processor
                    .on_event(log.address, log.block_number as u32, log.into(), snapshot)
                    .await
                {
                    Ok(Some(event_type)) => {
                        // Pair the event type with the TX hash here
                        let significant_event = SignificantChainEvent { tx_hash, event_type };

                        if let Err(e) = tx_significant_events.unbounded_send(significant_event) {
                            error!("failed to pass a significant chain event further: {}", e);
                        }
                    }
                    Ok(None) => {}
                    Err(_) => {
                        error!("failed to process logs");
                    }
                };
            }
        });

        rx.await
            .map_err(|_| crate::errors::CoreEthereumIndexerError::ProcessError("Error during indexing start".into()))
    }
}

#[cfg(test)]
pub mod tests {
    use std::pin::Pin;

    use async_trait::async_trait;
    use bindings::hopr_announcements::AddressAnnouncementFilter;
    use core_crypto::keypairs::{Keypair, OffchainKeypair};
    use core_ethereum_db::db::CoreEthereumDb;
    use core_ethereum_rpc::BlockWithLogs;
    use core_ethereum_types::chain_events::ChainEventType;
    use ethers::{
        abi::{encode, Token},
        contract::EthEvent,
    };
    use futures::{join, Stream};
    use mockall::mock;
    use multiaddr::Multiaddr;
    use utils_db::{db::DB, CurrentDbShim};
    use utils_types::traits::PeerIdLike;
    use utils_types::{primitives::Address, traits::BinarySerializable};

    use crate::traits::MockChainLogHandler;

    use super::*;

    async fn create_stub_db() -> Arc<RwLock<CoreEthereumDb<CurrentDbShim>>> {
        Arc::new(RwLock::new(CoreEthereumDb::new(
            DB::new(CurrentDbShim::new_in_memory().await),
            Address::random(),
        )))
    }

    fn build_announcement_logs(address: Address, size: usize, block_number: u64, log_index: U256) -> Vec<Log> {
        let mut logs: Vec<Log> = vec![];

        for i in 0..size {
            let test_multiaddr: Multiaddr = format!("/ip4/1.2.3.4/tcp/{}", 1000 + i).parse().unwrap();
            logs.push(Log {
                address,
                topics: vec![AddressAnnouncementFilter::signature().into()],
                data: encode(&[
                    Token::Address(ethers::abi::Address::from_slice(&address.to_bytes())),
                    Token::String(test_multiaddr.to_string()),
                ])
                .into(),
                tx_hash: Default::default(),
                tx_index: 0,
                block_number,
                log_index,
            });
        }

        logs
    }

    mock! {
        HoprIndexerOps {}     // Name of the mock struct, less the "Mock" prefix

        #[async_trait]
        impl HoprIndexerRpcOperations for HoprIndexerOps {
            async fn block_number(&self) -> core_ethereum_rpc::errors::Result<u64>;

            fn try_stream_logs<'a>(
                &'a self,
                start_block_number: u64,
                filter: LogFilter,
            ) -> core_ethereum_rpc::errors::Result<Pin<Box<dyn Stream<Item = BlockWithLogs> + Send + 'a>>>;
        }
    }

    #[async_std::test]
    async fn test_indexer_should_check_the_db_for_last_processed_block_and_supply_none_if_none_is_found() {
        let mut handlers = MockChainLogHandler::new();
        let mut rpc = MockHoprIndexerOps::new();
        let db = create_stub_db().await;

        handlers.expect_contract_addresses().return_const(vec![]);

        let head_block = 1000;
        rpc.expect_block_number().return_once(move || Ok(head_block));

        let (tx, rx) = futures::channel::mpsc::unbounded::<BlockWithLogs>();
        rpc.expect_try_stream_logs()
            .withf(move |x: &u64, _y: &core_ethereum_rpc::LogFilter| *x == 0)
            .return_once(move |_, _| Ok(Box::pin(rx)));

        let mut indexer = Indexer::new(
            rpc,
            handlers,
            db.clone(),
            IndexerConfig::default(),
            futures::channel::mpsc::unbounded().0,
        );
        let (indexing, _) = join!(indexer.start(), async move {
            async_std::task::sleep(std::time::Duration::from_millis(200)).await;
            tx.close_channel()
        });
        assert!(indexing.is_err()) // terminated by the close channel
    }

    #[async_std::test]
    async fn test_indexer_should_check_the_db_for_last_processed_block_and_supply_it_when_found() {
        let mut handlers = MockChainLogHandler::new();
        let mut rpc = MockHoprIndexerOps::new();
        let db = create_stub_db().await;

        handlers.expect_contract_addresses().return_const(vec![]);

        let head_block = 1000;
        let latest_block = 15u64;
        assert!(db
            .write()
            .await
            .update_latest_block_number(latest_block as u32)
            .await
            .is_ok());
        rpc.expect_block_number().return_once(move || Ok(head_block));

        let (tx, rx) = futures::channel::mpsc::unbounded::<BlockWithLogs>();
        rpc.expect_try_stream_logs()
            .withf(move |x: &u64, _y: &core_ethereum_rpc::LogFilter| *x == latest_block)
            .return_once(move |_, _| Ok(Box::pin(rx)));

        let mut indexer = Indexer::new(
            rpc,
            handlers,
            db.clone(),
            IndexerConfig::default(),
            futures::channel::mpsc::unbounded().0,
        );
        let (indexing, _) = join!(indexer.start(), async move {
            async_std::task::sleep(std::time::Duration::from_millis(200)).await;
            tx.close_channel()
        });
        assert!(indexing.is_err()) // terminated by the close channel
    }

    #[async_std::test]
    async fn test_indexer_should_not_pass_blocks_unless_finalized() {
        let mut handlers = MockChainLogHandler::new();
        let mut rpc = MockHoprIndexerOps::new();
        let db = create_stub_db().await;

        handlers.expect_contract_addresses().return_const(vec![]);

        let head_block = 1000;
        rpc.expect_block_number().return_once(move || Ok(head_block));

        let (mut tx, rx) = futures::channel::mpsc::unbounded::<BlockWithLogs>();
        rpc.expect_try_stream_logs()
            .times(1)
            .withf(move |x: &u64, _y: &core_ethereum_rpc::LogFilter| *x == 0)
            .return_once(move |_, _| Ok(Box::pin(rx)));

        let expected = BlockWithLogs {
            block_id: head_block - 1,
            logs: vec![],
        };

        handlers.expect_on_event().times(0);

        assert!(tx.start_send(expected.clone()).is_ok());

        let mut indexer = Indexer::new(
            rpc,
            handlers,
            db.clone(),
            IndexerConfig::default(),
            futures::channel::mpsc::unbounded().0,
        );
        let _ = join!(indexer.start(), async move {
            async_std::task::sleep(std::time::Duration::from_millis(200)).await;
            tx.close_channel()
        });
    }

    #[async_std::test]
    async fn test_indexer_should_pass_blocks_that_are_finalized() {
        let mut handlers = MockChainLogHandler::new();
        let mut rpc = MockHoprIndexerOps::new();
        let db = create_stub_db().await;

        let cfg = IndexerConfig::default();

        handlers.expect_contract_addresses().return_const(vec![]);

        let (mut tx, rx) = futures::channel::mpsc::unbounded::<BlockWithLogs>();
        rpc.expect_try_stream_logs()
            .times(1)
            .withf(move |x: &u64, _y: &core_ethereum_rpc::LogFilter| *x == 0)
            .return_once(move |_, _| Ok(Box::pin(rx)));

        let head_block = 1000;
        rpc.expect_block_number().returning(move || Ok(head_block));
        rpc.expect_block_number().returning(move || Ok(head_block));

        let finalized_block = BlockWithLogs {
            block_id: head_block - cfg.finalization - 1,
            logs: build_announcement_logs(
                Address::random(),
                4,
                head_block - cfg.finalization - 1,
                U256::from(23u8),
            ),
        };
        let head_allowing_finalization = BlockWithLogs {
            block_id: head_block,
            logs: vec![],
        };

        handlers
            .expect_on_event()
            .times(finalized_block.logs.len())
            .returning(|_, _, _, _| Ok(None));

        assert!(tx.start_send(finalized_block.clone()).is_ok());
        assert!(tx.start_send(head_allowing_finalization.clone()).is_ok());

        let mut indexer = Indexer::new(rpc, handlers, db.clone(), cfg, futures::channel::mpsc::unbounded().0);
        let _ = join!(indexer.start(), async move {
            async_std::task::sleep(std::time::Duration::from_millis(200)).await;
            tx.close_channel()
        });
    }

    fn random_announcement_chain_event() -> ChainEventType {
        ChainEventType::Announcement {
            peer: OffchainKeypair::random().public().to_peerid(),
            address: Address::random(),
            multiaddresses: vec![Multiaddr::empty()],
        }
    }

    #[async_std::test]
    async fn test_indexer_should_yield_back_once_the_past_events_are_indexed() {
        let mut handlers = MockChainLogHandler::new();
        let mut rpc = MockHoprIndexerOps::new();
        let db = create_stub_db().await;

        let cfg = IndexerConfig::default();

        let expected_finalized_event_count = 4;

        handlers.expect_contract_addresses().return_const(vec![]);

        let (mut tx, rx) = futures::channel::mpsc::unbounded::<BlockWithLogs>();
        rpc.expect_try_stream_logs()
            .times(1)
            .withf(move |x: &u64, _y: &core_ethereum_rpc::LogFilter| *x == 0)
            .return_once(move |_, _| Ok(Box::pin(rx)));

        let head_block = 1000;
        for i in 0..2 {
            let current_block = head_block + i;
            rpc.expect_block_number().returning(move || Ok(current_block));
        }

        let finalized_block = BlockWithLogs {
            block_id: head_block - cfg.finalization - 1,
            logs: build_announcement_logs(
                Address::random(),
                expected_finalized_event_count,
                head_block - cfg.finalization - 1,
                U256::from(23u8),
            ),
        };

        let head_allowing_finalization = BlockWithLogs {
            block_id: head_block,
            logs: vec![],
        };

        handlers
            .expect_on_event()
            .times(expected_finalized_event_count)
            .returning(|_, _, _, _| Ok(Some(random_announcement_chain_event())));

        assert!(tx.start_send(finalized_block.clone()).is_ok());
        assert!(tx.start_send(head_allowing_finalization.clone()).is_ok());

        let (tx_events, rx_events) = futures::channel::mpsc::unbounded();
        let mut indexer = Indexer::new(rpc, handlers, db.clone(), cfg, tx_events.into());
        assert!(indexer.start().await.is_ok());

        tx.close_channel();

        let received = async_std::future::timeout(
            std::time::Duration::from_millis(500),
            rx_events.take(expected_finalized_event_count).collect::<Vec<_>>(),
        )
        .await;

        assert!(received.is_ok());
        assert_eq!(received.unwrap().len(), expected_finalized_event_count)
    }
}
