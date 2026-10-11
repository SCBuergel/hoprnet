use std::{
    collections::{HashMap, VecDeque},
    sync::{
        Arc, Mutex,
        atomic::{AtomicU64, Ordering},
    },
    time::{Duration, Instant},
};

use dashmap::DashMap;
use hopr_api::types::crypto::types::OffchainPublicKey;

/// Minimal atomic counters for per-peer protocol conformance tracking.
///
/// Tracks the number of messages sent and acknowledgments received for a
/// single peer. All operations are lock-free using relaxed atomic ordering.
#[derive(Debug, Default)]
pub struct PeerProtocolCounters {
    messages_sent: AtomicU64,
    acks_received: AtomicU64,
    ack_delay: Mutex<AckDelay>,
}

impl PeerProtocolCounters {
    /// Record that a message was sent to this peer.
    #[inline]
    pub fn record_message_sent(&self) {
        self.messages_sent.fetch_add(1, Ordering::Relaxed);
        if let Ok(mut delay) = self.ack_delay.lock() {
            delay.sent(Instant::now());
        }
    }

    /// How much longer this peer's acknowledgements currently take than they did at their fastest
    /// lately; `None` without recent evidence. See [`AckDelay`].
    pub fn queueing_delay(&self) -> Option<Duration> {
        self.ack_delay.lock().ok().and_then(|delay| delay.queueing_delay(Instant::now()))
    }

    /// The smoothed acknowledgement delay and its recent floor, for diagnostics.
    pub fn ack_delay_estimate(&self) -> Option<(Duration, Duration)> {
        self.ack_delay.lock().ok().and_then(|delay| delay.estimate(Instant::now()))
    }

    /// Record that an acknowledgment was received from this peer.
    #[inline]
    pub fn record_ack_received(&self) {
        self.record_acks_received(1);
    }

    /// Record that `count` acknowledgments were received from this peer in a batch.
    #[inline]
    pub fn record_acks_received(&self, count: u64) {
        self.acks_received.fetch_add(count, Ordering::Relaxed);
        if let Ok(mut delay) = self.ack_delay.lock() {
            delay.acked(count, Instant::now());
        }
    }

    /// Swap the sent-message counter to 0, returning the accumulated value.
    #[inline]
    pub fn take_sent(&self) -> u64 {
        self.messages_sent.swap(0, Ordering::Relaxed)
    }

    /// Swap the acknowledgment counter to 0, returning the accumulated value.
    #[inline]
    pub fn take_acks(&self) -> u64 {
        self.acks_received.swap(0, Ordering::Relaxed)
    }

    /// Swap both counters to 0, returning accumulated values.
    ///
    /// Each counter is swapped independently via its own atomic operation;
    /// there is no single atomic snapshot of the pair.
    pub fn take(&self) -> (u64, u64) {
        (self.take_sent(), self.take_acks())
    }
}

/// Send times kept per peer at most. Past it the oldest are dropped, as if never acknowledged.
const ACK_DELAY_MAX_PENDING: usize = 50_000;

/// A packet unacknowledged for this long is taken as lost: its acknowledgement can no longer be
/// told from that of a later packet.
const ACK_DELAY_MAX_WAIT: Duration = Duration::from_secs(10);

/// Weight of one acknowledged packet in the smoothed delay. About 50 packets make up the average:
/// a relay batches its acknowledgements every 200 ms, so a single packet's delay carries up to
/// 200 ms of batching noise, and the average needs that many to settle within a few milliseconds.
const ACK_DELAY_ALPHA: f64 = 0.02;

/// Width of one slice of the floor's window, and the number of slices.
///
/// The floor is the lowest the smoothed delay has been over five minutes: long enough that a relay
/// kept congested by steady load does not become its own baseline (and so stop being avoided),
/// short enough to follow a route that really did get slower.
const ACK_DELAY_FLOOR_SLICE: Duration = Duration::from_secs(30);
const ACK_DELAY_FLOOR_SLICES: usize = 10;

/// Without an acknowledgement for this long the estimate says nothing. A relay the planner has
/// steered away from still gets probes, so this ages out only a peer that went quiet altogether.
const ACK_DELAY_STALE: Duration = Duration::from_secs(10);

/// Delay of the acknowledgements a peer returns, and how far it currently sits above its floor.
///
/// The packets a node sends to a peer come back acknowledged by that peer, in batches. The batch
/// travels back on the peer's own outgoing link, behind whatever else the peer is sending — so when
/// that link is saturated, the acknowledgements queue with everything else. Their delay above its
/// usual floor is a measure of the queue at the peer, available at data rates for every peer the
/// node sends to. Probes measure the same thing but once every few seconds, and their average
/// barely moves while a saturated relay's queue swings between empty and full.
///
/// Acknowledgements carry no reference the sender can match cheaply, so a batch of `n` is taken to
/// acknowledge the `n` oldest unacknowledged packets. Acknowledgements return roughly in sending
/// order, and the error only shifts which packet's delay is counted, not the average.
#[derive(Debug, Default)]
struct AckDelay {
    sent_at: VecDeque<Instant>,
    smoothed_ms: Option<f64>,
    last_ack: Option<Instant>,
    /// `(slice start, lowest smoothed delay in it)`, oldest first.
    floor: VecDeque<(Instant, f64)>,
}

impl AckDelay {
    fn sent(&mut self, now: Instant) {
        if self.sent_at.len() >= ACK_DELAY_MAX_PENDING {
            self.sent_at.pop_front();
        }
        self.sent_at.push_back(now);
    }

    fn acked(&mut self, count: u64, now: Instant) {
        while self
            .sent_at
            .front()
            .is_some_and(|sent| now.saturating_duration_since(*sent) > ACK_DELAY_MAX_WAIT)
        {
            self.sent_at.pop_front();
        }

        let (mut n, mut total_ms) = (0u64, 0.0);
        for _ in 0..count {
            let Some(sent) = self.sent_at.pop_front() else { break };
            total_ms += now.saturating_duration_since(sent).as_secs_f64() * 1000.0;
            n += 1;
        }
        if n == 0 {
            return;
        }

        let mean_ms = total_ms / n as f64;
        let smoothed = match self.smoothed_ms {
            Some(prev) => {
                let alpha = 1.0 - (1.0 - ACK_DELAY_ALPHA).powi(n.min(i32::MAX as u64) as i32);
                prev + alpha * (mean_ms - prev)
            }
            None => mean_ms,
        };
        self.smoothed_ms = Some(smoothed);
        self.last_ack = Some(now);

        match self.floor.back_mut() {
            Some((start, min)) if now.saturating_duration_since(*start) < ACK_DELAY_FLOOR_SLICE => {
                *min = min.min(smoothed);
            }
            _ => self.floor.push_back((now, smoothed)),
        }
        while self.floor.len() > ACK_DELAY_FLOOR_SLICES {
            self.floor.pop_front();
        }
    }

    fn estimate(&self, now: Instant) -> Option<(Duration, Duration)> {
        let last = self.last_ack?;
        if now.saturating_duration_since(last) > ACK_DELAY_STALE {
            return None;
        }
        let smoothed = self.smoothed_ms?;
        let floor = self.floor.iter().map(|(_, min)| *min).fold(f64::INFINITY, f64::min);
        let ms = |v: f64| Duration::from_secs_f64(v.max(0.0) / 1000.0);
        Some((ms(smoothed), ms(floor.min(smoothed))))
    }

    fn queueing_delay(&self, now: Instant) -> Option<Duration> {
        self.estimate(now).map(|(smoothed, floor)| smoothed.saturating_sub(floor))
    }
}

/// Thread-safe registry of per-peer protocol conformance counters.
///
/// Keyed by [`OffchainPublicKey`] — no PeerId conversion needed since the
/// protocol pipeline already operates on offchain keys.
#[derive(Debug, Default, Clone)]
pub struct PeerProtocolCounterRegistry {
    inner: Arc<DashMap<OffchainPublicKey, Arc<PeerProtocolCounters>>>,
}

impl PeerProtocolCounterRegistry {
    /// Get or create counters for the given peer.
    pub fn get_or_create(&self, peer: &OffchainPublicKey) -> Arc<PeerProtocolCounters> {
        self.inner
            .entry(*peer)
            .or_insert_with(|| Arc::new(PeerProtocolCounters::default()))
            .value()
            .clone()
    }

    /// The current queueing delay at `peer`, as its acknowledgements show it; `None` without recent
    /// evidence or for a peer never sent to.
    pub fn queueing_delay(&self, peer: &OffchainPublicKey) -> Option<Duration> {
        self.inner.get(peer).and_then(|entry| entry.value().queueing_delay())
    }

    /// Sets `peer`'s acknowledgement delay outright: `smoothed` now, against a floor of `floor`.
    #[cfg(test)]
    pub(crate) fn set_ack_delay_for_test(&self, peer: &OffchainPublicKey, smoothed: Duration, floor: Duration) {
        let counters = self.get_or_create(peer);
        let mut delay = counters.ack_delay.lock().expect("lock is not poisoned");
        let now = Instant::now();
        delay.smoothed_ms = Some(smoothed.as_secs_f64() * 1000.0);
        delay.last_ack = Some(now);
        delay.floor = [(now, floor.as_secs_f64() * 1000.0)].into_iter().collect();
    }

    /// Every peer with a current acknowledgement-delay estimate: `(peer, smoothed, floor)`.
    pub fn ack_delays(&self) -> Vec<(OffchainPublicKey, Duration, Duration)> {
        self.inner
            .iter()
            .filter_map(|entry| {
                entry
                    .value()
                    .ack_delay_estimate()
                    .map(|(smoothed, floor)| (*entry.key(), smoothed, floor))
            })
            .collect()
    }

    /// Drains the counters for one conformance report, giving every packet in it `ack_grace` to be
    /// acknowledged.
    ///
    /// [`Self::drain`] reads both counters at the same instant, so a packet still in flight then is
    /// reported as sent but never acknowledged. In steady traffic that is a sliver of the window. A
    /// burst that starts just before the flush, though, is in flight almost entirely: the report
    /// reads as an acknowledgment rate near zero, the edge falls below the path planner's
    /// `min_ack_rate`, and it stays out of path selection until the next report (hoprnet#8484).
    ///
    /// So the sent counts are taken first, and the acknowledgments only `ack_grace` later. Packets
    /// sent during the grace stay counted for the next report. Their acknowledgments that arrive
    /// within it are credited to this one, a report early. The decayed totals the graph keeps absorb
    /// that: a report later, a burst's acknowledgments still weigh 0.9 against its packets.
    ///
    /// Returns `(peer, msgs_sent, acks_received)` for non-zero entries, like [`Self::drain`].
    pub async fn drain_settled(&self, ack_grace: Duration) -> Vec<(OffchainPublicKey, u64, u64)> {
        let sent = self
            .inner
            .iter()
            .map(|entry| (*entry.key(), entry.value().take_sent()))
            .filter(|(_, sent)| *sent > 0)
            .collect::<HashMap<_, _>>();

        if !ack_grace.is_zero() {
            futures_timer::Delay::new(ack_grace).await;
        }

        self.inner
            .iter()
            .filter_map(|entry| {
                let sent = sent.get(entry.key()).copied().unwrap_or_default();
                let received = entry.value().take_acks();
                (sent > 0 || received > 0).then_some((*entry.key(), sent, received))
            })
            .collect()
    }

    /// Swap all counters to 0, returning `(peer, msgs_sent, acks_received)` for non-zero entries.
    pub fn drain(&self) -> Vec<(OffchainPublicKey, u64, u64)> {
        self.inner
            .iter()
            .filter_map(|entry| {
                let (sent, received) = entry.value().take();
                if sent > 0 || received > 0 {
                    Some((*entry.key(), sent, received))
                } else {
                    None
                }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use anyhow::Context;
    use hopr_api::types::crypto::prelude::{Keypair, OffchainKeypair};

    use super::*;

    #[test]
    fn counters_should_start_at_zero() {
        let counters = PeerProtocolCounters::default();
        let (sent, received) = counters.take();

        assert_eq!(sent, 0);
        assert_eq!(received, 0);
    }

    #[test]
    fn counters_should_record_messages_sent() {
        let counters = PeerProtocolCounters::default();

        counters.record_message_sent();
        counters.record_message_sent();
        counters.record_message_sent();

        let (sent, received) = counters.take();
        assert_eq!(sent, 3);
        assert_eq!(received, 0);
    }

    #[test]
    fn counters_should_record_acks_received() {
        let counters = PeerProtocolCounters::default();

        counters.record_ack_received();
        counters.record_ack_received();

        let (sent, received) = counters.take();
        assert_eq!(sent, 0);
        assert_eq!(received, 2);
    }

    #[test]
    fn take_should_reset_counters_to_zero() {
        let counters = PeerProtocolCounters::default();

        counters.record_message_sent();
        counters.record_ack_received();

        let (sent1, received1) = counters.take();
        assert_eq!(sent1, 1);
        assert_eq!(received1, 1);

        let (sent2, received2) = counters.take();
        assert_eq!(sent2, 0);
        assert_eq!(received2, 0);
    }

    #[test]
    fn registry_should_create_and_retrieve_counters() -> anyhow::Result<()> {
        let registry = PeerProtocolCounterRegistry::default();
        let peer = *OffchainKeypair::random().public();

        let counters = registry.get_or_create(&peer);
        counters.record_message_sent();

        let same_counters = registry.get_or_create(&peer);
        same_counters.record_message_sent();

        let (sent, _) = counters.take();
        assert_eq!(sent, 2, "both calls should share the same counter instance");
        Ok(())
    }

    #[test]
    fn drain_should_return_only_nonzero_entries() -> anyhow::Result<()> {
        let registry = PeerProtocolCounterRegistry::default();
        let peer_a = *OffchainKeypair::random().public();
        let peer_b = *OffchainKeypair::random().public();
        let peer_c = *OffchainKeypair::random().public();

        registry.get_or_create(&peer_a).record_message_sent();
        registry.get_or_create(&peer_b); // no activity
        registry.get_or_create(&peer_c).record_ack_received();

        let drained = registry.drain();
        assert_eq!(drained.len(), 2, "only peers with non-zero counters should be drained");

        let a_entry = drained
            .iter()
            .find(|(p, ..)| *p == peer_a)
            .context("peer_a should be in drain results")?;
        assert_eq!(a_entry.1, 1);
        assert_eq!(a_entry.2, 0);

        let c_entry = drained
            .iter()
            .find(|(p, ..)| *p == peer_c)
            .context("peer_c should be in drain results")?;
        assert_eq!(c_entry.1, 0);
        assert_eq!(c_entry.2, 1);

        Ok(())
    }

    #[test]
    fn drain_should_reset_counters() {
        let registry = PeerProtocolCounterRegistry::default();
        let peer = *OffchainKeypair::random().public();

        registry.get_or_create(&peer).record_message_sent();

        let first_drain = registry.drain();
        assert_eq!(first_drain.len(), 1);

        let second_drain = registry.drain();
        assert!(second_drain.is_empty(), "counters should be zero after drain");
    }

    #[tokio::test]
    async fn drain_settled_should_credit_acks_that_arrive_within_the_grace() -> anyhow::Result<()> {
        let registry = PeerProtocolCounterRegistry::default();
        let peer = *OffchainKeypair::random().public();
        let counters = registry.get_or_create(&peer);

        // A burst whose acknowledgments are all still in flight when the flush starts.
        for _ in 0..10 {
            counters.record_message_sent();
        }

        // Drive the drain into its grace deterministically rather than racing a timer: the first
        // poll takes the `sent` snapshot and starts the grace, so the acks recorded next are
        // guaranteed to land inside the window before `take_acks` reads them. A starved executor
        // could otherwise let the grace timer and an ack timer fire together, with the drain polled
        // first and reading zero acks.
        let mut drained = std::pin::pin!(registry.drain_settled(Duration::from_millis(50)));
        anyhow::ensure!(
            futures::poll!(drained.as_mut()).is_pending(),
            "the drain must be waiting out its grace before the acks are recorded"
        );
        counters.record_acks_received(10);
        let drained = drained.await;

        let (_, sent, received) = drained.first().context("the peer must be reported")?;
        assert_eq!(
            (*sent, *received),
            (10, 10),
            "acknowledgments arriving within the grace must be credited"
        );
        Ok(())
    }

    #[tokio::test]
    async fn drain_settled_should_carry_packets_sent_during_the_grace_into_the_next_report() -> anyhow::Result<()> {
        let registry = PeerProtocolCounterRegistry::default();
        let peer = *OffchainKeypair::random().public();
        let counters = registry.get_or_create(&peer);

        counters.record_message_sent();

        // First poll takes the `sent` snapshot (one packet) and starts the grace; the packets
        // recorded next therefore land after the snapshot, inside the grace, with no dependence on
        // timer order.
        let mut first = std::pin::pin!(registry.drain_settled(Duration::from_millis(50)));
        anyhow::ensure!(
            futures::poll!(first.as_mut()).is_pending(),
            "the drain must be waiting out its grace before the in-grace packets are recorded"
        );
        counters.record_message_sent();
        counters.record_message_sent();
        let first = first.await;
        let (_, sent, _) = first.first().context("the peer must be reported")?;
        assert_eq!(*sent, 1, "only the packet sent before the grace belongs to this report");

        let second = registry.drain_settled(Duration::ZERO).await;
        let (_, sent, _) = second.first().context("the carried packets must be reported")?;
        assert_eq!(*sent, 2, "packets sent during the grace must be reported next time");
        Ok(())
    }

    #[tokio::test]
    async fn drain_settled_should_report_only_active_peers() {
        let registry = PeerProtocolCounterRegistry::default();
        registry.get_or_create(OffchainKeypair::random().public());

        assert!(
            registry.drain_settled(Duration::ZERO).await.is_empty(),
            "a peer with nothing sent or received must not be reported"
        );
    }

    /// Sends `per_batch` packets every 10 ms and acknowledges them in 200 ms batches, each packet
    /// `delay` after it was sent, from `start` for `secs` seconds. Returns the time it ended at.
    fn feed(est: &mut AckDelay, start: std::time::Instant, secs: u64, delay: Duration) -> std::time::Instant {
        let mut now = start;
        let end = start + Duration::from_secs(secs);
        let mut pending: std::collections::VecDeque<std::time::Instant> = Default::default();
        let mut next_flush = start + Duration::from_millis(200);
        while now < end {
            est.sent(now);
            pending.push_back(now);
            now += Duration::from_millis(10);
            if now >= next_flush {
                // The batch carries every packet that has reached the peer by the flush.
                let n = pending.iter().take_while(|sent| **sent + delay <= now).count();
                pending.drain(..n);
                if n > 0 {
                    est.acked(n as u64, now);
                }
                next_flush += Duration::from_millis(200);
            }
        }
        now
    }

    #[test]
    fn ack_delay_should_report_no_queue_on_a_steady_link() -> anyhow::Result<()> {
        let mut est = AckDelay::default();
        let now = feed(&mut est, std::time::Instant::now(), 30, Duration::from_millis(40));
        let q = est.queueing_delay(now).context("a busy peer must have an estimate")?;
        assert!(q < Duration::from_millis(10), "a constant delay is the floor, not a queue: {q:?}");
        Ok(())
    }

    #[test]
    fn ack_delay_should_report_a_queue_that_builds_up() -> anyhow::Result<()> {
        let mut est = AckDelay::default();
        let start = std::time::Instant::now();
        let now = feed(&mut est, start, 30, Duration::from_millis(40));
        let now = feed(&mut est, now, 10, Duration::from_millis(190));
        let q = est.queueing_delay(now).context("a busy peer must have an estimate")?;
        assert!(
            q > Duration::from_millis(120) && q < Duration::from_millis(170),
            "150 ms of extra delay must read as about 150 ms of queue: {q:?}"
        );
        Ok(())
    }

    #[test]
    fn ack_delay_should_say_nothing_once_the_peer_went_quiet() {
        let mut est = AckDelay::default();
        let now = feed(&mut est, std::time::Instant::now(), 5, Duration::from_millis(40));
        assert!(est.queueing_delay(now).is_some());
        assert!(est.queueing_delay(now + ACK_DELAY_STALE + Duration::from_secs(1)).is_none());
    }

    #[test]
    fn ack_delay_should_forget_packets_that_were_never_acknowledged() -> anyhow::Result<()> {
        let mut est = AckDelay::default();
        let start = std::time::Instant::now();
        // 100 packets lost: sent, never acknowledged.
        for i in 0..100 {
            est.sent(start + Duration::from_millis(i));
        }
        let later = start + ACK_DELAY_MAX_WAIT + Duration::from_secs(1);
        let now = feed(&mut est, later, 30, Duration::from_millis(40));
        let (smoothed, _) = est.estimate(now).context("a busy peer must have an estimate")?;
        assert!(
            smoothed < Duration::from_millis(200),
            "lost packets must not inflate the delay of the ones acknowledged later: {smoothed:?}"
        );
        Ok(())
    }

    #[test]
    fn registry_should_expose_the_queueing_delay_per_peer() {
        let registry = PeerProtocolCounterRegistry::default();
        let peer = *OffchainKeypair::random().public();
        assert!(registry.queueing_delay(&peer).is_none(), "an unknown peer has no estimate");
        let counters = registry.get_or_create(&peer);
        counters.record_message_sent();
        counters.record_acks_received(1);
        assert!(registry.queueing_delay(&peer).is_some());
        assert_eq!(registry.ack_delays().len(), 1);
    }
}
