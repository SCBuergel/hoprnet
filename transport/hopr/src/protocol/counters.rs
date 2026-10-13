use std::{
    collections::{HashMap, VecDeque},
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
    time::{Duration, Instant},
};

use dashmap::DashMap;
use hopr_api::types::crypto::types::{HalfKey, HalfKeyChallenge, OffchainPublicKey};

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
    }

    /// Record that a message awaiting the acknowledgement that solves `challenge` was sent now.
    pub fn record_awaiting_ack(&self, challenge: HalfKeyChallenge) {
        if let Ok(mut delay) = self.ack_delay.lock() {
            delay.sent(challenge, Instant::now());
        }
    }

    /// Record that acknowledgements solving `challenges` arrived at `at`, in one batch.
    pub fn record_acked(&self, challenges: &[HalfKeyChallenge], at: Instant) {
        if let Ok(mut delay) = self.ack_delay.lock() {
            delay.acked(challenges, at);
        }
    }

    /// This peer's current queueing delay and loss, as its acknowledgements show them. See
    /// [`LinkCongestion`].
    pub fn congestion(&self) -> LinkCongestion {
        self.ack_delay
            .lock()
            .map(|delay| delay.congestion(Instant::now()))
            .unwrap_or_default()
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

/// What a peer's acknowledgements say about the link through it. `None` where they say too little.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct LinkCongestion {
    /// The acknowledgement delay above its recent floor: the queue at the peer.
    pub queueing: Option<Duration>,
    /// The smoothed acknowledgement delay and its floor, for diagnostics.
    pub delay: Option<(Duration, Duration)>,
    /// The share of packets never acknowledged.
    pub loss: Option<f64>,
}

/// Packets awaiting acknowledgement kept per peer at most; past it the oldest are taken as lost.
/// Bound on challenges awaiting an acknowledgement per peer: five seconds (`ACK_LOST_AFTER`) at a few
/// thousand packets per second, about 1.5 MB at the worst. The expiry sweep normally keeps it far lower.
const ACK_PENDING_MAX: usize = 16_384;

/// A packet unacknowledged for this long is taken as lost.
const ACK_LOST_AFTER: Duration = Duration::from_secs(5);

/// Time constant of the smoothed delay: two seconds of acknowledgements make up the average.
///
/// In time rather than in packets, so a relay the planner has steered away from -- sent only a
/// probe every few seconds -- is judged by its latest few samples, not by a backlog of old ones,
/// and can recover.
const ACK_DELAY_TAU: Duration = Duration::from_secs(2);

/// Batches within the last two time constants needed before the delay says anything. A peer
/// measured by sparse probes alone reads as unknown, not as whatever its last probe happened to
/// show -- a single sample carries up to a relay's whole acknowledgement batching interval.
const ACK_DELAY_MIN_BATCHES: f64 = 5.0;

/// Time constant of the loss share.
const ACK_LOSS_TAU: Duration = Duration::from_secs(10);

/// Packets, decayed as in [`ACK_LOSS_TAU`], needed before the loss share says anything.
const ACK_LOSS_MIN_PACKETS: f64 = 20.0;

/// Width of one slice of the floor's window, and the number of slices.
///
/// The floor is the lowest the smoothed delay has been over five minutes: long enough that a relay
/// kept congested by steady load does not become its own baseline (and so stop being avoided),
/// short enough to follow a route that really did get slower.
const ACK_DELAY_FLOOR_SLICE: Duration = Duration::from_secs(30);
const ACK_DELAY_FLOOR_SLICES: usize = 10;

/// Exponential weight of something that happened `age` ago, with time constant `tau`.
fn decay(age: Duration, tau: Duration) -> f64 {
    (-age.as_secs_f64() / tau.as_secs_f64()).exp()
}

/// Delay and loss of the acknowledgements a peer returns.
///
/// The packets a node sends to a peer come back acknowledged by that peer, in batches. A batch
/// travels back on the peer's own outgoing link, behind whatever else the peer is sending -- so
/// when that link is saturated, the acknowledgements queue with everything else, and when it
/// drops, some never come. Their delay above its usual floor measures the queue at the peer, their
/// absence its loss, at data rates for every peer the node sends to. Probes measure the same once
/// every few seconds.
///
/// Each acknowledgement is matched to its packet by the challenge it solves, so a lost packet
/// cannot shift the timing of the ones after it. Within a batch, the newest packet is the one
/// timed: the others also waited for the peer's batching interval, which says nothing about its
/// link.
#[derive(Debug, Default)]
struct AckDelay {
    pending: HashMap<HalfKeyChallenge, Instant>,
    /// Send order of `pending`, for expiry; may still hold challenges acknowledged since.
    order: VecDeque<(Instant, HalfKeyChallenge)>,
    /// Decayed sums of batch delays (ms) and of batch weights, as of `delay_at`.
    delay_sum: f64,
    delay_weight: f64,
    delay_at: Option<Instant>,
    /// Decayed counts of lost and of all packets, as of `loss_at`.
    lost: f64,
    outcomes: f64,
    loss_at: Option<Instant>,
    /// `(slice start, lowest smoothed delay in it)`, oldest first.
    floor: VecDeque<(Instant, f64)>,
}

impl AckDelay {
    fn sent(&mut self, challenge: HalfKeyChallenge, now: Instant) {
        self.expire(now);
        if self.pending.len() >= ACK_PENDING_MAX {
            while let Some((_, oldest)) = self.order.pop_front() {
                if self.pending.remove(&oldest).is_some() {
                    self.outcome(false, now);
                    break;
                }
            }
        }
        if self.pending.insert(challenge, now).is_none() {
            self.order.push_back((now, challenge));
        }
    }

    fn expire(&mut self, now: Instant) {
        while let Some((sent, challenge)) = self.order.front().copied() {
            if now.saturating_duration_since(sent) <= ACK_LOST_AFTER {
                break;
            }
            self.order.pop_front();
            if self.pending.remove(&challenge).is_some() {
                self.outcome(false, now);
            }
        }
    }

    fn outcome(&mut self, acknowledged: bool, now: Instant) {
        let w = self.loss_at.map_or(1.0, |at| decay(now.saturating_duration_since(at), ACK_LOSS_TAU));
        self.lost = self.lost * w + if acknowledged { 0.0 } else { 1.0 };
        self.outcomes = self.outcomes * w + 1.0;
        self.loss_at = Some(now);
    }

    fn acked(&mut self, challenges: &[HalfKeyChallenge], now: Instant) {
        let mut newest: Option<Instant> = None;
        for challenge in challenges {
            if let Some(sent) = self.pending.remove(challenge) {
                newest = Some(newest.map_or(sent, |n| n.max(sent)));
                self.outcome(true, now);
            }
        }
        self.expire(now);
        let Some(newest) = newest else { return };

        let sample_ms = now.saturating_duration_since(newest).as_secs_f64() * 1000.0;
        let w = self.delay_at.map_or(1.0, |at| decay(now.saturating_duration_since(at), ACK_DELAY_TAU));
        self.delay_sum = self.delay_sum * w + sample_ms;
        self.delay_weight = self.delay_weight * w + 1.0;
        self.delay_at = Some(now);

        if self.delay_weight < ACK_DELAY_MIN_BATCHES {
            return;
        }
        let smoothed = self.delay_sum / self.delay_weight;
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

    fn congestion(&self, now: Instant) -> LinkCongestion {
        let ms = |v: f64| Duration::from_secs_f64(v.max(0.0) / 1000.0);
        let delay = self.delay_at.and_then(|at| {
            let w = decay(now.saturating_duration_since(at), ACK_DELAY_TAU);
            if self.delay_weight * w < ACK_DELAY_MIN_BATCHES / 2.0 || self.delay_weight < ACK_DELAY_MIN_BATCHES {
                return None;
            }
            let smoothed = self.delay_sum / self.delay_weight;
            let floor = self.floor.iter().map(|(_, min)| *min).fold(f64::INFINITY, f64::min);
            floor.is_finite().then(|| (ms(smoothed), ms(floor.min(smoothed))))
        });
        // Packets still awaiting an acknowledgement past the deadline count as lost even before an
        // expiry sweep ran: a peer that stopped answering altogether must not read as clean.
        let overdue = self
            .order
            .iter()
            .take_while(|(sent, _)| now.saturating_duration_since(*sent) > ACK_LOST_AFTER)
            .filter(|(_, challenge)| self.pending.contains_key(challenge))
            .count() as f64;
        let w = self.loss_at.map_or(1.0, |at| decay(now.saturating_duration_since(at), ACK_LOSS_TAU));
        let (lost, outcomes) = (self.lost * w + overdue, self.outcomes * w + overdue);
        LinkCongestion {
            queueing: delay.map(|(smoothed, floor)| smoothed.saturating_sub(floor)),
            delay,
            loss: (outcomes >= ACK_LOSS_MIN_PACKETS).then(|| lost / outcomes),
        }
    }
}

/// The challenge an acknowledgement solves, for timing it; `None` for a malformed one.
///
/// Read from the unverified acknowledgement: what it times is the path back from the peer, and a
/// peer forging acknowledgements could as well delay genuine ones.
pub fn ack_challenge(ack: &hopr_api::types::internal::protocol::Acknowledgement) -> Option<HalfKeyChallenge> {
    use hopr_api::types::primitive::prelude::BytesRepresentable;
    HalfKey::try_from(&ack.as_ref()[..HalfKey::SIZE]).ok()?.to_challenge().ok()
}

/// Thread-safe registry of per-peer protocol conformance counters.
///
/// Keyed by [`OffchainPublicKey`] — no PeerId conversion needed since the
/// protocol pipeline already operates on offchain keys.
#[derive(Debug, Default, Clone)]
pub struct PeerProtocolCounterRegistry {
    inner: Arc<DashMap<OffchainPublicKey, Arc<PeerProtocolCounters>>>,
    ack_timing: Arc<AtomicBool>,
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

    /// Times every acknowledgement against its packet from now on. Costs one scalar multiplication
    /// per acknowledgement, so only nodes that route by it (the session's entry) turn it on.
    pub fn enable_ack_timing(&self) {
        self.ack_timing.store(true, Ordering::Relaxed);
    }

    /// Whether acknowledgements are timed; see [`Self::enable_ack_timing`].
    pub fn ack_timing(&self) -> bool {
        self.ack_timing.load(Ordering::Relaxed)
    }

    /// The link through `peer` as its acknowledgements show it; all `None` for a peer never sent to.
    pub fn congestion(&self, peer: &OffchainPublicKey) -> LinkCongestion {
        self.inner
            .get(peer)
            .map(|entry| entry.value().congestion())
            .unwrap_or_default()
    }

    /// Sets `peer`'s acknowledgement delay outright: `smoothed` now, against a floor of `floor`.
    #[cfg(test)]
    pub(crate) fn set_ack_delay_for_test(&self, peer: &OffchainPublicKey, smoothed: Duration, floor: Duration) {
        let counters = self.get_or_create(peer);
        let mut delay = counters.ack_delay.lock().expect("lock is not poisoned");
        let now = Instant::now();
        delay.delay_sum = smoothed.as_secs_f64() * 1000.0 * 100.0;
        delay.delay_weight = 100.0;
        delay.delay_at = Some(now);
        delay.floor = [(now, floor.as_secs_f64() * 1000.0)].into_iter().collect();
    }

    /// Sets `peer`'s loss share outright.
    #[cfg(test)]
    pub(crate) fn set_loss_for_test(&self, peer: &OffchainPublicKey, loss: f64) {
        let counters = self.get_or_create(peer);
        let mut delay = counters.ack_delay.lock().expect("lock is not poisoned");
        delay.lost = loss * 1000.0;
        delay.outcomes = 1000.0;
        delay.loss_at = Some(Instant::now());
    }

    /// Every peer with something to report: `(peer, congestion)`.
    pub fn congestions(&self) -> Vec<(OffchainPublicKey, LinkCongestion)> {
        self.inner
            .iter()
            .map(|entry| (*entry.key(), entry.value().congestion()))
            .filter(|(_, c)| c.delay.is_some() || c.loss.is_some())
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
    use hopr_api::types::crypto::{
        crypto_traits::Randomizable,
        prelude::{Keypair, OffchainKeypair},
    };

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

    /// Sends a packet every 10 ms for `secs` seconds and acknowledges them in 200 ms batches, each
    /// packet `delay(elapsed)` after it was sent; `lost(i)` packets are never acknowledged. Returns
    /// the time it ended at.
    fn feed(
        est: &mut AckDelay,
        start: std::time::Instant,
        secs: u64,
        delay: impl Fn(Duration) -> Duration,
        lost: impl Fn(usize) -> bool,
    ) -> std::time::Instant {
        let mut now = start;
        let end = start + Duration::from_secs(secs);
        let mut pending: std::collections::VecDeque<(std::time::Instant, HalfKeyChallenge)> = Default::default();
        let mut next_flush = start + Duration::from_millis(200);
        let mut i = 0usize;
        while now < end {
            let challenge = HalfKey::random().to_challenge().expect("a random half key is a valid scalar");
            est.sent(challenge, now);
            if !lost(i) {
                pending.push_back((now, challenge));
            }
            i += 1;
            now += Duration::from_millis(10);
            if now >= next_flush {
                // The batch carries every packet that has reached the peer by the flush.
                let d = delay(now - start);
                let n = pending.iter().take_while(|(sent, _)| *sent + d <= now).count();
                let batch: Vec<_> = pending.drain(..n).map(|(_, c)| c).collect();
                est.acked(&batch, now);
                next_flush += Duration::from_millis(200);
            }
        }
        now
    }

    #[test]
    fn ack_delay_should_report_no_queue_on_a_steady_link() -> anyhow::Result<()> {
        let mut est = AckDelay::default();
        let now = feed(&mut est, std::time::Instant::now(), 20, |_| Duration::from_millis(40), |_| false);
        let c = est.congestion(now);
        let q = c.queueing.context("a busy peer must have an estimate")?;
        assert!(q < Duration::from_millis(10), "a constant delay is the floor, not a queue: {q:?}");
        assert_eq!(c.loss, Some(0.0));
        Ok(())
    }

    #[test]
    fn ack_delay_should_report_a_queue_that_builds_up() -> anyhow::Result<()> {
        let mut est = AckDelay::default();
        let now = feed(
            &mut est,
            std::time::Instant::now(),
            30,
            |elapsed| Duration::from_millis(if elapsed < Duration::from_secs(20) { 40 } else { 190 }),
            |_| false,
        );
        let q = est.congestion(now).queueing.context("a busy peer must have an estimate")?;
        assert!(
            q > Duration::from_millis(125) && q < Duration::from_millis(175),
            "150 ms of extra delay must read as about 150 ms of queue: {q:?}"
        );
        Ok(())
    }

    /// The flaw of matching acknowledgements in sending order: every lost packet shifted the timing of
    /// all the packets after it, and a lossy link read as seconds of queue.
    #[test]
    fn ack_delay_should_not_mistake_loss_for_a_queue() -> anyhow::Result<()> {
        let mut est = AckDelay::default();
        let now = feed(&mut est, std::time::Instant::now(), 20, |_| Duration::from_millis(40), |i| i % 10 < 3);
        let c = est.congestion(now);
        let q = c.queueing.context("a busy peer must have an estimate")?;
        assert!(q < Duration::from_millis(10), "loss must not read as a queue: {q:?}");
        let loss = c.loss.context("enough packets for a loss share")?;
        assert!((0.25..0.35).contains(&loss), "30 % of the packets were lost: {loss}");
        Ok(())
    }

    #[test]
    fn ack_delay_should_report_a_peer_that_stopped_answering_as_lossy() -> anyhow::Result<()> {
        let mut est = AckDelay::default();
        let start = std::time::Instant::now();
        let now = feed(&mut est, start, 10, |_| Duration::from_millis(40), |_| false);
        // From here on nothing comes back.
        let now = feed(&mut est, now, 10, |_| Duration::from_millis(40), |_| true);
        let loss = est.congestion(now).loss.context("a peer sent to has a loss share")?;
        assert!(loss > 0.5, "a peer that stopped answering must read as lossy: {loss}");
        Ok(())
    }

    #[test]
    fn ack_delay_should_say_nothing_about_delay_once_the_peer_went_quiet() {
        let mut est = AckDelay::default();
        let now = feed(&mut est, std::time::Instant::now(), 5, |_| Duration::from_millis(40), |_| false);
        assert!(est.congestion(now).queueing.is_some());
        assert!(est.congestion(now + Duration::from_secs(10)).queueing.is_none());
    }

    #[test]
    fn registry_should_time_acknowledgements_against_their_packets() {
        let registry = PeerProtocolCounterRegistry::default();
        let peer = *OffchainKeypair::random().public();
        assert_eq!(registry.congestion(&peer), LinkCongestion::default(), "an unknown peer has no estimate");
        assert!(!registry.ack_timing(), "timing is off unless a node turns it on");
        registry.enable_ack_timing();
        assert!(registry.ack_timing());

        let counters = registry.get_or_create(&peer);
        let mut keys = Vec::new();
        for _ in 0..30 {
            let key = HalfKey::random();
            counters.record_awaiting_ack(key.to_challenge().expect("valid"));
            keys.push(key);
        }
        // Acknowledgements carry the half key; its challenge is what matches the packet.
        let challenges: Vec<_> = keys.iter().map(|k| k.to_challenge().expect("valid")).collect();
        counters.record_acked(&challenges, Instant::now());
        assert_eq!(registry.congestion(&peer).loss, Some(0.0));
    }
}
